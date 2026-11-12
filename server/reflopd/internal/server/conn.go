// SPDX-License-Identifier: Apache-2.0

package server

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net"
	"sync"
	"sync/atomic"
	"time"

	pb "github.com/chaseungjoon/reflop/pkg/reflopv1"
	"github.com/chaseungjoon/reflop/pkg/wire"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/audit"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/auth"
)

// ProtocolVersion is reflop.v1.
const ProtocolVersion = 1

var connGen atomic.Uint64

// clientConn is one authenticated (or authenticating) connection.
type clientConn struct {
	srv       *Server
	rw        io.ReadWriteCloser
	r         *wire.Reader
	w         *wire.Writer
	gen       uint64
	transport string // "tls" or "ssh"
	remote    string
	sshID     string // SSH mode: identity id fixed by the forced command
	nonce     []byte
	identity  *auth.Identity
	session   *Session
	closeOnce sync.Once
	done      chan struct{}
	bytesUp   atomic.Uint64
	bytesDown atomic.Uint64
	eventsRID atomic.Uint64
	account   atomic.Pointer[string] // identity name, readable from other goroutines
}

// handleConn serves one connection. adm (from admit) is released by the
// caller; handleConn only gives back its pre-authentication slot.
func (s *Server) handleConn(rw io.ReadWriteCloser, transport, remote, sshID string, adm *admission) {
	c := &clientConn{
		srv: s, rw: rw, transport: transport, remote: remote, sshID: sshID,
		gen:  connGen.Add(1),
		done: make(chan struct{}),
	}
	max := uint32(s.cfg().Limits.MaxFrameBytes)
	// Until AuthResult, frames are capped at a few KiB: an unauthenticated
	// peer cannot make reflopd hold max_frame_bytes per connection (M1).
	c.r = wire.NewReader(&countingReader{rw, &c.bytesUp}, preAuthMaxFrame)
	c.w = wire.NewWriter(&countingWriter{rw, &c.bytesDown}, max)
	defer c.close()
	if err := c.handshake(); err != nil {
		s.logf("connection from %s: %v", remote, err)
		return
	}
	c.r.SetMax(max)
	adm.authenticated()
	s.addConn(c)
	defer s.removeConn(c)
	c.loop()
}

func (c *clientConn) close() {
	c.closeOnce.Do(func() {
		close(c.done)
		c.rw.Close()
		if c.session != nil {
			c.session.detach(c)
		}
		if c.identity != nil {
			c.srv.audit.Write(audit.Record{Identity: c.identity.Name, ID: c.identity.ID, Event: "disconnect",
				Fields: map[string]any{"bytes_up": c.bytesUp.Load(), "bytes_down": c.bytesDown.Load()}})
			c.srv.usage.Transfer(c.identity.Name, c.bytesUp.Load(), c.bytesDown.Load())
		}
	})
}

func (c *clientConn) send(env *pb.Envelope) error { return c.w.Write(env) }

func (c *clientConn) sendErr(rid uint64, code pb.ErrorCode, msg string) {
	_ = c.send(&pb.Envelope{RequestId: rid, End: true, Body: &pb.Envelope_Error{Error: &pb.Error{Code: code, Message: msg}}})
}

func (c *clientConn) read(timeout time.Duration) (*pb.Envelope, error) {
	if d, ok := c.rw.(interface{ SetReadDeadline(time.Time) error }); ok {
		if timeout > 0 {
			d.SetReadDeadline(time.Now().Add(timeout))
		} else {
			d.SetReadDeadline(time.Time{})
		}
	}
	b, err := c.r.ReadFrame()
	if err != nil {
		return nil, err
	}
	env := &pb.Envelope{}
	if err := wire.Unmarshal(b, env); err != nil {
		return nil, fmt.Errorf("%w: %v", errDecode, err)
	}
	if err := wire.CheckEnums(env.ProtoReflect()); err != nil {
		return nil, fmt.Errorf("%w: %v", errDecode, err)
	}
	return env, nil
}

var errDecode = errors.New("malformed message")

// handshake: Hello → HelloResponse → (Pair →) Authenticate → AuthResult.
func (c *clientConn) handshake() error {
	env, err := c.read(15 * time.Second)
	if err != nil {
		return err
	}
	h := env.GetHello()
	if h == nil {
		c.sendErr(env.RequestId, pb.ErrorCode_ERROR_PROTOCOL, "expected Hello")
		return errors.New("first message was not Hello")
	}
	if !contains(h.ProtocolVersions, ProtocolVersion) || !contains(h.OpsetVersions, 1) {
		c.sendErr(env.RequestId, pb.ErrorCode_ERROR_UNSUPPORTED_VERSION,
			fmt.Sprintf("this server speaks protocol %d and opset 1", ProtocolVersion))
		return errors.New("unsupported client version")
	}
	c.nonce = make([]byte, 32)
	if _, err := rand.Read(c.nonce); err != nil {
		return err
	}
	fp := c.srv.key.Fingerprint
	if err := c.send(&pb.Envelope{RequestId: env.RequestId, End: true, Body: &pb.Envelope_HelloResponse{
		HelloResponse: &pb.HelloResponse{ProtocolVersion: ProtocolVersion, ServerNonce: c.nonce, ServerKeyFingerprint: fp[:]}}}); err != nil {
		return err
	}
	for attempts := 0; attempts < 3; attempts++ {
		env, err = c.read(30 * time.Second)
		if err != nil {
			return err
		}
		switch b := env.Body.(type) {
		case *pb.Envelope_Pair:
			if c.transport != "tls" {
				c.sendErr(env.RequestId, pb.ErrorCode_ERROR_PERMISSION_DENIED, "pairing is only available in daemon mode")
				return errors.New("pair over ssh")
			}
			id, err := c.srv.ids.Redeem(b.Pair, c.nonce, fp[:])
			if err != nil {
				fields := map[string]any{"remote": c.remote}
				var ie *auth.InviteError
				if errors.As(err, &ie) {
					fields["reason"] = ie.Reason
				}
				c.auditFailure("pair_failed", fields)
				msg := auth.ErrInvite.Error()
				if !errors.Is(err, auth.ErrInvite) {
					msg = err.Error()
				}
				c.sendErr(env.RequestId, pb.ErrorCode_ERROR_INVITE, msg)
				return err
			}
			c.srv.audit.Write(audit.Record{Identity: id.Name, ID: id.ID, Event: "paired", Fields: map[string]any{"remote": c.remote}})
			c.srv.logf("paired a new key for %q from %s", id.Name, c.remote)
			if err := c.send(&pb.Envelope{RequestId: env.RequestId, End: true, Body: &pb.Envelope_PairResult{
				PairResult: &pb.PairResult{IdentityName: id.Name, IdentityId: id.IDBytes()}}}); err != nil {
				return err
			}
		case *pb.Envelope_Authenticate:
			id, err := c.srv.ids.Authenticate(b.Authenticate, c.nonce, fp[:], c.sshID)
			if err != nil {
				c.auditFailure("auth_failed", map[string]any{"remote": c.remote, "transport": c.transport})
				time.Sleep(500 * time.Millisecond)
				c.sendErr(env.RequestId, pb.ErrorCode_ERROR_UNAUTHENTICATED, "authentication failed")
				return err
			}
			c.identity = id
			c.account.Store(&id.Name)
			caps := c.srv.capabilitiesFor(id)
			c.srv.audit.Write(audit.Record{Identity: id.Name, ID: id.ID, Event: "connect",
				Fields: map[string]any{"remote": c.remote, "transport": c.transport}})
			return c.send(&pb.Envelope{RequestId: env.RequestId, End: true, Body: &pb.Envelope_AuthResult{
				AuthResult: &pb.AuthResult{IdentityName: id.Name, IdentityId: id.IDBytes(), Capabilities: caps}}})
		default:
			c.sendErr(env.RequestId, pb.ErrorCode_ERROR_UNAUTHENTICATED, "authenticate first")
			return errors.New("request before authentication")
		}
	}
	return errors.New("too many handshake messages")
}

// auditFailure writes a failed-handshake record, rate-limited per source
// address and globally (L8). Suppressed records are counted in the next one.
func (c *clientConn) auditFailure(event string, fields map[string]any) {
	ok, suppressed := c.srv.fails.allow(c.remote, time.Now())
	if !ok {
		return
	}
	if suppressed > 0 {
		fields["suppressed_before"] = suppressed
	}
	c.srv.audit.Write(audit.Record{Identity: "?", Event: event, Fields: fields})
}

func contains(xs []uint32, v uint32) bool {
	for _, x := range xs {
		if x == v {
			return true
		}
	}
	return false
}

// loop routes requests until the connection ends.
func (c *clientConn) loop() {
	for {
		env, err := c.read(0)
		if err != nil {
			if errors.Is(err, errDecode) || errors.Is(err, wire.ErrFrameTooLarge) {
				c.sendErr(0, pb.ErrorCode_ERROR_PROTOCOL, err.Error())
				c.srv.audit.Write(audit.Record{Identity: c.identity.Name, ID: c.identity.ID, Event: "validation_failed",
					Fields: map[string]any{"stage": "V0/V1", "error": err.Error()}})
			}
			return
		}
		if !c.srv.ids.IsActive(c.identity.ID) {
			c.sendErr(env.RequestId, pb.ErrorCode_ERROR_UNAUTHENTICATED, "this identity was revoked")
			return
		}
		if err := c.route(env); err != nil {
			c.srv.logf("%s: %v", c.identity.Name, err)
			return
		}
	}
}

func (c *clientConn) route(env *pb.Envelope) error {
	rid := env.RequestId
	switch b := env.Body.(type) {
	case *pb.Envelope_Ping:
		return c.send(&pb.Envelope{RequestId: rid, End: true, Body: &pb.Envelope_Pong{Pong: &pb.Pong{Nonce: b.Ping.Nonce}}})
	case *pb.Envelope_GetUsage:
		return c.send(&pb.Envelope{RequestId: rid, End: true, Body: &pb.Envelope_UsageReport{UsageReport: c.srv.guestUsage(c.identity.Name)}})
	case *pb.Envelope_SubscribeEvents:
		c.eventsRID.Store(rid)
		c.srv.events.subscribe(c)
		return nil
	case *pb.Envelope_OpenSession:
		return c.openSession(rid, b.OpenSession)
	case *pb.Envelope_CloseSession:
		if c.session != nil {
			c.session.close("closed by client")
			c.session = nil
		}
		return c.send(&pb.Envelope{RequestId: rid, End: true, Body: &pb.Envelope_SessionClosed{SessionClosed: &pb.SessionClosed{}}})
	case *pb.Envelope_PackBegin, *pb.Envelope_PackChunk, *pb.Envelope_PackCommit, *pb.Envelope_ListDatasets,
		*pb.Envelope_DeleteDataset:
		return c.dataRequest(env)
	case *pb.Envelope_ListSnapshots, *pb.Envelope_DeleteSnapshot:
		return c.snapshotAdmin(env)
	case *pb.Envelope_RegisterGraph, *pb.Envelope_Execute, *pb.Envelope_UploadTensor, *pb.Envelope_Fetch,
		*pb.Envelope_Free, *pb.Envelope_Snapshot, *pb.Envelope_Restore, *pb.Envelope_Reseed:
		if c.session == nil {
			c.sendErr(rid, pb.ErrorCode_ERROR_UNKNOWN_SESSION, "open a session first")
			return nil
		}
		return c.session.forward(c, env)
	default:
		c.sendErr(rid, pb.ErrorCode_ERROR_PROTOCOL, "unexpected message")
		return errors.New("unexpected client message")
	}
}

func (c *clientConn) openSession(rid uint64, req *pb.OpenSession) error {
	if c.session != nil {
		c.sendErr(rid, pb.ErrorCode_ERROR_PROTOCOL, "this connection already has a session")
		return nil
	}
	if len(req.ResumeSessionId) > 0 {
		sess, err := c.srv.resume(c, req.ResumeSessionId)
		if err != nil {
			c.sendErr(rid, pb.ErrorCode_ERROR_UNKNOWN_SESSION, err.Error())
			return nil
		}
		c.session = sess
		return c.send(&pb.Envelope{RequestId: rid, End: true, Body: &pb.Envelope_SessionOpened{SessionOpened: &pb.SessionOpened{
			SessionId: sess.ID[:], Resumed: true, Numerics: sess.numerics, Seed: sess.seed,
			EnabledOpIds: c.srv.filterOps(sess.Device)}}})
	}
	sess, err := c.srv.openSession(c, req)
	if err != nil {
		var se *statusError
		if errors.As(err, &se) {
			c.sendErr(rid, se.code, se.msg)
			return nil
		}
		c.sendErr(rid, pb.ErrorCode_ERROR_INTERNAL, err.Error())
		return nil
	}
	c.session = sess
	return c.send(&pb.Envelope{RequestId: rid, End: true, Body: &pb.Envelope_SessionOpened{SessionOpened: &pb.SessionOpened{
		SessionId: sess.ID[:], Numerics: sess.numerics, Seed: sess.seed, EnabledOpIds: c.srv.filterOps(sess.Device)}}})
}

type statusError struct {
	code pb.ErrorCode
	msg  string
}

func (e *statusError) Error() string { return e.msg }

func statusf(code pb.ErrorCode, f string, a ...any) error {
	return &statusError{code: code, msg: fmt.Sprintf(f, a...)}
}

type countingReader struct {
	r io.Reader
	n *atomic.Uint64
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n.Add(uint64(n))
	return n, err
}

type countingWriter struct {
	w io.Writer
	n *atomic.Uint64
}

func (c *countingWriter) Write(p []byte) (int, error) {
	n, err := c.w.Write(p)
	c.n.Add(uint64(n))
	return n, err
}

func hexID(b []byte) string { return hex.EncodeToString(b) }

var _ net.Conn = (*net.TCPConn)(nil)
