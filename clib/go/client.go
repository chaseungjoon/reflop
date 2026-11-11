// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"golang.org/x/crypto/ssh"

	pb "github.com/chaseungjoon/reflop/pkg/reflopv1"
	"github.com/chaseungjoon/reflop/pkg/wire"
)

const maxFrame = 64 << 20

// maxResponseBytes caps one fetch, whatever the server sends (SEC_SWEEP M3).
const maxResponseBytes = 16 << 30

type serverEntry struct {
	Name        string   `json:"name"`
	Transport   string   `json:"transport"`
	Addresses   []string `json:"addresses"`
	Fingerprint string   `json:"fingerprint"`
	Identity    string   `json:"identity"`
	IdentityID  string   `json:"identity_id"`
	SSHPort     *int     `json:"ssh_port"`
	SSHHostKeys []string `json:"ssh_host_keys"`
	Extra       any      `json:"extra"`
}

func configDir() string {
	if h := os.Getenv("REFLOP_CLIENT_HOME"); h != "" {
		return h
	}
	if x := os.Getenv("XDG_CONFIG_HOME"); x != "" {
		return filepath.Join(x, "reflop")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "reflop")
}

func loadServer(name string) (*serverEntry, error) {
	b, err := os.ReadFile(filepath.Join(configDir(), "servers.json"))
	if err != nil {
		return nil, errors.New("no paired servers: run `reflop pair <invite-code>` first")
	}
	var cfg struct {
		Default string                  `json:"default"`
		Servers map[string]*serverEntry `json:"servers"`
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		return nil, fmt.Errorf("servers.json: %w", err)
	}
	if name == "" {
		name = os.Getenv("REFLOP_SERVER")
	}
	if name == "" {
		name = cfg.Default
	}
	e := cfg.Servers[name]
	if e == nil {
		return nil, fmt.Errorf("no paired server named %q", name)
	}
	return e, nil
}

func loadKey() (ed25519.PrivateKey, error) {
	p := filepath.Join(configDir(), "id_ed25519")
	st, err := os.Stat(p)
	if err != nil {
		return nil, fmt.Errorf("%s not found; on macOS the key is in the Keychain: run `reflop key --write-file` once", p)
	}
	if st.Mode().Perm()&0o077 != 0 {
		return nil, fmt.Errorf("%s is readable by other users; chmod 600 it", p)
	}
	b, err := os.ReadFile(p)
	if err != nil {
		return nil, err
	}
	k, err := ssh.ParseRawPrivateKey(b)
	if err != nil {
		return nil, fmt.Errorf("identity key: %w", err)
	}
	switch v := k.(type) {
	case ed25519.PrivateKey:
		return v, nil
	case *ed25519.PrivateKey:
		return *v, nil
	}
	return nil, errors.New("identity key is not Ed25519")
}

type conn struct {
	rw     io.ReadWriteCloser
	r      *wire.Reader
	w      *wire.Writer
	nextID uint64
	caps   *pb.ServerCapabilities
	device *pb.Device
	chunk  int
}

type pipeRW struct {
	io.Reader
	io.WriteCloser
	cmd *exec.Cmd
}

func (p *pipeRW) Close() error { p.WriteCloser.Close(); return p.cmd.Process.Kill() }

func dial(e *serverEntry, fp []byte) (io.ReadWriteCloser, error) {
	if e.Transport == "ssh" {
		bin := os.Getenv("REFLOP_SSH_COMMAND")
		if bin == "" {
			bin = "ssh"
		}
		// The same options as the Python client (SEC_SWEEP L6).
		args := []string{"-T", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", "-o", "IdentitiesOnly=yes",
			"-o", "ForwardAgent=no", "-o", "ForwardX11=no", "-o", "ClearAllForwardings=yes",
			"-o", "PermitLocalCommand=no", "-o", "StrictHostKeyChecking=yes", "-o", "UpdateHostKeys=no"}
		pinned, err := pinnedHostArgs(e)
		if err != nil {
			return nil, err
		}
		args = append(args, pinned...)
		args = append(args, "-i", filepath.Join(configDir(), "id_ed25519"))
		if e.SSHPort != nil {
			args = append(args, "-p", fmt.Sprint(*e.SSHPort))
		}
		if len(e.Addresses) == 0 {
			return nil, errors.New("servers.json: no SSH target")
		}
		args = append(args, "--", e.Addresses[0], "reflop-stdio")
		cmd := exec.Command(bin, args...)
		in, err := cmd.StdinPipe()
		if err != nil {
			return nil, err
		}
		out, err := cmd.StdoutPipe()
		if err != nil {
			return nil, err
		}
		if err := cmd.Start(); err != nil {
			return nil, err
		}
		return &pipeRW{Reader: bufio.NewReader(out), WriteCloser: in, cmd: cmd}, nil
	}
	cfg := &tls.Config{
		MinVersion:         tls.VersionTLS13,
		NextProtos:         []string{"reflop/1"},
		InsecureSkipVerify: true, // replaced by the pinned SPKI check
		VerifyConnection: func(cs tls.ConnectionState) error {
			if len(cs.PeerCertificates) == 0 {
				return errors.New("no server certificate")
			}
			got := sha256.Sum256(cs.PeerCertificates[0].RawSubjectPublicKeyInfo)
			if subtle.ConstantTimeCompare(got[:], fp) != 1 {
				return errors.New("the server presented a different key than the one pinned at pairing; nothing was sent")
			}
			return nil
		},
	}
	var last error
	for _, a := range e.Addresses {
		d := &net.Dialer{Timeout: 15 * time.Second}
		c, err := tls.DialWithDialer(d, "tcp", a, cfg)
		if err == nil {
			return c, nil
		}
		last = err
	}
	return nil, fmt.Errorf("could not reach the server: %v", last)
}

func connect(server, device string) (*conn, error) {
	e, err := loadServer(server)
	if err != nil {
		return nil, err
	}
	key, err := loadKey()
	if err != nil {
		return nil, err
	}
	fp, err := hex.DecodeString(e.Fingerprint)
	if err != nil || len(fp) != 32 {
		return nil, errors.New("servers.json: bad fingerprint")
	}
	rw, err := dial(e, fp)
	if err != nil {
		return nil, err
	}
	c := &conn{rw: rw, r: wire.NewReader(rw, maxFrame+(1<<20)), w: wire.NewWriter(rw, maxFrame), chunk: 4 << 20}
	hr, err := c.call(&pb.Envelope{Body: &pb.Envelope_Hello{Hello: &pb.Hello{ProtocolVersions: []uint32{1},
		OpsetVersions: []uint32{1}, ClientKind: "c", ClientVersion: version}}})
	if err != nil {
		rw.Close()
		return nil, err
	}
	h := hr.GetHelloResponse()
	if h == nil || subtle.ConstantTimeCompare(h.ServerKeyFingerprint, fp) != 1 {
		rw.Close()
		return nil, errors.New("server key mismatch")
	}
	pub := key.Public().(ed25519.PublicKey)
	msg := append(append(append([]byte("reflop-auth-v1\x00"), h.ServerNonce...), fp...), pub...)
	ar, err := c.call(&pb.Envelope{Body: &pb.Envelope_Authenticate{Authenticate: &pb.Authenticate{
		PublicKey: pub, Signature: ed25519.Sign(key, msg)}}})
	if err != nil {
		rw.Close()
		return nil, err
	}
	c.caps = ar.GetAuthResult().GetCapabilities()
	if l := c.caps.GetLimits(); l != nil && l.UploadChunkBytes > 0 {
		c.chunk = int(l.UploadChunkBytes)
	}
	for _, want := range kinds(device) {
		for _, d := range c.caps.GetDevices() {
			if d.Device.Kind == want && c.device == nil {
				c.device = d.Device
			}
		}
	}
	if c.device == nil {
		rw.Close()
		return nil, fmt.Errorf("the server has no %s device", device)
	}
	if _, err := c.call(&pb.Envelope{Body: &pb.Envelope_OpenSession{OpenSession: &pb.OpenSession{Device: c.device}}}); err != nil {
		rw.Close()
		return nil, err
	}
	return c, nil
}

func kinds(device string) []pb.DeviceKind {
	switch device {
	case "cpu":
		return []pb.DeviceKind{pb.DeviceKind_DEVICE_CPU}
	case "cuda":
		return []pb.DeviceKind{pb.DeviceKind_DEVICE_CUDA}
	}
	return []pb.DeviceKind{pb.DeviceKind_DEVICE_CUDA, pb.DeviceKind_DEVICE_CPU}
}

// call sends a request and waits for its (single or stream) response.
func (c *conn) call(env *pb.Envelope) (*pb.Envelope, error) {
	frames, err := c.stream(env)
	if err != nil {
		return nil, err
	}
	return frames[len(frames)-1], nil
}

func (c *conn) stream(env *pb.Envelope) ([]*pb.Envelope, error) {
	c.nextID++
	env.RequestId = c.nextID
	if err := c.w.Write(env); err != nil {
		return nil, err
	}
	var out []*pb.Envelope
	for {
		r := &pb.Envelope{}
		if err := c.r.Read(r); err != nil {
			return nil, fmt.Errorf("connection lost: %w", err)
		}
		if r.GetEvent() != nil || r.RequestId != env.RequestId {
			continue // events are informational; unknown ids are ignored
		}
		if e := r.GetError(); e != nil {
			return nil, fmt.Errorf("[%s] %s", e.Code.String(), sanitize(e.Message))
		}
		out = append(out, r)
		if r.End || r.GetFetchChunk() == nil {
			return out, nil
		}
	}
}

// fetch downloads one tensor. Every chunk is checked against the size its
// metadata declares, so a hostile server cannot make the caller buffer
// without bound or pass off a partial tensor (SEC_SWEEP M3).
func (c *conn) fetch(h []byte) ([]byte, error) {
	c.nextID++
	id := c.nextID
	if err := c.w.Write(&pb.Envelope{RequestId: id, Body: &pb.Envelope_Fetch{Fetch: &pb.Fetch{Handles: [][]byte{h}}}}); err != nil {
		return nil, err
	}
	var data []byte
	want := int64(-1)
	for {
		r := &pb.Envelope{}
		if err := c.r.Read(r); err != nil {
			return nil, fmt.Errorf("connection lost: %w", err)
		}
		if r.GetEvent() != nil || r.RequestId != id {
			continue
		}
		if e := r.GetError(); e != nil {
			return nil, fmt.Errorf("[%s] %s", e.Code.String(), sanitize(e.Message))
		}
		fc := r.GetFetchChunk()
		if fc == nil || fc.Index != 0 {
			return nil, errors.New("the server sent an unexpected fetch response")
		}
		if fc.Meta != nil {
			if want >= 0 {
				return nil, errors.New("the server sent the tensor's metadata twice")
			}
			n, err := tensorBytes(fc.Meta)
			if err != nil {
				return nil, fmt.Errorf("server tensor: %w", err)
			}
			if n > maxResponseBytes {
				return nil, fmt.Errorf("the server's tensor has %d bytes; the limit is %d", n, uint64(maxResponseBytes))
			}
			want = int64(n)
			data = make([]byte, 0, min(n, 64<<20))
		} else if want < 0 {
			return nil, errors.New("the server sent tensor data before its metadata")
		}
		if fc.Offset != uint64(len(data)) {
			return nil, errors.New("the server sent tensor data out of order")
		}
		if int64(len(data))+int64(len(fc.Data)) > want {
			return nil, errors.New("the server sent more data than the tensor's metadata declares")
		}
		data = append(data, fc.Data...)
		if r.End {
			if int64(len(data)) != want {
				return nil, errors.New("the server's fetch response ended before the tensor was complete")
			}
			return data, nil
		}
	}
}

var dtypeSize = map[pb.DType]uint64{
	pb.DType_DTYPE_BOOL: 1, pb.DType_DTYPE_UINT8: 1, pb.DType_DTYPE_INT8: 1, pb.DType_DTYPE_INT16: 2,
	pb.DType_DTYPE_INT32: 4, pb.DType_DTYPE_INT64: 8, pb.DType_DTYPE_FLOAT16: 2, pb.DType_DTYPE_BFLOAT16: 2,
	pb.DType_DTYPE_FLOAT32: 4, pb.DType_DTYPE_FLOAT64: 8, pb.DType_DTYPE_COMPLEX64: 8, pb.DType_DTYPE_COMPLEX128: 16,
	pb.DType_DTYPE_FLOAT8_E4M3FN: 1, pb.DType_DTYPE_FLOAT8_E5M2: 1,
}

// tensorBytes is numel x itemsize, with overflow checks.
func tensorBytes(m *pb.TensorMeta) (uint64, error) {
	size, ok := dtypeSize[m.Dtype]
	if !ok {
		return 0, fmt.Errorf("unknown dtype %d", m.Dtype)
	}
	n := size
	for _, d := range m.Shape {
		if d < 0 {
			return 0, errors.New("negative dimension")
		}
		if d != 0 && n > (1<<62)/uint64(d) {
			return 0, errors.New("tensor size overflows")
		}
		n *= uint64(d)
	}
	return n, nil
}

func (c *conn) newHandle() [16]byte {
	var h [16]byte
	for {
		rand.Read(h[:])
		if h[0] != 0xff {
			return h
		}
	}
}

func (c *conn) upload(meta *pb.TensorMeta, data []byte) ([16]byte, error) {
	h := c.newHandle()
	c.nextID++
	id := c.nextID
	sum := sha256.Sum256(data)
	for off := 0; ; off += c.chunk {
		end := off + c.chunk
		if end > len(data) {
			end = len(data)
		}
		u := &pb.UploadTensor{Handle: h[:], Offset: uint64(off), Data: data[off:end], Final: end == len(data)}
		if off == 0 {
			u.Meta = meta
		}
		if u.Final {
			u.Sha256 = sum[:]
		}
		if err := c.w.Write(&pb.Envelope{RequestId: id, Body: &pb.Envelope_UploadTensor{UploadTensor: u}}); err != nil {
			return h, err
		}
		if u.Final {
			break
		}
	}
	for {
		r := &pb.Envelope{}
		if err := c.r.Read(r); err != nil {
			return h, err
		}
		if r.RequestId != id {
			continue
		}
		if e := r.GetError(); e != nil {
			return h, fmt.Errorf("[%s] %s", e.Code.String(), sanitize(e.Message))
		}
		return h, nil
	}
}

func sanitize(s string) string {
	out := make([]rune, 0, len(s))
	for _, r := range s {
		if r == '\n' || r == '\t' || (r >= 0x20 && r != 0x7f && !(r >= 0x80 && r <= 0x9f)) {
			out = append(out, r)
		}
	}
	if len(out) > 2000 {
		out = out[:2000]
	}
	return string(out)
}

var _ = x509.ParseCertificate
