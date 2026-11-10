// SPDX-License-Identifier: Apache-2.0

package auth

import (
	"bytes"
	"crypto/ed25519"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base32"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	pb "github.com/chaseungjoon/reflop/pkg/reflopv1"
	"github.com/chaseungjoon/reflop/pkg/wire"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/config"
)

// NameRe constrains identity names.
var NameRe = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]{0,31}$`)

// Key is one registered public key.
type Key struct {
	Public  []byte    `json:"public"` // 32-byte Ed25519
	Added   time.Time `json:"added"`
	Via     string    `json:"via"` // "tls" or "ssh"
	Comment string    `json:"comment,omitempty"`
}

// Identity is one paired Guest.
type Identity struct {
	ID      string    `json:"id"` // 32 hex chars
	Name    string    `json:"name"`
	Keys    []Key     `json:"keys"`
	Created time.Time `json:"created"`
	Revoked bool      `json:"revoked,omitempty"`
}

// IDBytes returns the 16-byte identity id.
func (i *Identity) IDBytes() []byte { b, _ := hex.DecodeString(i.ID); return b }

// Invite is a pending one-time pairing secret.
type Invite struct {
	ID      string    `json:"id"`     // 32 hex
	Secret  []byte    `json:"secret"` // 32 bytes
	Name    string    `json:"name"`
	Expires time.Time `json:"expires"`
	Used    bool      `json:"used,omitempty"`
}

type db struct {
	Identities []*Identity `json:"identities"`
	Invites    []*Invite   `json:"invites"`
}

// Store persists identities and invites in one 0600 JSON file. All
// methods are safe for concurrent use; another reflopd command (e.g.
// `reflopd revoke`) changes the file and the daemon reloads on notice.
type Store struct {
	mu    sync.Mutex
	path  string
	db    db
	mtime time.Time
	size  int64
}

// Open loads (or initializes) the store.
func Open(dir string) (*Store, error) {
	s := &Store{path: filepath.Join(dir, "identities.json")}
	return s, s.Reload()
}

// Reload re-reads the file.
func (s *Store) Reload() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.reloadLocked()
}

// refreshLocked reloads when another reflopd command changed the file
// (invite, revoke, ssh key registration).
func (s *Store) refreshLocked() {
	st, err := os.Stat(s.path)
	if err != nil || (st.ModTime().Equal(s.mtime) && st.Size() == s.size) {
		return
	}
	_ = s.reloadLocked()
}

func (s *Store) reloadLocked() error {
	if st, err := os.Stat(s.path); err == nil {
		s.mtime, s.size = st.ModTime(), st.Size()
	}
	b, err := os.ReadFile(s.path)
	if errors.Is(err, os.ErrNotExist) {
		s.db = db{}
		return nil
	}
	if err != nil {
		return err
	}
	var d db
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&d); err != nil {
		return fmt.Errorf("%s: %w", s.path, err)
	}
	s.db = d
	return nil
}

func (s *Store) saveLocked() error {
	b, err := json.MarshalIndent(&s.db, "", "  ")
	if err != nil {
		return err
	}
	if err := config.WriteFileAtomic(s.path, append(b, '\n'), 0o600); err != nil {
		return err
	}
	if st, err := os.Stat(s.path); err == nil {
		s.mtime, s.size = st.ModTime(), st.Size()
	}
	return nil
}

func randHex(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		panic(err)
	}
	return hex.EncodeToString(b)
}

// CreateInvite records a one-time invite for name.
func (s *Store) CreateInvite(name string, ttl time.Duration) (*Invite, error) {
	if !NameRe.MatchString(name) {
		return nil, fmt.Errorf("invalid name %q: use 1-32 of a-z 0-9 _ - (starting with a letter or digit)", name)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.refreshLocked()
	for _, id := range s.db.Identities {
		if id.Name == name && id.Revoked {
			return nil, fmt.Errorf("identity %q was revoked; choose another name", name)
		}
	}
	secret := make([]byte, 32)
	if _, err := rand.Read(secret); err != nil {
		return nil, err
	}
	inv := &Invite{ID: randHex(16), Secret: secret, Name: name, Expires: time.Now().Add(ttl).UTC().Truncate(time.Second)}
	s.pruneLocked()
	s.db.Invites = append(s.db.Invites, inv)
	return inv, s.saveLocked()
}

func (s *Store) pruneLocked() {
	now := time.Now()
	kept := s.db.Invites[:0]
	for _, inv := range s.db.Invites {
		if !inv.Used && now.Before(inv.Expires) {
			kept = append(kept, inv)
		}
	}
	s.db.Invites = kept
}

// ErrInvite covers every invite failure; callers must not reveal which.
var ErrInvite = errors.New("invalid, expired or already used invite")

// Redeem verifies a Pair request and registers the key. The invite is
// consumed on the first attempt that names it, successful or not, so a
// guessed or stolen invite id cannot be brute-forced.
func (s *Store) Redeem(req *pb.Pair, nonce, serverFP []byte) (*Identity, error) {
	if len(req.InviteId) != 16 || len(req.PublicKey) != ed25519.PublicKeySize || len(req.Mac) != 32 || len(req.Signature) != ed25519.SignatureSize {
		return nil, ErrInvite
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.refreshLocked()
	var inv *Invite
	idHex := hex.EncodeToString(req.InviteId)
	for _, x := range s.db.Invites {
		if subtle.ConstantTimeCompare([]byte(x.ID), []byte(idHex)) == 1 {
			inv = x
		}
	}
	if inv == nil || inv.Used || time.Now().After(inv.Expires) {
		return nil, ErrInvite
	}
	inv.Used = true
	msg := PairMessage(nonce, serverFP, req.InviteId, req.PublicKey)
	mac := hmac.New(sha256.New, inv.Secret)
	mac.Write(msg)
	if !hmac.Equal(mac.Sum(nil), req.Mac) || !ed25519.Verify(req.PublicKey, msg, req.Signature) {
		_ = s.saveLocked()
		return nil, ErrInvite
	}
	if owner := s.byKeyLocked(req.PublicKey); owner != nil {
		_ = s.saveLocked()
		return nil, fmt.Errorf("this key is already registered to %q", owner.Name)
	}
	id := s.byNameLocked(inv.Name)
	if id == nil {
		id = &Identity{ID: randHex(16), Name: inv.Name, Created: time.Now().UTC().Truncate(time.Second)}
		s.db.Identities = append(s.db.Identities, id)
	}
	id.Keys = append(id.Keys, Key{Public: req.PublicKey, Added: time.Now().UTC().Truncate(time.Second), Via: "tls"})
	s.pruneLocked()
	return id, s.saveLocked()
}

// AddKey registers a key directly (SSH pipe mode: the Owner pastes the
// Guest's public key).
func (s *Store) AddKey(name string, pub []byte, via, comment string) (*Identity, error) {
	if !NameRe.MatchString(name) {
		return nil, fmt.Errorf("invalid name %q", name)
	}
	if len(pub) != ed25519.PublicKeySize {
		return nil, errors.New("public key must be Ed25519")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.refreshLocked()
	if owner := s.byKeyLocked(pub); owner != nil {
		if owner.Name == name {
			return owner, nil
		}
		return nil, fmt.Errorf("this key is already registered to %q", owner.Name)
	}
	id := s.byNameLocked(name)
	if id != nil && id.Revoked {
		return nil, fmt.Errorf("identity %q was revoked; choose another name", name)
	}
	if id == nil {
		id = &Identity{ID: randHex(16), Name: name, Created: time.Now().UTC().Truncate(time.Second)}
		s.db.Identities = append(s.db.Identities, id)
	}
	id.Keys = append(id.Keys, Key{Public: pub, Added: time.Now().UTC().Truncate(time.Second), Via: via, Comment: comment})
	return id, s.saveLocked()
}

func (s *Store) byKeyLocked(pub []byte) *Identity {
	for _, id := range s.db.Identities {
		for _, k := range id.Keys {
			if subtle.ConstantTimeCompare(k.Public, pub) == 1 {
				return id
			}
		}
	}
	return nil
}

func (s *Store) byNameLocked(name string) *Identity {
	for _, id := range s.db.Identities {
		if id.Name == name {
			return id
		}
	}
	return nil
}

// ErrAuth covers every authentication failure.
var ErrAuth = errors.New("authentication failed")

// Authenticate checks a signature and returns the key's identity. If
// wantID is non-empty (SSH mode), the key must belong to that identity.
func (s *Store) Authenticate(req *pb.Authenticate, nonce, serverFP []byte, wantID string) (*Identity, error) {
	if len(req.PublicKey) != ed25519.PublicKeySize || len(req.Signature) != ed25519.SignatureSize {
		return nil, ErrAuth
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.refreshLocked()
	id := s.byKeyLocked(req.PublicKey)
	if id == nil || id.Revoked {
		return nil, ErrAuth
	}
	if wantID != "" && id.ID != wantID {
		return nil, ErrAuth
	}
	if !ed25519.Verify(req.PublicKey, AuthMessage(nonce, serverFP, req.PublicKey), req.Signature) {
		return nil, ErrAuth
	}
	c := *id
	return &c, nil
}

// Revoke marks an identity revoked and drops its keys.
func (s *Store) Revoke(name string) (*Identity, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.refreshLocked()
	id := s.byNameLocked(name)
	if id == nil {
		return nil, fmt.Errorf("no identity named %q", name)
	}
	id.Revoked = true
	removed := *id
	id.Keys = nil
	for i, inv := range s.db.Invites {
		if inv.Name == name {
			s.db.Invites[i].Used = true
		}
	}
	return &removed, s.saveLocked()
}

// List returns all identities sorted by name.
func (s *Store) List() []Identity {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.refreshLocked()
	out := make([]Identity, 0, len(s.db.Identities))
	for _, id := range s.db.Identities {
		out = append(out, *id)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// Get returns one identity by name.
func (s *Store) Get(name string) (Identity, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.refreshLocked()
	if id := s.byNameLocked(name); id != nil {
		return *id, true
	}
	return Identity{}, false
}

// IsActive reports whether identity id exists and is not revoked.
func (s *Store) IsActive(idHex string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.refreshLocked()
	for _, id := range s.db.Identities {
		if id.ID == idHex {
			return !id.Revoked
		}
	}
	return false
}

// InvitePrefix starts every invite code.
const InvitePrefix = "reflop1-"

var b32 = base32.StdEncoding.WithPadding(base32.NoPadding)

// EncodeInvite renders an invite code.
func EncodeInvite(c *pb.InviteCode) (string, error) {
	b, err := wire.Marshal(c)
	if err != nil {
		return "", err
	}
	return InvitePrefix + strings.ToLower(b32.EncodeToString(b)), nil
}

// DecodeInvite parses an invite code strictly.
func DecodeInvite(s string) (*pb.InviteCode, error) {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, InvitePrefix) {
		return nil, errors.New("not a reflop invite code")
	}
	raw, err := b32.DecodeString(strings.ToUpper(strings.TrimPrefix(s, InvitePrefix)))
	if err != nil {
		return nil, fmt.Errorf("invite code: %w", err)
	}
	c := &pb.InviteCode{}
	if err := wire.Unmarshal(raw, c); err != nil {
		return nil, err
	}
	return c, nil
}
