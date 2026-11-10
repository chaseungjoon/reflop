// SPDX-License-Identifier: Apache-2.0

// Package sshpipe manages SSH pipe mode (PLAN.md §4.4, §15.4): the Guest's
// key goes into authorized_keys with restrict and a forced command, so SSH
// is a pipe that can only run `reflopd serve --stdio`, never a shell.
package sshpipe

import (
	"bufio"
	"bytes"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/chaseungjoon/reflop/server/reflopd/internal/config"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/paths"
)

// Marker tags reflop-managed lines.
const Marker = "reflop-identity="

// ParseEd25519 parses "ssh-ed25519 AAAA… [comment]".
func ParseEd25519(line string) (ed25519.PublicKey, string, error) {
	f := strings.Fields(strings.TrimSpace(line))
	if len(f) < 2 || f[0] != "ssh-ed25519" {
		return nil, "", errors.New("expected an ssh-ed25519 public key (from `reflop key --ssh`)")
	}
	raw, err := base64.StdEncoding.DecodeString(f[1])
	if err != nil {
		return nil, "", errors.New("bad base64 in the public key")
	}
	readStr := func(b []byte) ([]byte, []byte, error) {
		if len(b) < 4 {
			return nil, nil, errors.New("truncated key")
		}
		n := binary.BigEndian.Uint32(b)
		if uint32(len(b)-4) < n {
			return nil, nil, errors.New("truncated key")
		}
		return b[4 : 4+n], b[4+n:], nil
	}
	typ, rest, err := readStr(raw)
	if err != nil || string(typ) != "ssh-ed25519" {
		return nil, "", errors.New("not an ed25519 key")
	}
	pub, rest, err := readStr(rest)
	if err != nil || len(pub) != ed25519.PublicKeySize || len(rest) != 0 {
		return nil, "", errors.New("malformed ed25519 key")
	}
	comment := ""
	if len(f) > 2 {
		comment = strings.Join(f[2:], " ")
	}
	return ed25519.PublicKey(pub), comment, nil
}

// AuthorizedKeysLine builds the exact restricted line.
func AuthorizedKeysLine(reflopd, identityID, key string, l paths.Layout) string {
	f := strings.Fields(key)
	env := ""
	if h := os.Getenv("REFLOP_HOME"); h != "" {
		env = "REFLOP_HOME=" + h + " "
	}
	cmd := fmt.Sprintf("%s%s serve --stdio --identity %s", env, reflopd, identityID)
	return fmt.Sprintf(`restrict,command="%s" %s %s %s%s`, strings.ReplaceAll(cmd, `"`, `\"`), f[0], f[1], Marker, identityID)
}

// AuthorizedKeysPath is the Owner's (or the reflop user's) authorized_keys.
func AuthorizedKeysPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ssh", "authorized_keys"), nil
}

// Append adds a line atomically, keeping the file's mode.
func Append(path, line string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	old, err := os.ReadFile(path)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if bytes.Contains(old, []byte(line)) {
		return nil
	}
	if len(old) > 0 && !bytes.HasSuffix(old, []byte("\n")) {
		old = append(old, '\n')
	}
	return config.WriteFileAtomic(path, append(old, []byte(line+"\n")...), 0o600)
}

// RemoveIdentity deletes the managed lines for an identity.
func RemoveIdentity(path, identityID string) (int, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	var keep []string
	n := 0
	for _, l := range strings.Split(strings.TrimRight(string(b), "\n"), "\n") {
		if strings.Contains(l, Marker+identityID) {
			n++
			continue
		}
		keep = append(keep, l)
	}
	if n == 0 {
		return 0, nil
	}
	return n, config.WriteFileAtomic(path, []byte(strings.Join(keep, "\n")+"\n"), 0o600)
}

// Finding is a doctor result.
type Finding struct {
	Line     int
	Critical bool
	Msg      string
}

// Scan reports reflop-managed keys missing restrict or command=.
func Scan(path string) ([]Finding, error) {
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	var out []Finding
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	i := 0
	for sc.Scan() {
		i++
		l := strings.TrimSpace(sc.Text())
		if !strings.Contains(l, Marker) {
			continue
		}
		opts := ""
		if !strings.HasPrefix(l, "ssh-") {
			opts, _, _ = strings.Cut(l, " ")
		}
		if !hasOpt(opts, "restrict") {
			out = append(out, Finding{i, true, "reflop key without `restrict`: this key can open a shell"})
		}
		if !strings.Contains(opts, `command="`) || !strings.Contains(opts, "serve --stdio --identity") {
			out = append(out, Finding{i, true, "reflop key without the forced command: this key can run anything"})
		}
	}
	return out, sc.Err()
}

func hasOpt(opts, name string) bool {
	for _, o := range strings.Split(opts, ",") {
		if o == name {
			return true
		}
	}
	return false
}
