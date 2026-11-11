// SPDX-License-Identifier: Apache-2.0

// Package sshhost handles the sshd host keys that SSH invites carry, so the
// Guest's ssh trusts exactly the host the Owner paired them with
// (SEC_SWEEP L6). reflopd reads them from /etc/ssh; libreflop writes them
// into a reflop-owned known_hosts file. The Python client mirrors this.
package sshhost

import (
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Types are the host key algorithms accepted in invites, most preferred
// first. RSA keys are long, so they are used only when nothing else exists.
var Types = []string{"ssh-ed25519", "ecdsa-sha2-nistp256", "ecdsa-sha2-nistp384", "ecdsa-sha2-nistp521", "ssh-rsa"}

// MaxKeys bounds how many host keys an invite may carry.
const MaxKeys = 4

func allowed(t string) bool {
	for _, x := range Types {
		if x == t {
			return true
		}
	}
	return false
}

// Parse checks a "type base64 [comment]" line and returns the type and the
// key blob. The blob's embedded type must match.
func Parse(line string) (string, []byte, error) {
	f := strings.Fields(line)
	if len(f) < 2 {
		return "", nil, errors.New("expected \"type base64\"")
	}
	if !allowed(f[0]) {
		return "", nil, fmt.Errorf("unsupported host key type %q", f[0])
	}
	blob, err := base64.StdEncoding.DecodeString(f[1])
	if err != nil || len(blob) < 4 || len(blob) > 2048 {
		return "", nil, errors.New("malformed host key")
	}
	n := binary.BigEndian.Uint32(blob)
	if uint64(n) > uint64(len(blob)-4) || string(blob[4:4+n]) != f[0] {
		return "", nil, errors.New("host key type does not match its data")
	}
	return f[0], blob, nil
}

// Alias is the HostKeyAlias for a server, from its reflop key fingerprint
// (hex). Tying the alias to the reflop key means a known_hosts entry for
// one paired server never vouches for another.
func Alias(fingerprintHex string) string { return "reflop-" + fingerprintHex }

// QuoteOption quotes a path for an ssh -o value: double quotes keep spaces,
// and %% keeps ssh from expanding % tokens.
func QuoteOption(path string) (string, error) {
	if strings.ContainsAny(path, "\"\n\r") {
		return "", fmt.Errorf("unsupported character in path %q", path)
	}
	return `"` + strings.ReplaceAll(path, "%", "%%") + `"`, nil
}

// LocalHostKeys reads sshd's public host keys from the given glob (default
// /etc/ssh/ssh_host_*_key.pub), returning "type base64" lines, preferred
// types first. RSA is included only if there is no other key.
func LocalHostKeys(pattern string) ([]string, error) {
	if pattern == "" {
		pattern = "/etc/ssh/ssh_host_*_key.pub"
	}
	files, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}
	rank := map[string]int{}
	for i, t := range Types {
		rank[t] = i
	}
	var keys []string
	for _, p := range files {
		b, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		typ, blob, err := Parse(string(b))
		if err != nil {
			continue
		}
		keys = append(keys, typ+" "+base64.StdEncoding.EncodeToString(blob))
	}
	sort.SliceStable(keys, func(i, j int) bool {
		return rank[strings.Fields(keys[i])[0]] < rank[strings.Fields(keys[j])[0]]
	})
	var out []string
	for _, k := range keys {
		if strings.HasPrefix(k, "ssh-rsa ") && len(out) > 0 {
			continue
		}
		out = append(out, k)
		if len(out) == MaxKeys {
			break
		}
	}
	return out, nil
}
