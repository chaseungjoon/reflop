// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"os"
	"path/filepath"

	"golang.org/x/crypto/ssh"

	"github.com/chaseungjoon/reflop/pkg/sshhost"
)

// pinnedHostArgs returns the ssh options that make ssh trust only the host
// keys recorded at pairing (SEC_SWEEP L6). The keys go into a known_hosts
// file owned by reflop under an alias tied to the server's reflop key, so
// the user's own known_hosts and ssh_config cannot vouch for an impostor.
// Servers paired before 0.1.6 have no recorded keys: ssh then checks the
// user's known_hosts with StrictHostKeyChecking=yes.
func pinnedHostArgs(e *serverEntry) ([]string, error) {
	if len(e.SSHHostKeys) == 0 {
		return nil, nil
	}
	alias := sshhost.Alias(e.Fingerprint)
	var buf bytes.Buffer
	for _, line := range e.SSHHostKeys {
		typ, blob, err := sshhost.Parse(line)
		if err != nil {
			return nil, fmt.Errorf("servers.json: SSH host key: %w", err)
		}
		if _, err := ssh.ParsePublicKey(blob); err != nil {
			return nil, fmt.Errorf("servers.json: SSH host key: %w", err)
		}
		fmt.Fprintf(&buf, "%s %s %s\n", alias, typ, base64.StdEncoding.EncodeToString(blob))
	}
	dir := filepath.Join(configDir(), "ssh_known_hosts")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, err
	}
	path := filepath.Join(dir, e.Fingerprint)
	if old, err := os.ReadFile(path); err != nil || !bytes.Equal(old, buf.Bytes()) {
		if err := writeAtomic(path, buf.Bytes()); err != nil {
			return nil, err
		}
	}
	q, err := sshhost.QuoteOption(path)
	if err != nil {
		return nil, err
	}
	return []string{"-o", "UserKnownHostsFile=" + q, "-o", "GlobalKnownHostsFile=/dev/null",
		"-o", "HostKeyAlias=" + alias, "-o", "CheckHostIP=no"}, nil
}

func writeAtomic(path string, data []byte) error {
	f, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	if _, err := f.Write(data); err != nil {
		f.Close()
		return err
	}
	if err := f.Chmod(0o600); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(f.Name(), path)
}
