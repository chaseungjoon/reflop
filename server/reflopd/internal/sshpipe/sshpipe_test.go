// SPDX-License-Identifier: Apache-2.0

package sshpipe

import (
	"os"
	"path/filepath"
	"testing"
)

const pub = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILs7graug4rm/Iz4hvKqazO5MLF8HcSzVedtFmkYjbTH guest@mac"

func TestParseAndLine(t *testing.T) {
	k, c, err := ParseEd25519(pub)
	if err != nil || len(k) != 32 || c != "guest@mac" {
		t.Fatalf("parse: %v %d %q", err, len(k), c)
	}
	if _, _, err := ParseEd25519("ssh-rsa AAAAB3Nza"); err == nil {
		t.Fatal("rsa accepted")
	}
}

func TestScan(t *testing.T) {
	dir := t.TempDir()
	ak := filepath.Join(dir, "authorized_keys")
	id := "09eff0b86efa2d80abf75f7dff02589f"
	good := `restrict,command="/usr/bin/reflopd serve --stdio --identity ` + id + `" ` + pub + " " + Marker + id
	noRestrict := `command="/usr/bin/reflopd serve --stdio --identity ` + id + `" ` + pub + " " + Marker + id
	noCommand := "restrict " + pub + " " + Marker + id
	bare := pub + " " + Marker + id
	other := "ssh-ed25519 AAAAOTHER owner@laptop"
	os.WriteFile(ak, []byte(good+"\n"+noRestrict+"\n"+noCommand+"\n"+bare+"\n"+other+"\n"), 0o600)
	fs, err := Scan(ak)
	if err != nil {
		t.Fatal(err)
	}
	byLine := map[int]int{}
	for _, f := range fs {
		byLine[f.Line]++
	}
	if byLine[1] != 0 || byLine[2] == 0 || byLine[3] == 0 || byLine[4] < 2 || byLine[5] != 0 {
		t.Fatalf("findings %v", fs)
	}
	n, err := RemoveIdentity(ak, id)
	if err != nil || n != 4 {
		t.Fatalf("remove: %d %v", n, err)
	}
	b, _ := os.ReadFile(ak)
	if string(b) != other+"\n" {
		t.Fatalf("left %q", b)
	}
}
