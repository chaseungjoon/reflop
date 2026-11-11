// SPDX-License-Identifier: Apache-2.0

package sshpipe

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/chaseungjoon/reflop/server/reflopd/internal/paths"
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

// SEC_SWEEP I1: the forced command goes through the Owner's shell, so a
// reflopd path or REFLOP_HOME with spaces, quotes or $(...) must reach
// reflopd as one argument and never run anything.
func TestForcedCommandQuoting(t *testing.T) {
	dir := t.TempDir()
	weird := filepath.Join(dir, `my "dir" $(touch pwned) it's`)
	if err := os.MkdirAll(weird, 0o755); err != nil {
		t.Fatal(err)
	}
	exe := filepath.Join(weird, "reflopd")
	out := filepath.Join(dir, "args")
	script := "#!/bin/sh\nprintf '%s\\n' \"$REFLOP_HOME\" \"$@\" > " + out + "\n"
	if err := os.WriteFile(exe, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("REFLOP_HOME", weird+"; touch pwned2")
	id := "09eff0b86efa2d80abf75f7dff02589f"
	line, err := AuthorizedKeysLine(exe, id, pub, paths.Layout{})
	if err != nil {
		t.Fatal(err)
	}
	// What sshd does: take command="...", turning \" into ".
	opts := optionsField(line)
	start := strings.Index(opts, `command="`) + len(`command="`)
	var cmd strings.Builder
	for i := start; i < len(opts); i++ {
		if opts[i] == '\\' && i+1 < len(opts) && opts[i+1] == '"' {
			cmd.WriteByte('"')
			i++
			continue
		}
		if opts[i] == '"' {
			break
		}
		cmd.WriteByte(opts[i])
	}
	c := exec.Command("/bin/sh", "-c", cmd.String())
	c.Dir = dir
	if b, err := c.CombinedOutput(); err != nil {
		t.Fatalf("forced command failed: %v\n%s\ncommand: %s", err, b, cmd.String())
	}
	got, _ := os.ReadFile(out)
	want := weird + "; touch pwned2\nserve\n--stdio\n--identity\n" + id + "\n"
	if string(got) != want {
		t.Fatalf("reflopd saw:\n%s\nwant:\n%s", got, want)
	}
	for _, f := range []string{"pwned", "pwned2"} {
		if _, err := os.Stat(filepath.Join(dir, f)); err == nil {
			t.Fatalf("the forced command ran injected code (%s)", f)
		}
	}
	if fs, _ := scanLine(t, line); len(fs) != 0 {
		t.Fatalf("doctor flags the quoted line: %v", fs)
	}
	if _, err := AuthorizedKeysLine("/bin/reflopd\nevil", id, pub, paths.Layout{}); err == nil {
		t.Fatal("newline in path accepted")
	}
}

func scanLine(t *testing.T, line string) ([]Finding, error) {
	ak := filepath.Join(t.TempDir(), "authorized_keys")
	os.WriteFile(ak, []byte(line+"\n"), 0o600)
	return Scan(ak)
}
