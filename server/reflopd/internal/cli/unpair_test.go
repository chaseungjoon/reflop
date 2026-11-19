// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/reflopai/reflop/server/reflopd/internal/auth"
	"github.com/reflopai/reflop/server/reflopd/internal/paths"
)

// ADR-025: `reflopd unpair` is revoke + users delete in one confirmed step,
// and `reflopd users` asks the Owner to finish a Guest's `reflop unpair`.
func TestUnpairCmd(t *testing.T) {
	t.Setenv("REFLOP_HOME", t.TempDir())
	t.Setenv("REFLOP_AUTHORIZED_KEYS", filepath.Join(t.TempDir(), "authorized_keys"))
	l, err := paths.Resolve()
	if err != nil {
		t.Fatal(err)
	}
	if err := l.Ensure(); err != nil {
		t.Fatal(err)
	}
	ids, err := auth.Open(l.Identities())
	if err != nil {
		t.Fatal(err)
	}
	eve, _ := ids.AddKey("eve", bytes.Repeat([]byte{3}, 32), "ssh", "")
	fay, _ := ids.AddKey("fay", bytes.Repeat([]byte{4}, 32), "ssh", "")
	data := l.IdentityStore(eve.ID)
	if err := os.MkdirAll(filepath.Join(data, "blobs"), 0o700); err != nil {
		t.Fatal(err)
	}
	run := func(args ...string) (int, string) {
		var out, errb bytes.Buffer
		code := Main(args, &out, &errb)
		return code, out.String() + errb.String()
	}

	// Not a terminal and no --yes: nothing changes.
	if code, out := run("unpair", "eve"); code != ExitUsage || !strings.Contains(out, "--yes") {
		t.Fatalf("unpair without --yes: exit %d, %q", code, out)
	}
	if id, _ := ids.Get("eve"); id.Revoked {
		t.Fatal("unpair without confirmation revoked")
	}

	// An active identity: revoked, deleted with its data, name free again.
	code, out := run("unpair", "eve", "--yes")
	if code != ExitOK || !strings.Contains(out, `revoked "eve"`) || !strings.Contains(out, "reflop unpair") {
		t.Fatalf("unpair eve: exit %d, %q", code, out)
	}
	if _, ok := ids.Get("eve"); ok {
		t.Fatal("eve still in the store")
	}
	if _, err := os.Stat(data); !os.IsNotExist(err) {
		t.Fatalf("eve's data: %v", err)
	}
	if _, err := ids.CreateInvite("eve", time.Hour); err != nil {
		t.Fatalf("invite after unpair: %v", err)
	}

	// The Guest unpaired first: `reflopd users` and the error for a new
	// invite say what to run; unpair finishes it.
	if _, err := ids.GuestUnpair(fay.ID); err != nil {
		t.Fatal(err)
	}
	if _, out := run("users"); !strings.Contains(out, "unpaired by guest") || !strings.Contains(out, "reflopd unpair fay") {
		t.Fatalf("users after the Guest unpaired:\n%s", out)
	}
	if code, out := run("unpair", "fay", "-y"); code != ExitOK || !strings.Contains(out, "already run `reflop unpair`") {
		t.Fatalf("unpair fay: exit %d, %q", code, out)
	}
	if _, out := run("users"); strings.Contains(out, "fay") {
		t.Fatalf("fay still listed:\n%s", out)
	}
	if code, _ := run("unpair", "nobody", "-y"); code != ExitUsage {
		t.Fatalf("unpair of an unknown name: exit %d", code)
	}
}
