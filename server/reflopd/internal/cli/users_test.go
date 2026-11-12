// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/chaseungjoon/reflop/server/reflopd/internal/auth"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/config"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/paths"
)

// `reflopd users delete` refuses active identities and, for a revoked one,
// removes the identity, its stored data and its quotas, and nobody else's.
func TestUsersDelete(t *testing.T) {
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
	gone, _ := ids.AddKey("gone", bytes.Repeat([]byte{1}, 32), "ssh", "")
	kept, _ := ids.AddKey("kept", bytes.Repeat([]byte{2}, 32), "ssh", "")
	var dirs []string
	for _, id := range []string{gone.ID, kept.ID} {
		for _, d := range []string{l.IdentityStore(id), l.IdentitySnapshots(id)} {
			if err := os.MkdirAll(filepath.Join(d, "blobs"), 0o700); err != nil {
				t.Fatal(err)
			}
			dirs = append(dirs, d)
		}
	}
	quotas := map[string]config.Quota{"gone": {StorageBytes: 1 << 30}, "kept": {StorageBytes: 1 << 30}}
	if err := config.SaveSettings(settingsPath(l), config.Settings{Quotas: quotas}); err != nil {
		t.Fatal(err)
	}
	run := func(args ...string) (int, string) {
		var out, errb bytes.Buffer
		code := Main(args, &out, &errb)
		return code, out.String() + errb.String()
	}

	if code, out := run("users", "delete", "gone"); code != ExitUsage || !strings.Contains(out, "revoke it first") {
		t.Fatalf("deleting an active identity: exit %d, %q", code, out)
	}
	if _, ok := ids.Get("gone"); !ok {
		t.Fatal("active identity was deleted")
	}
	if _, err := os.Stat(dirs[0]); err != nil {
		t.Fatalf("active identity's data touched: %v", err)
	}

	if _, err := ids.Revoke("gone"); err != nil {
		t.Fatal(err)
	}
	if code, out := run("users", "delete", "gone"); code != ExitOK {
		t.Fatalf("deleting a revoked identity: exit %d, %q", code, out)
	}
	if _, ok := ids.Get("gone"); ok {
		t.Fatal("revoked identity still in the store")
	}
	if _, out := run("users"); strings.Contains(out, "gone") || !strings.Contains(out, "kept") {
		t.Fatalf("users after delete:\n%s", out)
	}
	for i, d := range dirs {
		_, err := os.Stat(d)
		if wantGone := i < 2; wantGone != os.IsNotExist(err) {
			t.Fatalf("%s: stat %v (should be gone: %v)", d, err, wantGone)
		}
	}
	s, _ := config.LoadSettings(settingsPath(l))
	if _, ok := s.Quotas["gone"]; ok {
		t.Fatal("quota kept for the deleted identity")
	}
	if _, ok := s.Quotas["kept"]; !ok {
		t.Fatal("another identity's quota removed")
	}
	if code, _ := run("users", "delete", "gone"); code != ExitUsage {
		t.Fatalf("deleting twice: exit %d", code)
	}
}
