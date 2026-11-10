// SPDX-License-Identifier: Apache-2.0

package usage

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/chaseungjoon/reflop/server/reflopd/internal/audit"
)

func writeLog(t *testing.T, dir string, base time.Time) {
	t.Helper()
	l, err := audit.Open(dir, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	at := func(m int) time.Time { return base.Add(time.Duration(m) * time.Minute) }
	for _, r := range []audit.Record{
		{Time: at(0), Identity: "alice", Session: "s1", Event: "session_open", Fields: map[string]any{"device": "cuda:0"}},
		{Time: at(1), Identity: "alice", Session: "s1", Event: "execute", Fields: map[string]any{"us": 2000, "flops": 1e9, "peak": 100}},
		{Time: at(2), Identity: "alice", Session: "s1", Event: "execute", Fields: map[string]any{"us": 3000, "flops": 2e9, "peak": 50}},
		{Time: at(10), Identity: "alice", Session: "s1", Event: "session_end", Fields: map[string]any{"reason": "closed"}},
		{Time: at(11), Identity: "alice", Event: "disconnect", Fields: map[string]any{"bytes_up": 10, "bytes_down": 20}},
		{Time: at(20), Identity: "bob", Session: "s2", Event: "session_open", Fields: map[string]any{"device": "cpu"}},
		{Time: at(21), Identity: "bob", Session: "s2", Event: "execute", Fields: map[string]any{"us": 7000, "flops": 5e9, "peak": 9}},
		{Time: at(21), Identity: "carol", Event: "connect"},
	} {
		if err := l.Write(r); err != nil {
			t.Fatal(err)
		}
	}
}

func TestFromAudit(t *testing.T) {
	dir := t.TempDir()
	base := time.Now().UTC().Add(-time.Hour).Truncate(time.Second)
	writeLog(t, dir, base)

	rep, err := FromAudit(dir, time.Time{}, base.Add(30*time.Minute))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := rep.Identities["carol"]; ok {
		t.Fatal("an identity that used nothing should not be listed")
	}
	a := rep.Identities["alice"]
	if a.Executes != 2 || a.DeviceUs != 5000 || a.FLOPs != 3e9 || a.PeakMemory != 100 || a.Sessions != 1 ||
		a.SessionUs != uint64((10*time.Minute).Microseconds()) || a.BytesUp != 10 || a.BytesDown != 20 ||
		a.ByDevice["cuda:0"] != 5000 {
		t.Fatalf("alice: %+v", a)
	}
	// bob's session is still open: it counts up to `until`.
	if b := rep.Identities["bob"]; b.DeviceUs != 7000 || b.SessionUs != uint64((10*time.Minute).Microseconds()) {
		t.Fatalf("bob: %+v", b)
	}
	if !rep.Since.Equal(base) {
		t.Fatalf("since = %v, want first record %v", rep.Since, base)
	}

	// A window starting mid-session counts the session from the window start.
	rep, err = FromAudit(dir, base.Add(90*time.Second), base.Add(30*time.Minute))
	if err != nil {
		t.Fatal(err)
	}
	if a := rep.Identities["alice"]; a.Executes != 1 || a.DeviceUs != 3000 || a.Sessions != 0 ||
		a.SessionUs != uint64((10*time.Minute-90*time.Second).Microseconds()) {
		t.Fatalf("windowed alice: %+v", a)
	}
}

func TestLedgerBackfillAndPersist(t *testing.T) {
	dir := t.TempDir()
	base := time.Now().UTC().Add(-time.Hour).Truncate(time.Second)
	writeLog(t, filepath.Join(dir, "audit"), base)
	path := filepath.Join(dir, "usage.json")

	l, err := Open(path, filepath.Join(dir, "audit"))
	if err != nil {
		t.Fatal(err)
	}
	snap := l.Snapshot()
	if snap.Identities["alice"].DeviceUs != 5000 {
		t.Fatalf("backfill missed alice: %+v", snap.Identities["alice"])
	}
	// bob's session never ended in the log (its daemon died): not held time.
	if b := snap.Identities["bob"]; b.DeviceUs != 7000 || b.SessionUs != 0 {
		t.Fatalf("backfilled bob: %+v", b)
	}

	l.SessionOpen("bob")
	l.Execute("bob", "cuda:0", 1000, 4e12, 2048)
	l.SessionEnd("bob", 3*time.Second)
	l.Transfer("bob", 5, 6)
	if err := l.Save(); err != nil {
		t.Fatal(err)
	}

	// Reopening reads the ledger, not the audit log again.
	l2, err := Open(path, filepath.Join(dir, "audit"))
	if err != nil {
		t.Fatal(err)
	}
	b := l2.Snapshot().Identities["bob"]
	if b.Executes != 2 || b.DeviceUs != 8000 || b.FLOPs != 4e12+5e9 || b.Sessions != 2 || b.SessionUs != 3e6 ||
		b.BytesUp != 5 || b.BytesDown != 6 || b.ByDevice["cuda:0"] != 1000 || b.ByDevice["cpu"] != 7000 || b.PeakMemory != 2048 {
		t.Fatalf("persisted bob: %+v", b)
	}
	if !l2.Snapshot().Since.Equal(base) {
		t.Fatalf("since not persisted: %v", l2.Snapshot().Since)
	}
}
