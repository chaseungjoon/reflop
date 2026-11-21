// SPDX-License-Identifier: Apache-2.0

package server

import (
	"bufio"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/reflopai/reflop/server/reflopd/internal/cgroup"
)

func TestWorkerLimitArithmetic(t *testing.T) {
	if got := workerHeadroom(10<<30, 4<<30); got != 6<<30 {
		t.Fatalf("headroom = %d", got)
	}
	if got := workerHeadroom(4<<30, 10<<30); got != 0 {
		t.Fatalf("headroom over budget = %d, want 0", got)
	}
	// Own use plus the pool's free headroom, less 1/8 of the headroom.
	if got := workerLimit(3<<30, 8<<30, false); got != 3<<30+7<<30 {
		t.Fatalf("limit = %d", got)
	}
	// The margin is at least workerLimitMargin...
	if got := workerLimit(1<<30, 100<<20, false); got != 1<<30+100<<20-workerLimitMargin {
		t.Fatalf("limit with little headroom = %d", got)
	}
	// ...but never more than the headroom: never below the worker's own use.
	if got := workerLimit(1<<30, 16<<20, false); got != 1<<30 {
		t.Fatalf("limit with tiny headroom = %d, want the worker's own use", got)
	}
	if got := workerLimit(1<<30, 0, false); got != 1<<30 {
		t.Fatalf("limit with no headroom = %d, want the worker's own use", got)
	}
	// Never below the minimum.
	if got := workerLimit(0, 0, false); got != minWorkerMemory {
		t.Fatalf("limit for an empty pool = %d", got)
	}
	// Suspended: no limit of its own.
	if got := workerLimit(3<<30, 2<<30, true); got != 0 {
		t.Fatalf("suspended limit = %d, want 0 (none)", got)
	}
}

// Issue #16, against the kernel: two workers share a small pool. A holds
// 350 MiB (an innocent large session); B keeps growing (the Guest causing
// the pressure). With only the shared limit, the kernel kills the biggest
// worker, A. With each worker's own limit set as the RAM rail sets it, B
// reaches its own limit first and only B dies.
//
// It needs a delegated cgroup v2 subtree and python3:
//
//	systemd-run --user --scope -p Delegate=yes --quiet --collect -- \
//	  env REFLOP_TEST_CGROUP=1 go test ./server/reflopd/internal/server -run TestWorkerLimitsKill -v -count=1
func TestWorkerLimitsKillTheGrowerNotTheBiggest(t *testing.T) {
	if os.Getenv("REFLOP_TEST_CGROUP") != "1" {
		t.Skip("needs a delegated cgroup scope (REFLOP_TEST_CGROUP=1)")
	}
	py, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("needs python3")
	}
	m, err := cgroup.Setup()
	if err != nil {
		t.Fatalf("cgroup setup (run inside systemd-run --user --scope -p Delegate=yes): %v", err)
	}
	const pool = 600 << 20
	const holdA = 350 << 20
	const growB = 400 << 20
	hold := "import sys,time\nn=int(sys.argv[1]);b=bytearray(n);b[::4096]=b'\\x01'*len(range(0,n,4096))\n" +
		"print('ready',flush=True);time.sleep(120)\n"
	grow := "import sys,time\nn=int(sys.argv[1]);s=[];step=16<<20\nwhile len(s)*step<n:\n" +
		" x=bytearray(step);x[::4096]=b'\\x01'*(step//4096);s.append(x);time.sleep(0.05)\n" +
		"print('grown',flush=True);time.sleep(120)\n"

	start := func(t *testing.T, w *cgroup.Worker, script string, n int) (*exec.Cmd, *bufio.Scanner) {
		cmd := exec.Command(py, "-c", script, strconv.Itoa(n))
		cmd.SysProcAttr = &syscall.SysProcAttr{UseCgroupFD: true, CgroupFD: w.FD}
		out, err := cmd.StdoutPipe()
		if err != nil {
			t.Fatal(err)
		}
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		return cmd, bufio.NewScanner(out)
	}
	run := func(t *testing.T, name string, ownLimits bool) (aDied, bDied bool) {
		if err := m.SetMemoryMax(pool); err != nil {
			t.Fatal(err)
		}
		a, err := m.NewWorker(name+"-a", 64, 0)
		if err != nil {
			t.Fatal(err)
		}
		b, err := m.NewWorker(name+"-b", 64, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer a.Remove()
		defer b.Remove()
		ca, outA := start(t, a, hold, holdA)
		if !outA.Scan() || !strings.Contains(outA.Text(), "ready") {
			t.Fatal("worker A did not start")
		}
		cb, _ := start(t, b, grow, growB)
		done := map[*exec.Cmd]bool{}
		ch := make(chan *exec.Cmd, 2)
		go func() { ca.Wait(); ch <- ca }()
		go func() { cb.Wait(); ch <- cb }()
		deadline := time.Now().Add(15 * time.Second)
		for time.Now().Before(deadline) && len(done) == 0 {
			if ownLimits {
				// What ramRail does every tick.
				ua, ub := a.MemoryCurrent(), b.MemoryCurrent()
				h := workerHeadroom(pool, ua+ub)
				_ = a.SetMemoryMax(workerLimit(ua, h, false))
				_ = b.SetMemoryMax(workerLimit(ub, h, false))
			}
			select {
			case c := <-ch:
				done[c] = true
			case <-time.After(20 * time.Millisecond):
			}
			if b.MemoryCurrent() >= growB {
				break // B got all it wanted: nobody had to die
			}
		}
		aDied, bDied = done[ca] || a.OOMKills() > 0, done[cb] || b.OOMKills() > 0
		ev := func(w *cgroup.Worker) string {
			bs, _ := os.ReadFile(w.Dir + "/memory.events")
			return strings.ReplaceAll(strings.TrimSpace(string(bs)), "\n", " ")
		}
		rd := func(p string) string { bs, _ := os.ReadFile(p); return strings.TrimSpace(string(bs)) }
		t.Logf("%s: oom.group root=%s workers=%s leafA=%s leafB=%s; memory.max workers=%s A=%s B=%s", name,
			rd(m.Root+"/memory.oom.group"), rd(m.Root+"/workers/memory.oom.group"), rd(a.Dir+"/memory.oom.group"),
			rd(b.Dir+"/memory.oom.group"), rd(m.Root+"/workers/memory.max"), rd(a.Dir+"/memory.max"), rd(b.Dir+"/memory.max"))
		t.Logf("%s: A %v (%s; mem %d MiB) | B %v (%s; mem %d MiB)", name, ca.ProcessState, ev(a), a.MemoryCurrent()>>20,
			cb.ProcessState, ev(b), b.MemoryCurrent()>>20)
		a.Kill()
		b.Kill()
		for len(done) < 2 {
			done[<-ch] = true
		}
		return aDied, bDied
	}

	t.Run("shared limit only: the bigger worker dies", func(t *testing.T) {
		aDied, _ := run(t, "shared", false)
		// The kernel kills the biggest worker, A, and often B as well while
		// A's memory is still being freed.
		if !aDied {
			t.Fatal("A survived; the old setup should kill it (the premise of #16)")
		}
	})
	t.Run("own limits: the growing worker dies", func(t *testing.T) {
		aDied, bDied := run(t, "own", true)
		if aDied || !bDied {
			t.Fatalf("A died %v, B died %v; B should reach its own limit and die alone", aDied, bDied)
		}
	})
}
