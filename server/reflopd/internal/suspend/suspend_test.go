// SPDX-License-Identifier: Apache-2.0

package suspend

import (
	"errors"
	"os"
	"os/exec"
	"testing"
)

func python(t *testing.T) string {
	p, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 not installed")
	}
	return p
}

// The helper acts only on reflop workers started by the same reflopd: a
// process that is not one (this test process, a child that is not a worker,
// init) is refused before any driver call (ADR-030).
func TestHelperRefusesOtherProcesses(t *testing.T) {
	h := Helper{Python: python(t)}
	child := exec.Command("sleep", "30")
	if err := child.Start(); err != nil {
		t.Fatal(err)
	}
	defer child.Process.Kill()
	for name, pid := range map[string]int{"self": os.Getpid(), "child that is not a worker": child.Process.Pid,
		"init": 1, "parent": os.Getppid()} {
		for _, op := range []func(int) error{h.Suspend, h.Resume} {
			if err := op(pid); !errors.Is(err, ErrNotWorker) {
				t.Errorf("%s (pid %d): %v, want ErrNotWorker", name, pid, err)
			}
		}
	}
}

func TestHelperUsage(t *testing.T) {
	out, err := exec.Command(python(t), "-I", "-c", helperPy, "dump", "1", "10").CombinedOutput()
	var ee *exec.ExitError
	if !errors.As(err, &ee) || ee.ExitCode() != 2 {
		t.Fatalf("unknown operation: %v %s", err, out)
	}
}
