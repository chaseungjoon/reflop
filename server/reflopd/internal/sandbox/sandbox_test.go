// SPDX-License-Identifier: Apache-2.0

package sandbox

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"syscall"
	"testing"
	"unsafe"

	"golang.org/x/sys/unix"
)

const childEnv = "REFLOP_SECCOMP_CHILD"

func TestMain(m *testing.M) {
	if os.Getenv(childEnv) == "1" {
		os.Exit(seccompChild())
	}
	os.Exit(m.Run())
}

// seccompChild installs the worker filter (TSYNC) and checks syscalls.
func seccompChild() int {
	runtime.LockOSThread()
	if err := unix.Prctl(unix.PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0); err != nil {
		fmt.Println("nnp:", err)
		return 2
	}
	f, err := Policy{SelfPID: os.Getpid()}.Filter()
	if err != nil {
		fmt.Println(err)
		return 2
	}
	fp := sockFprog{Len: uint16(len(f)), Filter: &f[0]}
	if _, _, e := unix.Syscall(unix.SYS_SECCOMP, 1, 1 /* TSYNC */, uintptr(unsafe.Pointer(&fp))); e != 0 {
		fmt.Println("seccomp:", e)
		return 2
	}
	fails := 0
	expectErr := func(name string, err error) {
		if err == nil {
			fmt.Println("ALLOWED:", name)
			fails++
		}
	}
	expectOK := func(name string, err error) {
		if err != nil {
			fmt.Println("BLOCKED (should work):", name, err)
			fails++
		}
	}
	_, err = unix.Socket(unix.AF_INET, unix.SOCK_STREAM, 0)
	expectErr("socket(AF_INET)", err)
	_, err = unix.Socket(unix.AF_INET6, unix.SOCK_DGRAM, 0)
	expectErr("socket(AF_INET6)", err)
	_, err = unix.Socket(unix.AF_UNIX, unix.SOCK_STREAM, 0)
	expectErr("socket(AF_UNIX)", err)
	_, err = unix.Socket(unix.AF_NETLINK, unix.SOCK_RAW, 0)
	expectErr("socket(AF_NETLINK)", err)
	_, _, e := unix.RawSyscall(unix.SYS_EXECVE, 0, 0, 0)
	expectErr("execve", errnoErr(e))
	_, _, e = unix.RawSyscall6(unix.SYS_CLONE, uintptr(syscall.SIGCHLD), 0, 0, 0, 0, 0)
	expectErr("clone(fork)", errnoErr(e))
	_, _, e = unix.RawSyscall(unix.SYS_CLONE3, 0, 0, 0)
	if e != unix.ENOSYS {
		fmt.Println("clone3 should be ENOSYS, got", e)
		fails++
	}
	expectErr("ptrace(attach ppid)", unix.PtraceAttach(os.Getppid()))
	expectErr("kill(1, 0)", unix.Kill(1, 0))
	expectErr("kill(-1, 0)", unix.Kill(-1, 0))
	expectErr("kill(ppid, 0)", unix.Kill(os.Getppid(), 0))
	expectOK("kill(self, 0)", unix.Kill(os.Getpid(), 0))
	expectErr("unshare(NEWUSER)", unix.Unshare(unix.CLONE_NEWUSER))
	expectErr("mount", unix.Mount("none", "/tmp", "tmpfs", 0, ""))
	_, _, e = unix.Syscall(unix.SYS_BPF, 0, 0, 0)
	expectErr("bpf", errnoErr(e))
	_, _, e = unix.Syscall(unix.SYS_USERFAULTFD, 0, 0, 0)
	expectErr("userfaultfd", errnoErr(e))
	_, _, e = unix.Syscall(unix.SYS_IO_URING_SETUP, 1, 0, 0)
	expectErr("io_uring_setup", errnoErr(e))
	expectErr("setuid(0)", unix.Setuid(0))
	// Normal work keeps running, threads included.
	done := make(chan int)
	for i := 0; i < 8; i++ {
		go func(i int) {
			b := make([]byte, 1<<20)
			b[i] = 1
			done <- int(b[i])
		}(i)
	}
	s := 0
	for i := 0; i < 8; i++ {
		s += <-done
	}
	if s != 8 {
		fails++
	}
	fd, err := unix.Open("/dev/null", unix.O_RDONLY, 0)
	expectOK("open /dev/null", err)
	unix.Close(fd)
	if fails > 0 {
		return 1
	}
	return 0
}

func errnoErr(e unix.Errno) error {
	if e == 0 {
		return nil
	}
	return e
}

func TestWorkerSeccompFilter(t *testing.T) {
	if !SeccompAvailable() {
		t.Skip("seccomp unavailable")
	}
	cmd := exec.Command(os.Args[0], "-test.run=^$")
	cmd.Env = append(os.Environ(), childEnv+"=1")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("seccomp child failed: %v\n%s", err, out)
	}
}

func TestFilterFits(t *testing.T) {
	f, err := Policy{SelfPID: 1234}.Filter()
	if err != nil {
		t.Fatal(err)
	}
	if len(f) > 4096 || len(Encode(f)) != 8*len(f) {
		t.Fatalf("bad filter size %d", len(f))
	}
	d, _ := DaemonFilter()
	if len(d) == 0 {
		t.Fatal("empty daemon filter")
	}
}

// TestFilterFitsKernelLimit pins the seccomp program well below the
// kernel's 4096-instruction limit. Filter() fails past the limit, and a
// failure must never mean "run without seccomp" (SEC_SWEEP M2), so growth
// of the allow/deny lists should trip this test long before the limit.
func TestFilterFitsKernelLimit(t *testing.T) {
	f, err := Policy{SelfPID: 4242}.Filter()
	if err != nil {
		t.Fatal(err)
	}
	const budget = 3072
	if len(f) > budget {
		t.Fatalf("seccomp filter has %d instructions; keep it under %d (kernel limit 4096)", len(f), budget)
	}
	t.Logf("seccomp filter: %d instructions", len(f))
}
