// SPDX-License-Identifier: Apache-2.0

package sandbox

import (
	"fmt"
	"os"
	"runtime"
	"unsafe"

	"golang.org/x/sys/unix"
)

// daemonDeny are syscalls reflopd itself never needs (PLAN.md §9.8): the
// daemon keeps the ability to spawn workers but cannot debug other
// processes, load kernel code, mount, or use eBPF/perf/keyrings.
var daemonDeny = append([]int{
	unix.SYS_PTRACE, unix.SYS_PROCESS_VM_READV, unix.SYS_PROCESS_VM_WRITEV, unix.SYS_MOUNT, unix.SYS_UMOUNT2,
	unix.SYS_PIVOT_ROOT, unix.SYS_SWAPON, unix.SYS_SWAPOFF, unix.SYS_REBOOT, unix.SYS_KEXEC_LOAD,
	unix.SYS_INIT_MODULE, unix.SYS_FINIT_MODULE, unix.SYS_DELETE_MODULE, unix.SYS_BPF, unix.SYS_PERF_EVENT_OPEN,
	unix.SYS_KEYCTL, unix.SYS_ADD_KEY, unix.SYS_REQUEST_KEY, unix.SYS_USERFAULTFD, unix.SYS_OPEN_BY_HANDLE_AT,
	unix.SYS_NAME_TO_HANDLE_AT, unix.SYS_ACCT, unix.SYS_SETTIMEOFDAY, unix.SYS_CLOCK_SETTIME, unix.SYS_CLOCK_ADJTIME,
	unix.SYS_ADJTIMEX, unix.SYS_QUOTACTL, unix.SYS_FANOTIFY_INIT, unix.SYS_SETNS, unix.SYS_FSOPEN,
	unix.SYS_FSMOUNT, unix.SYS_MOVE_MOUNT, unix.SYS_OPEN_TREE, unix.SYS_FSPICK, unix.SYS_MOUNT_SETATTR,
	unix.SYS_KCMP, unix.SYS_PIDFD_GETFD, unix.SYS_IO_URING_SETUP, unix.SYS_IO_URING_ENTER, unix.SYS_IO_URING_REGISTER,
}, daemonArchDeny...)

// DaemonFilter is a denylist filter for reflopd itself.
func DaemonFilter() ([]unix.SockFilter, error) {
	var p prog
	p.stmt(bpfLdWAbs, offArch)
	p.jump(bpfJeqK, auditArch, 1, 0)
	p.stmt(bpfRetK, seccompRetKillProcess)
	p.stmt(bpfLdWAbs, offNr)
	if x32Bit != 0 {
		p.jump(bpfJgeK, x32Bit, 0, 1)
		p.stmt(bpfRetK, seccompRetKillProcess)
	}
	for _, nr := range daemonDeny {
		p.jump(bpfJeqK, uint32(nr), 0, 1)
		p.stmt(bpfRetK, retErrno(unix.EPERM))
	}
	p.stmt(bpfRetK, seccompRetAllow)
	return p, nil
}

type sockFprog struct {
	Len    uint16
	_      [6]byte
	Filter *unix.SockFilter
}

// InstallFilterSelf installs a seccomp filter on the calling thread.
func InstallFilterSelf(f []unix.SockFilter) error {
	if len(f) == 0 {
		return nil
	}
	fp := sockFprog{Len: uint16(len(f)), Filter: &f[0]}
	_, _, errno := unix.Syscall(unix.SYS_SECCOMP, 1 /* SET_MODE_FILTER */, 0, uintptr(unsafe.Pointer(&fp)))
	if errno != 0 {
		return errno
	}
	return nil
}

// DaemonSandboxEnv marks a reflopd that already sandboxed itself.
const DaemonSandboxEnv = "REFLOP_DAEMON_SANDBOXED"

// ReexecSandboxed applies Landlock and the daemon seccomp filter on a
// locked thread and re-executes the current binary with argv,
// so that every thread of the daemon starts inside the sandbox (Landlock
// before ABI 8 cannot restrict existing threads). It returns only on error.
func ReexecSandboxed(rules []PathRule, bindPorts []uint16, argv []string) error {
	runtime.LockOSThread()
	if err := unix.Prctl(unix.PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0); err != nil {
		return err
	}
	if LandlockABI() > 0 {
		if _, err := restrictWithPorts(rules, bindPorts); err != nil {
			return fmt.Errorf("landlock: %w", err)
		}
	}
	if SeccompAvailable() {
		f, _ := DaemonFilter()
		if err := InstallFilterSelf(f); err != nil {
			return fmt.Errorf("seccomp: %w", err)
		}
	}
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	env := append(os.Environ(), DaemonSandboxEnv+"=1")
	return unix.Exec(exe, argv, env)
}

type netPortAttr struct {
	allowed uint64
	port    uint64
}

// restrictWithPorts is RestrictSelf plus TCP bind permissions for the
// daemon's listening ports (connect stays denied: reflopd makes no
// outbound TCP connections while serving).
func restrictWithPorts(rules []PathRule, ports []uint16) (LandlockResult, error) {
	abi := LandlockABI()
	res := LandlockResult{ABI: abi}
	if abi > 9 {
		abi = 9
	}
	attr := rulesetAttr{handledFS: fsMaskForABI(abi), handledNet: netMaskForABI(abi)}
	size := unsafe.Sizeof(attr)
	switch {
	case abi < 4:
		size = 8
	case abi < 6:
		size = 16
	}
	// No scoping for the daemon: it must signal its workers.
	fd, _, errno := unix.Syscall(unix.SYS_LANDLOCK_CREATE_RULESET, uintptr(unsafe.Pointer(&attr)), size, 0)
	if errno == unix.EINVAL && abi >= 9 {
		abi = 8
		attr = rulesetAttr{handledFS: fsMaskForABI(abi), handledNet: netMaskForABI(abi)}
		fd, _, errno = unix.Syscall(unix.SYS_LANDLOCK_CREATE_RULESET, uintptr(unsafe.Pointer(&attr)), unsafe.Sizeof(attr), 0)
	}
	if errno != 0 {
		return res, errno
	}
	rfd := int(fd)
	defer unix.Close(rfd)
	for _, r := range rules {
		access := r.Access & attr.handledFS
		pfd, err := unix.Open(r.Path, unix.O_PATH|unix.O_CLOEXEC, 0)
		if err != nil {
			res.Skipped = append(res.Skipped, r.Path)
			continue
		}
		var st unix.Stat_t
		if unix.Fstat(pfd, &st) == nil && st.Mode&unix.S_IFMT != unix.S_IFDIR {
			access &^= llReadDir | llRemoveDir | llRemoveFile | llMakeChar | llMakeDir | llMakeReg |
				llMakeSock | llMakeFifo | llMakeBlock | llMakeSym | llRefer
		}
		pb := pathBeneathAttr{allowed: access, parentFd: int32(pfd)}
		_, _, e := unix.Syscall6(unix.SYS_LANDLOCK_ADD_RULE, uintptr(rfd), unix.LANDLOCK_RULE_PATH_BENEATH,
			uintptr(unsafe.Pointer(&pb)), 0, 0, 0)
		unix.Close(pfd)
		if e != 0 {
			return res, fmt.Errorf("add rule %s: %w", r.Path, e)
		}
	}
	if attr.handledNet&llNetBindTCP != 0 {
		for _, port := range ports {
			np := netPortAttr{allowed: llNetBindTCP, port: uint64(port)}
			_, _, e := unix.Syscall6(unix.SYS_LANDLOCK_ADD_RULE, uintptr(rfd), 2, /* LANDLOCK_RULE_NET_PORT */
				uintptr(unsafe.Pointer(&np)), 0, 0, 0)
			if e != 0 {
				return res, fmt.Errorf("add port rule %d: %w", port, e)
			}
		}
	}
	if _, _, e := unix.Syscall(unix.SYS_LANDLOCK_RESTRICT_SELF, uintptr(rfd), 0, 0); e != 0 {
		return res, e
	}
	res.FS, res.Net = true, attr.handledNet != 0
	return res, nil
}

// AccessAll is every filesystem right (for the daemon's writable dirs).
const AccessAll = AccessScratch | llMakeSock | llMakeFifo
