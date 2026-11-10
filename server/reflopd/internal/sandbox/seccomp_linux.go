// SPDX-License-Identifier: Apache-2.0

package sandbox

import (
	"bytes"
	"encoding/binary"
	"fmt"

	"golang.org/x/sys/unix"
)

// Classic BPF opcodes used by seccomp filters.
const (
	bpfLdWAbs = unix.BPF_LD | unix.BPF_W | unix.BPF_ABS
	bpfJeqK   = unix.BPF_JMP | unix.BPF_JEQ | unix.BPF_K
	bpfJgeK   = unix.BPF_JMP | unix.BPF_JGE | unix.BPF_K
	bpfJsetK  = unix.BPF_JMP | unix.BPF_JSET | unix.BPF_K
	bpfRetK   = unix.BPF_RET | unix.BPF_K

	seccompRetKillProcess = 0x80000000
	seccompRetErrno       = 0x00050000
	seccompRetAllow       = 0x7fff0000

	offNr   = 0
	offArch = 4
	offArg0 = 16
)

func retErrno(e unix.Errno) uint32 { return seccompRetErrno | (uint32(e) & 0xffff) }

func argLow(i int) uint32 { return uint32(offArg0 + 8*i) }

type prog []unix.SockFilter

func (p *prog) stmt(code uint16, k uint32) { *p = append(*p, unix.SockFilter{Code: code, K: k}) }
func (p *prog) jump(code uint16, k uint32, jt, jf uint8) {
	*p = append(*p, unix.SockFilter{Code: code, Jt: jt, Jf: jf, K: k})
}

// Policy is the worker's post-warm-up syscall policy.
type Policy struct {
	// SelfPID is the worker's PID; signals may only target it.
	SelfPID int
}

// Filter builds the seccomp-BPF program for the worker (PLAN.md §9.4).
//
// The filter is an allowlist. Unlisted syscalls fail with EPERM. Process
// creation (anything but threads), exec, ptrace, namespace and mount
// operations, eBPF, perf, keyrings, userfaultfd and every socket operation
// that could create or reach a new endpoint are denied: after warm-up the
// worker talks only through the sockets it inherited. A foreign
// architecture or the x32 ABI kills the process.
func (pol Policy) Filter() ([]unix.SockFilter, error) {
	var p prog
	p.stmt(bpfLdWAbs, offArch)
	p.jump(bpfJeqK, auditArch, 1, 0)
	p.stmt(bpfRetK, seccompRetKillProcess)
	p.stmt(bpfLdWAbs, offNr)
	if x32Bit != 0 {
		p.jump(bpfJgeK, x32Bit, 0, 1)
		p.stmt(bpfRetK, seccompRetKillProcess)
	}

	// Conditional syscalls. Each block starts with A = nr and always returns.
	cond := func(nr int, body func(*prog)) {
		var b prog
		body(&b)
		if len(b) > 255 {
			panic("seccomp block too long")
		}
		p.jump(bpfJeqK, uint32(nr), 0, uint8(len(b)))
		p = append(p, b...)
	}
	// clone: threads only.
	cond(unix.SYS_CLONE, func(b *prog) {
		b.stmt(bpfLdWAbs, argLow(0))
		b.jump(bpfJsetK, unix.CLONE_THREAD, 0, 1)
		b.stmt(bpfRetK, seccompRetAllow)
		b.stmt(bpfRetK, retErrno(unix.EPERM))
	})
	// clone3 passes flags in memory, which BPF cannot inspect: make glibc
	// fall back to clone.
	cond(unix.SYS_CLONE3, func(b *prog) { b.stmt(bpfRetK, retErrno(unix.ENOSYS)) })
	// Signals: only to this process.
	// pid_t is an int: the kernel uses only the low 32 bits, so negative
	// values (process groups, -1 = everyone) never equal SelfPID.
	selfOnly := func(nr int) {
		cond(nr, func(b *prog) {
			b.stmt(bpfLdWAbs, argLow(0))
			b.jump(bpfJeqK, uint32(pol.SelfPID), 0, 1)
			b.stmt(bpfRetK, seccompRetAllow)
			b.stmt(bpfRetK, retErrno(unix.EPERM))
		})
	}
	selfOnly(unix.SYS_KILL)
	selfOnly(unix.SYS_TGKILL)
	selfOnly(unix.SYS_RT_SIGQUEUEINFO)
	selfOnly(unix.SYS_RT_TGSIGQUEUEINFO)

	for _, nr := range denyEPERM {
		cond(nr, func(b *prog) { b.stmt(bpfRetK, retErrno(unix.EPERM)) })
	}
	// No new sockets: inherited socketpairs only.
	for _, nr := range []int{unix.SYS_SOCKET, unix.SYS_SOCKETPAIR} {
		cond(nr, func(b *prog) { b.stmt(bpfRetK, retErrno(unix.EAFNOSUPPORT)) })
	}

	seen := map[int]bool{}
	for _, nr := range append(append([]int{}, hotSyscalls...), allowSyscalls...) {
		if seen[nr] {
			continue
		}
		seen[nr] = true
		p.jump(bpfJeqK, uint32(nr), 0, 1)
		p.stmt(bpfRetK, seccompRetAllow)
	}
	p.stmt(bpfRetK, retErrno(unix.EPERM))
	if len(p) > 4096 {
		return nil, fmt.Errorf("seccomp filter too long: %d instructions", len(p))
	}
	return p, nil
}

// Encode serializes a filter as struct sock_filter[] in native byte order,
// the form the worker passes to seccomp(2).
func Encode(f []unix.SockFilter) []byte {
	var buf bytes.Buffer
	for _, ins := range f {
		_ = binary.Write(&buf, binary.NativeEndian, ins)
	}
	return buf.Bytes()
}

// SeccompAvailable reports whether seccomp filter mode is supported.
func SeccompAvailable() bool {
	// PR_GET_SECCOMP succeeds on kernels with CONFIG_SECCOMP; filter mode
	// needs CONFIG_SECCOMP_FILTER, checked via the actions_avail file.
	if _, err := unix.PrctlRetInt(unix.PR_GET_SECCOMP, 0, 0, 0, 0); err != nil {
		return false
	}
	return pathExists("/proc/sys/kernel/seccomp/actions_avail")
}
