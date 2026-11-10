// SPDX-License-Identifier: Apache-2.0

package sandbox

import (
	"errors"
	"fmt"
	"os"
	"unsafe"

	"golang.org/x/sys/unix"
)

// Landlock access rights. Values from the kernel UAPI; ABI numbers note
// the first ABI that understands each right.
const (
	llExecute     = 1 << 0 // ABI 1
	llWriteFile   = 1 << 1
	llReadFile    = 1 << 2
	llReadDir     = 1 << 3
	llRemoveDir   = 1 << 4
	llRemoveFile  = 1 << 5
	llMakeChar    = 1 << 6
	llMakeDir     = 1 << 7
	llMakeReg     = 1 << 8
	llMakeSock    = 1 << 9
	llMakeFifo    = 1 << 10
	llMakeBlock   = 1 << 11
	llMakeSym     = 1 << 12
	llRefer       = 1 << 13 // ABI 2
	llTruncate    = 1 << 14 // ABI 3
	llIoctlDev    = 1 << 15 // ABI 5
	llResolveUnix = 1 << 16 // ABI 9 (connect to pathname Unix sockets)

	llNetBindTCP    = 1 << 0 // ABI 4
	llNetConnectTCP = 1 << 1 // ABI 4
	llNetBindUDP    = 1 << 2 // ABI 9
	llNetSendUDP    = 1 << 3 // ABI 9

	llScopeAbstractUnix = 1 << 0 // ABI 6
	llScopeSignal       = 1 << 1 // ABI 6
)

// Access groups used by rules.
const (
	AccessRead    = llReadFile | llReadDir
	AccessExec    = llExecute
	AccessWrite   = llWriteFile | llTruncate
	AccessDevice  = llReadFile | llWriteFile | llIoctlDev
	AccessScratch = llReadFile | llReadDir | llWriteFile | llTruncate | llRemoveDir |
		llRemoveFile | llMakeDir | llMakeReg | llMakeSym | llRefer
)

// LandlockABI returns the kernel's Landlock ABI version, or 0 if Landlock is
// unavailable (not built, disabled at boot, or blocked by a parent sandbox).
func LandlockABI() int {
	v, _, errno := unix.Syscall(unix.SYS_LANDLOCK_CREATE_RULESET, 0, 0, unix.LANDLOCK_CREATE_RULESET_VERSION)
	if errno != 0 {
		return 0
	}
	return int(v)
}

func fsMaskForABI(abi int) uint64 {
	if abi < 1 {
		return 0
	}
	m := uint64(llExecute | llWriteFile | llReadFile | llReadDir | llRemoveDir | llRemoveFile |
		llMakeChar | llMakeDir | llMakeReg | llMakeSock | llMakeFifo | llMakeBlock | llMakeSym)
	if abi >= 2 {
		m |= llRefer
	}
	if abi >= 3 {
		m |= llTruncate
	}
	if abi >= 5 {
		m |= llIoctlDev
	}
	if abi >= 9 {
		m |= llResolveUnix
	}
	return m
}

func netMaskForABI(abi int) uint64 {
	var m uint64
	if abi >= 4 {
		m |= llNetBindTCP | llNetConnectTCP
	}
	if abi >= 9 {
		m |= llNetBindUDP | llNetSendUDP
	}
	return m
}

func scopeMaskForABI(abi int) uint64 {
	if abi >= 6 {
		return llScopeAbstractUnix | llScopeSignal
	}
	return 0
}

// PathRule grants access beneath one path.
type PathRule struct {
	Path   string `json:"path"`
	Access uint64 `json:"access"`
	// Optional rules are skipped silently when the path does not exist.
	Optional bool `json:"optional,omitempty"`
}

type rulesetAttr struct {
	handledFS  uint64
	handledNet uint64
	scoped     uint64
}

type pathBeneathAttr struct {
	allowed  uint64
	parentFd int32
}

// LandlockResult describes what was enforced.
type LandlockResult struct {
	ABI     int
	FS      bool
	Net     bool
	Scoped  bool
	Skipped []string
}

// RestrictSelf enforces a Landlock domain on the calling thread. Every
// filesystem access not covered by rules is denied, every TCP (and, on newer
// kernels, UDP) bind/connect is denied, and abstract Unix sockets and signals
// outside the domain are blocked. The caller must hold the OS thread
// (runtime.LockOSThread) and exec right after, so the domain covers the new
// program from its first instruction.
func RestrictSelf(rules []PathRule) (LandlockResult, error) {
	abi := LandlockABI()
	res := LandlockResult{ABI: abi}
	if abi < 1 {
		return res, errors.New("landlock unavailable")
	}
	if abi > 9 {
		abi = 9 // only rights this code knows about
	}
	attr := rulesetAttr{handledFS: fsMaskForABI(abi), handledNet: netMaskForABI(abi), scoped: scopeMaskForABI(abi)}
	size := unsafe.Sizeof(attr)
	switch {
	case abi < 4:
		size = 8
	case abi < 6:
		size = 16
	}
	fd, _, errno := unix.Syscall(unix.SYS_LANDLOCK_CREATE_RULESET, uintptr(unsafe.Pointer(&attr)), size, 0)
	if errno == unix.EINVAL && abi >= 9 {
		// A kernel reporting ABI ≥ 9 without the newest rights: retry at 8.
		abi = 8
		attr = rulesetAttr{handledFS: fsMaskForABI(abi), handledNet: netMaskForABI(abi), scoped: scopeMaskForABI(abi)}
		fd, _, errno = unix.Syscall(unix.SYS_LANDLOCK_CREATE_RULESET, uintptr(unsafe.Pointer(&attr)), unsafe.Sizeof(attr), 0)
	}
	if errno != 0 {
		return res, fmt.Errorf("landlock_create_ruleset: %w", errno)
	}
	rfd := int(fd)
	defer unix.Close(rfd)
	for _, r := range rules {
		access := r.Access & attr.handledFS
		if access == 0 {
			continue
		}
		pfd, err := unix.Open(r.Path, unix.O_PATH|unix.O_CLOEXEC, 0)
		if err != nil {
			if r.Optional || errors.Is(err, unix.ENOENT) {
				res.Skipped = append(res.Skipped, r.Path)
				continue
			}
			return res, fmt.Errorf("landlock rule %s: %w", r.Path, err)
		}
		var st unix.Stat_t
		if err := unix.Fstat(pfd, &st); err == nil && st.Mode&unix.S_IFMT != unix.S_IFDIR {
			// Directory-only rights are invalid on files.
			access &^= llReadDir | llRemoveDir | llRemoveFile | llMakeChar | llMakeDir | llMakeReg |
				llMakeSock | llMakeFifo | llMakeBlock | llMakeSym | llRefer
		}
		pb := pathBeneathAttr{allowed: access, parentFd: int32(pfd)}
		_, _, errno := unix.Syscall6(unix.SYS_LANDLOCK_ADD_RULE, uintptr(rfd), unix.LANDLOCK_RULE_PATH_BENEATH,
			uintptr(unsafe.Pointer(&pb)), 0, 0, 0)
		unix.Close(pfd)
		if errno != 0 {
			return res, fmt.Errorf("landlock_add_rule %s: %w", r.Path, errno)
		}
	}
	if err := unix.Prctl(unix.PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0); err != nil {
		return res, fmt.Errorf("no_new_privs: %w", err)
	}
	if _, _, errno := unix.Syscall(unix.SYS_LANDLOCK_RESTRICT_SELF, uintptr(rfd), 0, 0); errno != 0 {
		return res, fmt.Errorf("landlock_restrict_self: %w", errno)
	}
	res.ABI = abi
	res.FS = true
	res.Net = attr.handledNet != 0
	res.Scoped = attr.scoped != 0
	return res, nil
}

// pathExists reports whether p exists.
func pathExists(p string) bool {
	_, err := os.Lstat(p)
	return err == nil
}
