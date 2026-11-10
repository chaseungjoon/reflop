// SPDX-License-Identifier: Apache-2.0

package sandbox

import "golang.org/x/sys/unix"

const (
	auditArch = 0xc00000b7 // AUDIT_ARCH_AARCH64
	x32Bit    = 0
)

var archAllowSyscalls = []int{}

var archDenySyscalls = []int{unix.SYS_KEXEC_FILE_LOAD}

var daemonArchDeny = []int{unix.SYS_KEXEC_FILE_LOAD}
