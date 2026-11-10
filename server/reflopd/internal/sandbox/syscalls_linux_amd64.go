// SPDX-License-Identifier: Apache-2.0

package sandbox

import "golang.org/x/sys/unix"

const (
	auditArch = 0xc000003e // AUDIT_ARCH_X86_64
	x32Bit    = 0x40000000 // __X32_SYSCALL_BIT
)

var archAllowSyscalls = []int{
	unix.SYS_OPEN, unix.SYS_STAT, unix.SYS_LSTAT, unix.SYS_ACCESS, unix.SYS_PIPE, unix.SYS_SELECT,
	unix.SYS_POLL, unix.SYS_EPOLL_CREATE, unix.SYS_EPOLL_WAIT, unix.SYS_EVENTFD, unix.SYS_DUP2,
	unix.SYS_GETDENTS, unix.SYS_RENAME, unix.SYS_MKDIR, unix.SYS_RMDIR, unix.SYS_UNLINK,
	unix.SYS_LINK, unix.SYS_SYMLINK, unix.SYS_READLINK, unix.SYS_CHMOD, unix.SYS_TIME,
	unix.SYS_ARCH_PRCTL, unix.SYS_SIGNALFD, unix.SYS_GETPGRP, unix.SYS_ALARM, unix.SYS_PAUSE,
	unix.SYS_MAP_SHADOW_STACK,
}

var archDenySyscalls = []int{
	unix.SYS_FORK, unix.SYS_VFORK, unix.SYS_IOPL, unix.SYS_IOPERM, unix.SYS_CHOWN, unix.SYS_LCHOWN,
	unix.SYS_USELIB, unix.SYS_KEXEC_FILE_LOAD, unix.SYS_MODIFY_LDT,
}
