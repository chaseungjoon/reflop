// SPDX-License-Identifier: Apache-2.0

package server

import (
	"net"

	"golang.org/x/sys/unix"
)

func umask(m int) int { return unix.Umask(m) }

func peerUID(c *net.UnixConn) (int, bool) {
	raw, err := c.SyscallConn()
	if err != nil {
		return 0, false
	}
	var cred *unix.Ucred
	var cerr error
	raw.Control(func(fd uintptr) { cred, cerr = unix.GetsockoptUcred(int(fd), unix.SOL_SOCKET, unix.SO_PEERCRED) })
	if cerr != nil || cred == nil {
		return 0, false
	}
	return int(cred.Uid), true
}
