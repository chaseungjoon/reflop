// SPDX-License-Identifier: Apache-2.0

// Command reflopd is the reFLOP server daemon.
package main

import (
	"os"
	"os/exec"
	"os/user"
	"strconv"
	"syscall"

	"github.com/chaseungjoon/reflop/server/reflopd/internal/cli"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/sandbox"
)

func main() {
	if len(os.Args) > 1 {
		switch os.Args[1] {
		case sandbox.HelperArg:
			sandbox.RunHelper() // never returns
			return
		case "__userns-probe":
			os.Exit(0)
		}
	}
	args := os.Args[1:]
	// `--system` targets the level 1 install: run as the reflop user with
	// REFLOP_SYSTEM=1 so files keep the service's ownership.
	if len(args) > 0 && args[0] == "--system" {
		args = args[1:]
		os.Setenv("REFLOP_SYSTEM", "1")
		if os.Geteuid() == 0 && !(len(args) > 0 && (args[0] == "install" || args[0] == "uninstall")) {
			if u, err := user.Lookup("reflop"); err == nil {
				uid, _ := strconv.Atoi(u.Uid)
				gid, _ := strconv.Atoi(u.Gid)
				exe, _ := os.Executable()
				c := exec.Command(exe, args...)
				c.Env = append(os.Environ(), "REFLOP_SYSTEM=1", "HOME=/var/lib/reflop")
				c.Stdin, c.Stdout, c.Stderr = os.Stdin, os.Stdout, os.Stderr
				c.SysProcAttr = &syscall.SysProcAttr{Credential: &syscall.Credential{Uid: uint32(uid), Gid: uint32(gid)}}
				if err := c.Run(); err != nil {
					if ee, ok := err.(*exec.ExitError); ok {
						os.Exit(ee.ExitCode())
					}
					os.Exit(1)
				}
				os.Exit(0)
			}
		}
	}
	os.Exit(cli.Main(args, os.Stdout, os.Stderr))
}
