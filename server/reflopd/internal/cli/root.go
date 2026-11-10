// SPDX-License-Identifier: Apache-2.0

// Package cli implements the reflopd command line (PLAN.md §16.1).
package cli

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/chaseungjoon/reflop/server/reflopd/internal/buildinfo"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/paths"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/server"
)

// Exit codes.
const (
	ExitOK     = 0
	ExitError  = 1
	ExitUsage  = 2
	ExitNotRun = 3
	ExitUnsafe = 4
)

type exitError struct {
	code int
	err  error
}

func (e *exitError) Error() string { return e.err.Error() }

func exitWith(code int, err error) error { return &exitError{code, err} }

// Main runs the CLI and returns the process exit code.
func Main(args []string, stdout, stderr io.Writer) int {
	root := newRoot(stdout, stderr)
	root.SetArgs(args)
	root.SetOut(stdout)
	root.SetErr(stderr)
	if err := root.Execute(); err != nil {
		fmt.Fprintf(stderr, "reflopd: %v\n", err)
		if e, ok := err.(*exitError); ok {
			return e.code
		}
		return ExitError
	}
	return ExitOK
}

func newRoot(stdout, stderr io.Writer) *cobra.Command {
	root := &cobra.Command{
		Use:   "reflopd",
		Short: "reFLOP server: lend your GPU as a calculator",
		Long: `reflopd runs on the machine with the GPU. Guests connect with the reflop
client and send operator graphs and tensors; reflopd validates them and runs
whitelisted tensor operators in sandboxed workers. Guests never get a shell.

  reflopd up                       first-time setup and start (rootless)
  reflopd invite --name alice      create a one-time pairing code
  reflopd status | top | audit     see who is using the GPU
  reflopd usage                    compute used per Guest (device time, FLOPs)
  reflopd pause | resume | kill    Owner controls
  reflopd doctor                   check the sandbox and the runtime`,
		SilenceUsage:  true,
		SilenceErrors: true,
	}
	root.AddCommand(
		upCmd(), serveCmd(), stopCmd(), statusCmd(), installCmd(), uninstallCmd(),
		inviteCmd(), usersCmd(), revokeCmd(), quotaCmd(), railsCmd(),
		pauseCmd(), resumeCmd(), killCmd(), yieldCmd(), sayCmd(), topCmd(), usageCmd(), auditCmd(),
		doctorCmd(), selftestCmd(), runtimeCmd(), updateCmd(), versionCmd(),
	)
	return root
}

func versionCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "version",
		Short: "Print the version",
		Args:  cobra.NoArgs,
		Run: func(cmd *cobra.Command, _ []string) {
			c := buildinfo.ShortCommit()
			if c != "" {
				c = " (" + c + ")"
			}
			fmt.Fprintf(cmd.OutOrStdout(), "reflopd %s%s %s\n", buildinfo.Version, c, buildinfo.Platform())
		},
	}
}

func layout() (paths.Layout, error) { return paths.Resolve() }

// control sends a command to the running daemon.
func control(cmd string, args any, out any) error {
	l, err := layout()
	if err != nil {
		return err
	}
	req := server.ControlRequest{Cmd: cmd}
	if args != nil {
		req.Args, _ = json.Marshal(args)
	}
	resp, err := server.ControlClient(l, req, 30*time.Second)
	if err != nil {
		if strings.Contains(err.Error(), "not running") {
			return exitWith(ExitNotRun, err)
		}
		return err
	}
	if out != nil && len(resp.Data) > 0 {
		return json.Unmarshal(resp.Data, out)
	}
	return nil
}

// notifyReload asks a running daemon to re-read identities and settings.
// It is fine if no daemon runs: the next start reads the files.
func notifyReload() { _ = control("reload", nil, nil) }

func humanBytes(n uint64) string {
	units := []string{"B", "KiB", "MiB", "GiB", "TiB"}
	v := float64(n)
	i := 0
	for v >= 1024 && i < len(units)-1 {
		v /= 1024
		i++
	}
	if i == 0 {
		return fmt.Sprintf("%d B", n)
	}
	return fmt.Sprintf("%.1f %s", v, units[i])
}

func isTerminal(f *os.File) bool {
	st, err := f.Stat()
	return err == nil && st.Mode()&os.ModeCharDevice != 0
}
