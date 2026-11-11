// SPDX-License-Identifier: Apache-2.0
// Adapted from recom (same sole author; relicensed under Apache-2.0, ADR-007).

package cli

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/chaseungjoon/reflop/server/reflopd/internal/buildinfo"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/selfupdate"
)

func updateCmd() *cobra.Command {
	var check, force bool
	var version string
	cmd := &cobra.Command{
		Use:   "update",
		Short: "Update reflopd to the latest release",
		Long: `Download the latest reflopd release from GitHub, verify it against the
release's SHA-256 checksums, and atomically replace this binary. Then run
` + "`reflopd runtime install`" + ` if the release pins a new runtime, and restart.

  reflopd update                  install the latest release if it is newer
  reflopd update --check          only report whether an update is available
  reflopd update --version v0.2.0 install a specific release (also downgrades)

Every release is checked against an offline signature on its checksums.txt
before anything from it runs. Releases before 0.1.6 are unsigned and are
refused.

The installer puts reflopd in ~/.local/bin, so no sudo is needed. If reflopd
lives in a directory you cannot write to (e.g. /usr/local/bin), run
` + "`sudo reflopd update`" + `, or reinstall into ~/.local/bin and remove the old copy.`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			out, errw := cmd.OutOrStdout(), cmd.ErrOrStderr()
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
			defer cancel()
			u, err := selfupdate.New()
			if err != nil {
				return err
			}
			exe, err := selfupdate.Executable()
			if err != nil {
				return fmt.Errorf("cannot locate the reflopd binary: %w", err)
			}
			if pm := selfupdate.ManagedBy(exe); pm != "" && !check {
				return fmt.Errorf("%s was installed with %s, which no longer distributes reflopd; remove it (brew uninstall reflopd) and reinstall with:\n  %s", exe, pm, selfupdate.InstallScript)
			}
			cur, curOK := selfupdate.Parse(buildinfo.Version)
			tag := version
			if tag == "" {
				if tag, err = u.Latest(ctx); err != nil {
					return err
				}
			} else if !strings.HasPrefix(tag, "v") {
				tag = "v" + tag
			}
			want, ok := selfupdate.Parse(tag)
			if !ok {
				return exitWith(ExitUsage, fmt.Errorf("invalid version %q (expected e.g. v0.2.0)", version))
			}
			current := buildinfo.Version
			switch {
			case !curOK:
				fmt.Fprintf(errw, "this is a development build (%s); the latest release is %s", current, tag)
				fmt.Fprintln(errw)
				if check {
					return nil
				}
				if !force && version == "" {
					return errors.New("not replacing a development build; pass --force to install the release anyway")
				}
			case version == "" && !cur.Less(want) && !force:
				fmt.Fprintf(out, "reflopd %s is up to date (latest release: %s)\n", cur, tag)
				return nil
			case check:
				fmt.Fprintf(out, "update available: reflopd %s → %s (run `reflopd update`)\n", cur, want)
				return nil
			}
			if check {
				return nil
			}
			fmt.Fprintf(errw, "downloading %s", u.Archive(tag))
			fmt.Fprintln(errw)
			bin, err := u.Fetch(ctx, tag)
			if err != nil {
				return err
			}
			fmt.Fprintf(errw, "checksum verified")
			fmt.Fprintln(errw)
			if err := selfupdate.Install(exe, bin, want.String()); err != nil {
				if errors.Is(err, selfupdate.ErrPermission) {
					return fmt.Errorf("%v; reinstall into ~/.local/bin (no sudo) and remove %s:\n  %s\nor run `sudo reflopd update`", err, exe, selfupdate.InstallScript)
				}
				return err
			}
			from := current
			if curOK {
				from = cur.String()
			}
			fmt.Fprintf(out, "updated %s: reflopd %s → %s\n", exe, from, want)
			fmt.Fprintln(errw, "next: reflopd runtime install && reflopd stop && reflopd up")
			return nil
		},
	}
	cmd.Flags().BoolVar(&check, "check", false, "only check whether a newer release exists")
	cmd.Flags().StringVar(&version, "version", "", "install this release tag instead of the latest (e.g. v0.2.0)")
	cmd.Flags().BoolVar(&force, "force", false, "reinstall even if already up to date (or replace a development build)")
	return cmd
}
