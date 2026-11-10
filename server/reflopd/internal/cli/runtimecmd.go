// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"

	"github.com/chaseungjoon/reflop/server/reflopd/internal/gpu"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/runtime"
)

func runtimeCmd() *cobra.Command {
	cmd := &cobra.Command{Use: "runtime", Short: "Manage the pinned compute runtime (Owner only)"}
	var variant string
	install := &cobra.Command{
		Use:     "install",
		Aliases: []string{"upgrade"},
		Short:   "Install (or upgrade to) the runtime pinned by this reflopd build",
		Long: `Download the CPython and PyTorch build pinned by this reflopd release and
verify every file's SHA-256. After ` + "`reflopd update`" + `, run this to move to the
newly pinned runtime, then restart reflopd. Guests can never trigger it.`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			l, err := layout()
			if err != nil {
				return err
			}
			if err := l.Ensure(); err != nil {
				return err
			}
			lock, err := runtime.Load()
			if err != nil {
				return err
			}
			ch, err := lock.Choose(gpu.Detect(), variant)
			if err != nil {
				return err
			}
			ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
			defer stop()
			out := cmd.OutOrStdout()
			inst, err := lock.Install(ctx, l.Runtime, ch.Variant, func(f string, a ...any) { fmt.Fprintf(out, "  "+f+"\n", a...) })
			if err != nil {
				return err
			}
			fmt.Fprintf(out, "installed %s; restart reflopd to use it (reflopd stop && reflopd up)\n", inst.Marker.ID)
			return nil
		},
	}
	install.Flags().StringVar(&variant, "variant", "", "force a variant (cu132, cu126, cpu)")
	list := &cobra.Command{
		Use:   "list",
		Short: "List installed runtimes",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			l, err := layout()
			if err != nil {
				return err
			}
			for _, r := range runtime.List(l.Runtime) {
				fmt.Fprintf(cmd.OutOrStdout(), "%s  torch %s  CUDA %s  installed %s\n", r.Marker.ID, r.Marker.Torch,
					orNone(r.Marker.CUDA), r.Marker.Installed.Format("2006-01-02"))
			}
			return nil
		},
	}
	prune := &cobra.Command{
		Use:   "prune",
		Short: "Remove runtimes other than the one this build uses, and the download cache",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			l, err := layout()
			if err != nil {
				return err
			}
			lock, err := runtime.Load()
			if err != nil {
				return err
			}
			ch, err := lock.Choose(gpu.Detect(), "")
			if err != nil {
				return err
			}
			arch, _ := runtime.Arch()
			keep := lock.ID(ch.Variant, arch)
			for _, r := range runtime.List(l.Runtime) {
				if r.Marker.ID != keep {
					if err := runtime.Remove(r.Dir); err != nil {
						return err
					}
					fmt.Fprintf(cmd.OutOrStdout(), "removed %s\n", r.Marker.ID)
				}
			}
			os.RemoveAll(l.Runtime + "/downloads")
			return nil
		},
	}
	cmd.AddCommand(install, list, prune)
	return cmd
}
