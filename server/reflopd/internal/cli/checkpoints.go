// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/chaseungjoon/reflop/server/reflopd/internal/config"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/gpu"
)

// checkpointsCmd turns Guest checkpoints on or off (ADR-029).
func checkpointsCmd() *cobra.Command {
	var interval time.Duration
	cmd := &cobra.Command{
		Use:   "checkpoints [on|off]",
		Short: "Let Guests save checkpoints of their sessions to this machine's disk (off by default)",
		Long: "Guests can save their session's state (parameters, optimizer state, buffers) at most once per\n" +
			"interval, so that a worker crash or a restart loses at most that much work.\n\n" +
			"Every checkpoint writes the Guest's whole state to this machine's disk, which wears out SSDs,\n" +
			"and the state's GPU memory doubles while it is copied. The interval cannot be below " +
			config.MinCheckpointInterval.String() + "\n(default " + config.DefaultCheckpointInterval.String() + ").",
		Args:      cobra.MaximumNArgs(1),
		ValidArgs: []string{"on", "off"},
		RunE: func(cmd *cobra.Command, args []string) error {
			out := cmd.OutOrStdout()
			l, err := layout()
			if err != nil {
				return err
			}
			s, err := config.LoadSettings(settingsPath(l))
			if err != nil {
				return err
			}
			cur := config.Defaults().Checkpoints
			if s.Checkpoints != nil {
				cur = *s.Checkpoints
			}
			if len(args) == 0 {
				if cmd.Flags().Changed("interval") {
					return exitWith(ExitUsage, errors.New("pass `on --interval DURATION` to change the interval"))
				}
				printCheckpointState(out, cur)
				return nil
			}
			switch args[0] {
			case "on":
				cur.Enabled = true
				if cmd.Flags().Changed("interval") {
					cur.Interval = config.Duration(interval)
				}
			case "off":
				cur.Enabled = false
			default:
				return exitWith(ExitUsage, fmt.Errorf("unknown argument %q: use on or off", args[0]))
			}
			if err := cur.Validate(); err != nil {
				return exitWith(ExitUsage, err)
			}
			s.Checkpoints = &cur
			if err := config.SaveSettings(settingsPath(l), s); err != nil {
				return err
			}
			notifyReload()
			if cur.Enabled {
				warn := checkpointWarning(time.Duration(cur.Interval), gpu.Detect().GPUs)
				fmt.Fprintln(out, red(out, warn))
			}
			printCheckpointState(out, cur)
			return nil
		},
	}
	cmd.Flags().DurationVar(&interval, "interval", config.DefaultCheckpointInterval,
		"shortest time between two checkpoints of a session, and Guests' default (at least "+
			config.MinCheckpointInterval.String()+")")
	return cmd
}

func printCheckpointState(out io.Writer, c config.Checkpoints) {
	if !c.Enabled {
		fmt.Fprintf(out, "checkpoints off (interval when on: %s)\n", time.Duration(c.Interval))
		return
	}
	fmt.Fprintf(out, "checkpoints on: Guests may save one every %s or less often\n", time.Duration(c.Interval))
}

// checkpointWarning is printed every time the Owner turns checkpoints on.
// The figures are the worst case: Guest state that fills every GPU.
func checkpointWarning(interval time.Duration, gpus []gpu.GPU) string {
	perHour := float64(time.Hour) / float64(interval)
	var totalGiB float64
	var names []string
	for _, g := range gpus {
		gib := float64(g.MemoryMiB) / 1024
		totalGiB += gib
		names = append(names, fmt.Sprintf("%.0f GiB on %s", gib, g.Name))
	}
	lines := []string{
		"WARNING: checkpoints wear out SSDs and use extra GPU memory.",
		fmt.Sprintf("  Every %s, each Guest session that uses checkpoints writes its whole state", interval),
		"  (parameters, optimizer state, buffers) to this machine's disk.",
	}
	if totalGiB > 0 {
		n := totalGiB * perHour
		lines = append(lines,
			fmt.Sprintf("  If Guest state fills the GPU memory (%s), that is", strings.Join(names, ", ")),
			fmt.Sprintf("  up to %.0f GiB written per hour of use (%s per day).", n, gibOrTiB(n*24)))
	}
	lines = append(lines,
		fmt.Sprintf("  Each GiB of Guest state writes %.0f GiB per hour.", perHour),
		"  While a checkpoint is copied, the GPU memory for the Guest's state doubles.")
	return strings.Join(lines, "\n")
}

func gibOrTiB(gib float64) string {
	if gib >= 1024 {
		return fmt.Sprintf("%.1f TiB", gib/1024)
	}
	return fmt.Sprintf("%.0f GiB", gib)
}

// red colors s when out is a terminal and NO_COLOR is unset.
func red(out io.Writer, s string) string {
	if !colorOK(out) {
		return s
	}
	return "\x1b[1;31m" + s + "\x1b[0m"
}

func colorOK(out io.Writer) bool {
	if os.Getenv("NO_COLOR") != "" {
		return false
	}
	f, ok := out.(*os.File)
	if !ok {
		return false
	}
	st, err := f.Stat()
	return err == nil && st.Mode()&os.ModeCharDevice != 0
}
