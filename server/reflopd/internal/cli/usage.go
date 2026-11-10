// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/spf13/cobra"

	"github.com/chaseungjoon/reflop/server/reflopd/internal/usage"
)

func usageCmd() *cobra.Command {
	var since time.Duration
	var asJSON bool
	cmd := &cobra.Command{
		Use:   "usage [NAME]",
		Short: "How much compute each Guest has used (device time, FLOPs, traffic)",
		Long: `Show cumulative compute per Guest identity.

DEVICE TIME is the primary measure: the time the device spent running the
Guest's executes, measured by the worker (like cloud GPU-hours). EST. TFLOP is
the server's static FLOP estimate (it undercounts memory-bound work). SESSION
TIME is how long the Guest's sessions held a worker (and its device memory);
BUSY is DEVICE TIME / SESSION TIME.

Without --since, totals are lifetime (kept in usage.json in the state
directory). With --since, they are recomputed from the audit log, which
only covers its retention window.`,
		Args: cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			rep, err := loadUsage(since)
			if err != nil {
				return err
			}
			name := ""
			if len(args) == 1 {
				name = args[0]
				if _, ok := rep.Identities[name]; !ok {
					return fmt.Errorf("no usage recorded for %q", name)
				}
				for k := range rep.Identities {
					if k != name {
						delete(rep.Identities, k)
					}
				}
			}
			if asJSON {
				enc := json.NewEncoder(cmd.OutOrStdout())
				enc.SetIndent("", "  ")
				return enc.Encode(rep)
			}
			return printUsage(cmd.OutOrStdout(), rep, name != "", time.Now())
		},
	}
	cmd.Flags().DurationVar(&since, "since", 0, "only the last DURATION, from the audit log (e.g. 24h)")
	cmd.Flags().BoolVar(&asJSON, "json", false, "JSON output")
	return cmd
}

// loadUsage asks the running daemon (live totals), falls back to the saved
// ledger when it is stopped, and to the audit log when there is no ledger,
// the daemon predates usage accounting, or a window is asked for.
func loadUsage(since time.Duration) (usage.Report, error) {
	l, err := layout()
	if err != nil {
		return usage.Report{}, err
	}
	now := time.Now()
	if since > 0 {
		return usage.FromAudit(l.Audit(), now.Add(-since), now)
	}
	var rep usage.Report
	err = control("usage", nil, &rep)
	if err == nil {
		rep.Source = "live from reflopd"
		return rep, nil
	}
	var ee *exitError
	if errors.As(err, &ee) && ee.code == ExitNotRun {
		if rep, err := usage.Load(filepath.Join(l.State, "usage.json")); err == nil {
			rep.Source = "saved ledger (reflopd is not running)"
			rep.Until = now
			return rep, nil
		} else if !errors.Is(err, os.ErrNotExist) {
			return usage.Report{}, err
		}
	} else if !strings.Contains(err.Error(), "unknown command") {
		return usage.Report{}, err
	}
	rep, err = usage.FromAudit(l.Audit(), time.Time{}, now)
	rep.Source = "audit log (no usage ledger yet; covers the log's retention window)"
	return rep, err
}

func printUsage(out io.Writer, rep usage.Report, detail bool, now time.Time) error {
	if len(rep.Identities) == 0 {
		fmt.Fprintf(out, "no compute used since %s\n", rep.Since.Local().Format("2006-01-02 15:04"))
		return nil
	}
	fmt.Fprintf(out, "compute used since %s, %s\n", rep.Since.Local().Format("2006-01-02 15:04"), rep.Source)
	names := make([]string, 0, len(rep.Identities))
	var total usage.Totals
	for n, t := range rep.Identities {
		names = append(names, n)
		total.DeviceUs += t.DeviceUs
		total.FLOPs += t.FLOPs
		total.Executes += t.Executes
		total.Sessions += t.Sessions
		total.SessionUs += t.SessionUs
		total.BytesUp += t.BytesUp
		total.BytesDown += t.BytesDown
	}
	sort.Slice(names, func(i, j int) bool {
		a, b := rep.Identities[names[i]], rep.Identities[names[j]]
		if a.DeviceUs != b.DeviceUs {
			return a.DeviceUs > b.DeviceUs
		}
		return names[i] < names[j]
	})
	tw := tabwriter.NewWriter(out, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "IDENTITY\tDEVICE TIME\tSHARE\tBUSY\tEST. TFLOP\tEXECUTES\tSESSIONS\tSESSION TIME\tUP\tDOWN\tLAST ACTIVE")
	row := func(name string, t *usage.Totals, last string) {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%.1f\t%d\t%d\t%s\t%s\t%s\t%s\n", name, fmtUs(t.DeviceUs),
			pct(t.DeviceUs, total.DeviceUs), pct(t.DeviceUs, t.SessionUs), t.FLOPs/1e12, t.Executes, t.Sessions,
			fmtUs(t.SessionUs), humanBytes(t.BytesUp), humanBytes(t.BytesDown), last)
	}
	for _, n := range names {
		t := rep.Identities[n]
		last := "-"
		if !t.Last.IsZero() {
			last = ago(now.Sub(t.Last))
		}
		row(n, t, last)
	}
	if len(names) > 1 {
		row("TOTAL", &total, "")
	}
	if err := tw.Flush(); err != nil {
		return err
	}
	if detail {
		t := rep.Identities[names[0]]
		devs := make([]string, 0, len(t.ByDevice))
		for d := range t.ByDevice {
			devs = append(devs, d)
		}
		sort.Strings(devs)
		for _, d := range devs {
			fmt.Fprintf(out, "  on %s: %s\n", d, fmtUs(t.ByDevice[d]))
		}
		if t.DeviceUs > 0 {
			fmt.Fprintf(out, "  average: %.1f ms per execute, %.2f TFLOP/s estimated\n",
				float64(t.DeviceUs)/1e3/float64(max(t.Executes, 1)), t.FLOPs/1e12/(float64(t.DeviceUs)/1e6))
		}
		if t.PeakMemory > 0 {
			fmt.Fprintf(out, "  largest execute peak memory: %s\n", humanBytes(t.PeakMemory))
		}
		if !t.First.IsZero() {
			fmt.Fprintf(out, "  first seen %s, last %s\n", t.First.Local().Format("2006-01-02 15:04"), t.Last.Local().Format("2006-01-02 15:04"))
		}
	}
	return nil
}

func fmtUs(us uint64) string {
	d := time.Duration(us) * time.Microsecond
	switch {
	case d > 0 && d < time.Second:
		return fmt.Sprintf("%.0fms", float64(d.Microseconds())/1e3)
	case d < time.Minute:
		return fmt.Sprintf("%.1fs", d.Seconds())
	case d < time.Hour:
		return fmt.Sprintf("%dm%02ds", int(d.Minutes()), int(d.Seconds())%60)
	case d < 48*time.Hour:
		return fmt.Sprintf("%dh%02dm", int(d.Hours()), int(d.Minutes())%60)
	}
	return fmt.Sprintf("%dd%02dh", int(d.Hours())/24, int(d.Hours())%24)
}

func pct(a, b uint64) string {
	if b == 0 {
		return "-"
	}
	return fmt.Sprintf("%.0f%%", 100*float64(a)/float64(b))
}

func ago(d time.Duration) string {
	if d < time.Minute {
		return "just now"
	}
	return fmtUs(uint64(d.Microseconds())) + " ago"
}
