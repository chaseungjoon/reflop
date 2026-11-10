// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"bufio"
	"context"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/spf13/cobra"

	"github.com/chaseungjoon/reflop/server/reflopd/internal/audit"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/auth"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/config"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/paths"
	pb "github.com/chaseungjoon/reflop/server/reflopd/internal/pb/reflopv1"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/server"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/sshpipe"
)

func makeInvite(l paths.Layout, name string, ttl time.Duration, listen []string) (string, error) {
	ids, err := auth.Open(l.Identities())
	if err != nil {
		return "", err
	}
	key, err := auth.LoadOrCreateServerKey(l.ServerKey(), l.ServerCert())
	if err != nil {
		return "", err
	}
	inv, err := ids.CreateInvite(name, ttl)
	if err != nil {
		return "", err
	}
	cfg, _ := config.Load(l.ConfigFile(), filepath.Join(l.State, "settings.json"))
	addrs := cfg.Advertise
	if len(addrs) == 0 && len(listen) == 0 {
		// Prefer what the running daemon actually listens on.
		var st server.StatusView
		if control("status", nil, &st) == nil && len(st.Listen) > 0 {
			listen = st.Listen
		}
	}
	if len(addrs) == 0 {
		addrs = advertised(listen, cfg.Listen)
	}
	idb, _ := hex.DecodeString(inv.ID)
	code, err := auth.EncodeInvite(&pb.InviteCode{Version: 1, Transport: pb.TransportKind_TRANSPORT_TLS, Addresses: addrs,
		ServerKeyFingerprint: key.Fingerprint[:], InviteId: idb, Secret: inv.Secret, ExpiresUnix: inv.Expires.Unix(),
		IdentityName: name})
	if err != nil {
		return "", err
	}
	audit.Open(l.Audit(), 0)
	return code, nil
}

// advertised turns listen addresses into addresses Guests can dial; a
// wildcard bind becomes every non-loopback interface address.
func advertised(flags, cfg []string) []string {
	src := cfg
	if len(flags) > 0 {
		src = flags
	}
	var out []string
	for _, a := range src {
		h, p, err := net.SplitHostPort(a)
		if err != nil {
			continue
		}
		ip := net.ParseIP(h)
		if h == "" || (ip != nil && ip.IsUnspecified()) {
			ifs, _ := net.InterfaceAddrs()
			for _, ia := range ifs {
				if n, ok := ia.(*net.IPNet); ok && !n.IP.IsLoopback() && !n.IP.IsLinkLocalUnicast() {
					out = append(out, net.JoinHostPort(n.IP.String(), p))
				}
			}
			continue
		}
		out = append(out, a)
	}
	return out
}

func inviteCmd() *cobra.Command {
	var name, via, key, sshTarget string
	var addresses []string
	var expires time.Duration
	var yes bool
	cmd := &cobra.Command{
		Use:   "invite --name <name>",
		Short: "Create a one-time pairing code for a Guest",
		Long: `Create a one-time invite. The code carries the server's addresses, its key
fingerprint (the Guest pins it), and a one-time secret.

  reflopd invite --name alice                 daemon mode (TLS)
  reflopd invite --name alice --expires 2h
  reflopd invite --name alice --via ssh --key "ssh-ed25519 AAAA… alice@mac" --ssh-target me@gpu-box

SSH pipe mode adds the Guest's key to authorized_keys with
restrict,command="reflopd serve --stdio …": SSH becomes a pipe that can only
run that command. Get the key on the Guest machine with: reflop key --ssh`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			out := cmd.OutOrStdout()
			l, err := layout()
			if err != nil {
				return err
			}
			if name == "" {
				return exitWith(ExitUsage, errors.New("--name is required"))
			}
			switch via {
			case "", "tls":
				code, err := makeInvite(l, name, expires, addresses)
				if err != nil {
					return err
				}
				fmt.Fprintf(out, "Invite for %q (expires %s):\n\n  %s\n\nOn the Guest machine: reflop pair <code>\n", name,
					time.Now().Add(expires).Format(time.RFC1123), code)
				return nil
			case "ssh":
				return inviteSSH(cmd, l, name, key, sshTarget, yes)
			default:
				return exitWith(ExitUsage, fmt.Errorf("--via must be tls or ssh"))
			}
		},
	}
	cmd.Flags().StringVar(&name, "name", "", "identity name for the Guest (a-z 0-9 _ -)")
	cmd.Flags().DurationVar(&expires, "expires", 24*time.Hour, "invite lifetime")
	cmd.Flags().StringSliceVar(&addresses, "address", nil, "address(es) Guests should dial (default: where reflopd listens)")
	cmd.Flags().StringVar(&via, "via", "tls", "transport: tls (daemon) or ssh (forced-command pipe)")
	cmd.Flags().StringVar(&key, "key", "", "ssh: the Guest's public key (from `reflop key --ssh`)")
	cmd.Flags().StringVar(&sshTarget, "ssh-target", "", "ssh: user@host[:port] the Guest should connect to")
	cmd.Flags().BoolVarP(&yes, "yes", "y", false, "ssh: append to authorized_keys without asking")
	return cmd
}

func inviteSSH(cmd *cobra.Command, l paths.Layout, name, key, target string, yes bool) error {
	out := cmd.OutOrStdout()
	pub, comment, err := sshpipe.ParseEd25519(key)
	if err != nil {
		return exitWith(ExitUsage, err)
	}
	ids, err := auth.Open(l.Identities())
	if err != nil {
		return err
	}
	id, err := ids.AddKey(name, pub, "ssh", comment)
	if err != nil {
		return err
	}
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	exe, _ = filepath.EvalSymlinks(exe)
	line := sshpipe.AuthorizedKeysLine(exe, id.ID, key, l)
	ak, err := sshpipe.AuthorizedKeysPath()
	if err != nil {
		return err
	}
	fmt.Fprintf(out, "authorized_keys line for %q:\n\n  %s\n\n", name, line)
	ok := yes
	if !ok && isTerminal(os.Stdin) {
		fmt.Fprintf(out, "Append it to %s? [y/N] ", ak)
		r := bufio.NewReader(os.Stdin)
		s, _ := r.ReadString('\n')
		ok = strings.HasPrefix(strings.ToLower(strings.TrimSpace(s)), "y")
	}
	if ok {
		if err := sshpipe.Append(ak, line); err != nil {
			return err
		}
		fmt.Fprintf(out, "appended to %s\n", ak)
	} else {
		fmt.Fprintln(out, "not changed; add the line yourself")
	}
	key2, err := auth.LoadOrCreateServerKey(l.ServerKey(), l.ServerCert())
	if err != nil {
		return err
	}
	if target == "" {
		h, _ := os.Hostname()
		target = os.Getenv("USER") + "@" + h
	}
	code, err := auth.EncodeInvite(&pb.InviteCode{Version: 1, Transport: pb.TransportKind_TRANSPORT_SSH,
		Addresses: []string{target}, ServerKeyFingerprint: key2.Fingerprint[:], IdentityName: name, ReflopdPath: exe,
		ExpiresUnix: time.Now().Add(365 * 24 * time.Hour).Unix()})
	if err != nil {
		return err
	}
	fmt.Fprintf(out, "\nGuest pairing code (SSH):\n\n  %s\n\nOn the Guest machine: reflop pair <code>\n", code)
	notifyReload()
	return nil
}

func usersCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "users",
		Short: "List paired identities",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			l, err := layout()
			if err != nil {
				return err
			}
			ids, err := auth.Open(l.Identities())
			if err != nil {
				return err
			}
			tw := tabwriter.NewWriter(cmd.OutOrStdout(), 0, 4, 2, ' ', 0)
			fmt.Fprintln(tw, "NAME\tKEYS\tVIA\tCREATED\tSTATE")
			for _, id := range ids.List() {
				via := map[string]bool{}
				for _, k := range id.Keys {
					via[k.Via] = true
				}
				var vs []string
				for v := range via {
					vs = append(vs, v)
				}
				sort.Strings(vs)
				state := "active"
				if id.Revoked {
					state = "revoked"
				}
				fmt.Fprintf(tw, "%s\t%d\t%s\t%s\t%s\n", id.Name, len(id.Keys), strings.Join(vs, ","), id.Created.Format("2006-01-02"), state)
			}
			return tw.Flush()
		},
	}
}

func revokeCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "revoke <name>",
		Short: "Revoke an identity: its keys stop working and its sessions end now",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			l, err := layout()
			if err != nil {
				return err
			}
			ids, err := auth.Open(l.Identities())
			if err != nil {
				return err
			}
			id, err := ids.Revoke(args[0])
			if err != nil {
				return err
			}
			if ak, err := sshpipe.AuthorizedKeysPath(); err == nil {
				if n, err := sshpipe.RemoveIdentity(ak, id.ID); err == nil && n > 0 {
					fmt.Fprintf(cmd.OutOrStdout(), "removed %d line(s) from %s\n", n, ak)
				}
			}
			notifyReload()
			_ = control("kill", map[string]string{"target": args[0]}, nil)
			fmt.Fprintf(cmd.OutOrStdout(), "revoked %q\n", args[0])
			return nil
		},
	}
}

func settingsPath(l paths.Layout) string { return filepath.Join(l.State, "settings.json") }

func quotaCmd() *cobra.Command {
	cmd := &cobra.Command{Use: "quota", Short: "Optional per-identity quotas (none are set by default)"}
	var q config.Quota
	var devMem, storage, bw string
	set := &cobra.Command{
		Use:   "set <name>",
		Short: "Set quotas for an identity",
		Long: `Quotas are about fairness and are off by default: Owner and Guests agree on
usage among themselves. Only the values you pass change.

  reflopd quota set alice --device-memory 6GiB --max-sessions 1
  reflopd quota set alice --execute-wall-ms 60000 --storage 50GiB --upload-bandwidth 20MiB`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			l, err := layout()
			if err != nil {
				return err
			}
			s, err := config.LoadSettings(settingsPath(l))
			if err != nil {
				return err
			}
			if s.Quotas == nil {
				s.Quotas = map[string]config.Quota{}
			}
			cur := s.Quotas[args[0]]
			f := cmd.Flags()
			parse := func(flag, v string, dst *config.Bytes) error {
				if !f.Changed(flag) {
					return nil
				}
				b, err := config.ParseBytes(v)
				*dst = b
				return err
			}
			if err := parse("device-memory", devMem, &cur.DeviceMemoryBytes); err != nil {
				return err
			}
			if err := parse("storage", storage, &cur.StorageBytes); err != nil {
				return err
			}
			if err := parse("upload-bandwidth", bw, &cur.UploadBytesPerSec); err != nil {
				return err
			}
			if f.Changed("max-sessions") {
				cur.MaxSessions = q.MaxSessions
			}
			if f.Changed("execute-wall-ms") {
				cur.ExecuteWallMS = q.ExecuteWallMS
			}
			if f.Changed("execute-flops") {
				cur.ExecuteFLOPs = q.ExecuteFLOPs
			}
			s.Quotas[args[0]] = cur
			if err := config.SaveSettings(settingsPath(l), s); err != nil {
				return err
			}
			notifyReload()
			fmt.Fprintf(cmd.OutOrStdout(), "quota for %q: %s\n", args[0], describeQuota(cur))
			return nil
		},
	}
	set.Flags().StringVar(&devMem, "device-memory", "", "max device memory per graph")
	set.Flags().Uint32Var(&q.MaxSessions, "max-sessions", 0, "max concurrent sessions")
	set.Flags().Uint64Var(&q.ExecuteWallMS, "execute-wall-ms", 0, "wall-clock limit per execute (kills the worker)")
	set.Flags().Uint64Var(&q.ExecuteFLOPs, "execute-flops", 0, "estimated FLOPs per execute")
	set.Flags().StringVar(&storage, "storage", "", "dataset storage")
	set.Flags().StringVar(&bw, "upload-bandwidth", "", "dataset upload bandwidth per second")
	clear := &cobra.Command{
		Use:   "clear <name>",
		Short: "Remove all quotas for an identity",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			l, err := layout()
			if err != nil {
				return err
			}
			s, err := config.LoadSettings(settingsPath(l))
			if err != nil {
				return err
			}
			delete(s.Quotas, args[0])
			if err := config.SaveSettings(settingsPath(l), s); err != nil {
				return err
			}
			notifyReload()
			fmt.Fprintf(cmd.OutOrStdout(), "cleared quotas for %q\n", args[0])
			return nil
		},
	}
	show := &cobra.Command{
		Use:   "show",
		Short: "Show quotas",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			l, err := layout()
			if err != nil {
				return err
			}
			cfg, err := config.Load(l.ConfigFile(), settingsPath(l))
			if err != nil {
				return err
			}
			if len(cfg.Quotas) == 0 {
				fmt.Fprintln(cmd.OutOrStdout(), "no quotas set (the default)")
			}
			for n, q := range cfg.Quotas {
				fmt.Fprintf(cmd.OutOrStdout(), "%s: %s\n", n, describeQuota(q))
			}
			return nil
		},
	}
	cmd.AddCommand(set, clear, show)
	return cmd
}

func describeQuota(q config.Quota) string {
	if q.IsZero() {
		return "none"
	}
	var p []string
	if q.DeviceMemoryBytes > 0 {
		p = append(p, "device memory "+q.DeviceMemoryBytes.String())
	}
	if q.MaxSessions > 0 {
		p = append(p, fmt.Sprintf("%d sessions", q.MaxSessions))
	}
	if q.ExecuteWallMS > 0 {
		p = append(p, fmt.Sprintf("%d ms per execute", q.ExecuteWallMS))
	}
	if q.ExecuteFLOPs > 0 {
		p = append(p, fmt.Sprintf("%d FLOPs per execute", q.ExecuteFLOPs))
	}
	if q.StorageBytes > 0 {
		p = append(p, "storage "+q.StorageBytes.String())
	}
	if q.UploadBytesPerSec > 0 {
		p = append(p, "upload "+q.UploadBytesPerSec.String()+"/s")
	}
	return strings.Join(p, ", ")
}

func railsCmd() *cobra.Command {
	cmd := &cobra.Command{Use: "rails", Short: "Safety rails that keep the machine alive (always on)"}
	show := &cobra.Command{
		Use:   "show",
		Short: "Show the active safety rails",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			l, err := layout()
			if err != nil {
				return err
			}
			cfg, err := config.Load(l.ConfigFile(), settingsPath(l))
			if err != nil {
				return err
			}
			r := cfg.Rails
			out := cmd.OutOrStdout()
			fmt.Fprintf(out, "disk free floor:  max(%d%%, %s) of the store filesystem\n", r.DiskFreeFloorPercent, r.DiskFreeFloorBytes)
			fmt.Fprintf(out, "host RAM floor:   %s stays available for you\n", r.HostRAMFloorBytes)
			fmt.Fprintf(out, "worker pids:      %d\n", r.MaxPIDs)
			l2 := cfg.Limits
			fmt.Fprintf(out, "protocol limits:  frame %s, %d nodes, %d dims, %d elements per tensor\n",
				l2.MaxFrameBytes, l2.MaxNodes, l2.MaxDims, l2.MaxTensorNumel)
			return nil
		},
	}
	var pct uint32
	var disk, ram string
	var pids uint32
	set := &cobra.Command{
		Use:   "set",
		Short: "Change a safety rail",
		Long: `Change safety rails. Defaults: disk max(10%, 20GiB), host RAM 4GiB.

  reflopd rails set --disk-percent 15 --disk 50GiB
  reflopd rails set --ram 8GiB`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			l, err := layout()
			if err != nil {
				return err
			}
			s, err := config.LoadSettings(settingsPath(l))
			if err != nil {
				return err
			}
			r := config.DefaultRails()
			if s.Rails != nil {
				r = *s.Rails
			}
			f := cmd.Flags()
			if f.Changed("disk-percent") {
				r.DiskFreeFloorPercent = pct
			}
			if f.Changed("disk") {
				b, err := config.ParseBytes(disk)
				if err != nil {
					return err
				}
				r.DiskFreeFloorBytes = b
			}
			if f.Changed("ram") {
				b, err := config.ParseBytes(ram)
				if err != nil {
					return err
				}
				r.HostRAMFloorBytes = b
			}
			if f.Changed("pids") {
				r.MaxPIDs = pids
			}
			c := config.Defaults()
			c.Rails = r
			if err := c.Validate(); err != nil {
				return err
			}
			s.Rails = &r
			if err := config.SaveSettings(settingsPath(l), s); err != nil {
				return err
			}
			notifyReload()
			fmt.Fprintln(cmd.OutOrStdout(), "rails updated")
			return nil
		},
	}
	set.Flags().Uint32Var(&pct, "disk-percent", 10, "keep this percent of the store filesystem free")
	set.Flags().StringVar(&disk, "disk", "20GiB", "keep at least this much disk free")
	set.Flags().StringVar(&ram, "ram", "4GiB", "keep at least this much host RAM available")
	set.Flags().Uint32Var(&pids, "pids", 4096, "max processes/threads per worker")
	cmd.AddCommand(show, set)
	return cmd
}

func pauseCmd() *cobra.Command {
	var now bool
	cmd := &cobra.Command{
		Use:   "pause",
		Short: "Queue new Guest work (with --now, also kill running workers)",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			if err := control("pause", map[string]bool{"now": now}, nil); err != nil {
				return err
			}
			fmt.Fprintln(cmd.OutOrStdout(), "paused")
			return nil
		},
	}
	cmd.Flags().BoolVar(&now, "now", false, "kill running workers too (their sessions are lost)")
	return cmd
}

func resumeCmd() *cobra.Command {
	return &cobra.Command{Use: "resume", Short: "Resume Guest work", Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			if err := control("resume", nil, nil); err != nil {
				return err
			}
			fmt.Fprintln(cmd.OutOrStdout(), "resumed")
			return nil
		}}
}

func killCmd() *cobra.Command {
	return &cobra.Command{Use: "kill <identity|session-id>", Short: "Kill an identity's sessions or one session",
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			var r map[string]int
			if err := control("kill", map[string]string{"target": args[0]}, &r); err != nil {
				return err
			}
			fmt.Fprintf(cmd.OutOrStdout(), "killed %d session(s)\n", r["killed"])
			return nil
		}}
}

func yieldCmd() *cobra.Command {
	var auto, off bool
	cmd := &cobra.Command{
		Use:   "yield --auto | --off",
		Short: "Pause Guests automatically while you use the GPU",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			if auto == off {
				return exitWith(ExitUsage, errors.New("pass --auto or --off"))
			}
			l, err := layout()
			if err != nil {
				return err
			}
			s, err := config.LoadSettings(settingsPath(l))
			if err != nil {
				return err
			}
			v := auto
			s.Yield = &v
			if err := config.SaveSettings(settingsPath(l), s); err != nil {
				return err
			}
			_ = control("yield", map[string]bool{"auto": auto}, nil)
			fmt.Fprintf(cmd.OutOrStdout(), "auto-yield %s\n", map[bool]string{true: "on", false: "off"}[auto])
			return nil
		},
	}
	cmd.Flags().BoolVar(&auto, "auto", false, "turn auto-yield on")
	cmd.Flags().BoolVar(&off, "off", false, "turn auto-yield off")
	return cmd
}

func sayCmd() *cobra.Command {
	return &cobra.Command{Use: "say <message>", Short: "Send a message to connected Guests", Args: cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return control("say", map[string]string{"text": strings.Join(args, " ")}, nil)
		}}
}

func statusCmd() *cobra.Command {
	var asJSON bool
	cmd := &cobra.Command{
		Use:   "status",
		Short: "Show whether reflopd runs, where it listens, and its sessions",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			var st server.StatusView
			if err := control("status", nil, &st); err != nil {
				return err
			}
			return printStatus(cmd.OutOrStdout(), st, false)
		},
	}
	cmd.Flags().BoolVar(&asJSON, "json", false, "JSON output")
	return cmd
}

func topCmd() *cobra.Command {
	var watch bool
	cmd := &cobra.Command{
		Use:   "top",
		Short: "Live view of sessions (memory, executes, FLOPs)",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			for {
				var st server.StatusView
				if err := control("top", nil, &st); err != nil {
					return err
				}
				if watch {
					fmt.Fprint(cmd.OutOrStdout(), "\x1b[H\x1b[2J")
				}
				printStatus(cmd.OutOrStdout(), st, true)
				if !watch {
					return nil
				}
				time.Sleep(2 * time.Second)
			}
		},
	}
	cmd.Flags().BoolVarP(&watch, "watch", "w", false, "refresh every 2 seconds")
	return cmd
}

func printStatus(out io.Writer, st server.StatusView, detail bool) error {
	state := "serving"
	if st.Paused {
		state = "paused"
	}
	if st.Yielding {
		state = "yielding to the Owner"
	}
	fmt.Fprintf(out, "reflopd %s, up %s, %s (level %d)\n", st.Version, st.Uptime, state, st.Level)
	fmt.Fprintf(out, "listen:   %s\n", strings.Join(st.Listen, ", "))
	fmt.Fprintf(out, "key:      %s\n", st.Fingerprint[:16])
	fmt.Fprintf(out, "runtime:  %s\n", st.Runtime)
	fmt.Fprintf(out, "devices:  %s\n", strings.Join(st.Devices, "; "))
	fmt.Fprintf(out, "sandbox:  %s\n", strings.Join(st.Sandbox, ", "))
	fmt.Fprintf(out, "%d connection(s), %d session(s)\n", st.Connections, len(st.Sessions))
	if len(st.Sessions) == 0 {
		return nil
	}
	tw := tabwriter.NewWriter(out, 0, 4, 2, ' ', 0)
	if detail {
		fmt.Fprintln(tw, "SESSION\tIDENTITY\tDEVICE\tAGE\tEXECUTES\tTFLOP\tHOST MEM\tDEVICE MEM\tHANDLES\tGRAPHS")
	} else {
		fmt.Fprintln(tw, "SESSION\tIDENTITY\tDEVICE\tAGE\tEXECUTES\tATTACHED")
	}
	for _, s := range st.Sessions {
		if detail {
			fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%d\t%.2f\t%s\t%s\t%d\t%d\n", s.ID[:8], s.Identity, s.Device, s.Age, s.Executes,
				float64(s.FLOPs)/1e12, humanBytes(s.HostMemory), humanBytes(s.DeviceAlloc), s.Handles, s.Graphs)
		} else {
			fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%d\t%v\n", s.ID[:8], s.Identity, s.Device, s.Age, s.Executes, s.Attached)
		}
	}
	return tw.Flush()
}

func auditCmd() *cobra.Command {
	var name string
	var since time.Duration
	var limit int
	cmd := &cobra.Command{
		Use:   "audit",
		Short: "Show the audit log (never tensor contents)",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			l, err := layout()
			if err != nil {
				return err
			}
			recs, err := audit.Query(l.Audit(), name, time.Now().Add(-since), limit)
			if err != nil {
				return err
			}
			for _, r := range recs {
				fields := ""
				if len(r.Fields) > 0 {
					keys := make([]string, 0, len(r.Fields))
					for k := range r.Fields {
						keys = append(keys, k)
					}
					sort.Strings(keys)
					var parts []string
					for _, k := range keys {
						parts = append(parts, fmt.Sprintf("%s=%v", k, r.Fields[k]))
					}
					fields = " " + strings.Join(parts, " ")
				}
				sess := ""
				if r.Session != "" {
					sess = " " + r.Session[:8]
				}
				fmt.Fprintf(cmd.OutOrStdout(), "%s %-10s%s %s%s\n", r.Time.Local().Format("01-02 15:04:05"), r.Identity, sess, r.Event, fields)
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&name, "name", "", "only this identity")
	cmd.Flags().DurationVar(&since, "since", 24*time.Hour, "how far back")
	cmd.Flags().IntVar(&limit, "limit", 200, "max records")
	return cmd
}

func selftestCmd() *cobra.Command {
	var security, weak bool
	cmd := &cobra.Command{
		Use:   "selftest",
		Short: "Run a sandboxed worker on every device and check it",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			l, err := layout()
			if err != nil {
				return err
			}
			return runSelftest(context.Background(), cmd.OutOrStdout(), l, security, weak)
		},
	}
	cmd.Flags().BoolVar(&security, "security", false, "also verify isolation (no_new_privs, seccomp, environment, sockets)")
	cmd.Flags().BoolVar(&weak, "accept-weak-isolation", false, "allow running without seccomp/Landlock")
	return cmd
}

func runSelftest(ctx context.Context, out io.Writer, l paths.Layout, security, weak bool) error {
	srv, err := server.New(server.Options{Layout: l, AcceptWeakIsolation: weak, SkipProbe: true})
	if err != nil {
		return err
	}
	if err := srv.Prepare(ctx); err != nil {
		return err
	}
	return srv.SelfTest(ctx, out, security)
}
