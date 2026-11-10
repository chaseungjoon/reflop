// SPDX-License-Identifier: Apache-2.0

// Package paths decides every server-side location. Guests never choose a
// path (invariant I7): everything below is derived from the Owner's
// environment (Level 0) or fixed system locations (Level 1).
package paths

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
)

// Layout holds reflopd's directories.
type Layout struct {
	Config  string // config.toml, server key
	State   string // identities, invites, audit log, datasets, snapshots
	Runtime string // pinned compute runtimes
	Run     string // control socket, pid file (XDG_RUNTIME_DIR)
	Scratch string // per-session scratch dirs
	System  bool   // Level 1 layout
}

// SystemRoot is the Level 1 state root.
const SystemRoot = "/var/lib/reflop"

// Resolve returns the layout for this process. REFLOP_HOME (if set) puts
// everything under one directory; it exists for tests and for Owners who
// want a self-contained install.
func Resolve() (Layout, error) {
	if h := os.Getenv("REFLOP_HOME"); h != "" {
		h, err := filepath.Abs(h)
		if err != nil {
			return Layout{}, err
		}
		return Layout{
			Config:  filepath.Join(h, "config"),
			State:   filepath.Join(h, "state"),
			Runtime: filepath.Join(h, "runtime"),
			Run:     filepath.Join(h, "run"),
			Scratch: filepath.Join(h, "scratch"),
		}, nil
	}
	if os.Getenv("REFLOP_SYSTEM") == "1" {
		return Layout{
			Config:  "/etc/reflop",
			State:   SystemRoot,
			Runtime: filepath.Join(SystemRoot, "runtime"),
			Run:     "/run/reflop",
			Scratch: filepath.Join(SystemRoot, "scratch"),
			System:  true,
		}, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return Layout{}, fmt.Errorf("cannot determine home directory: %w", err)
	}
	cfg := xdg("XDG_CONFIG_HOME", filepath.Join(home, ".config"))
	data := xdg("XDG_DATA_HOME", filepath.Join(home, ".local", "share"))
	state := xdg("XDG_STATE_HOME", filepath.Join(home, ".local", "state"))
	cache := xdg("XDG_CACHE_HOME", filepath.Join(home, ".cache"))
	run := os.Getenv("XDG_RUNTIME_DIR")
	if run == "" || !filepath.IsAbs(run) {
		run = filepath.Join(state, "reflop", "run")
	} else {
		run = filepath.Join(run, "reflop")
	}
	_ = state
	return Layout{
		Config:  filepath.Join(cfg, "reflop"),
		State:   filepath.Join(data, "reflop"),
		Runtime: filepath.Join(data, "reflop", "runtime"),
		Run:     run,
		Scratch: filepath.Join(cache, "reflop", "scratch"),
	}, nil
}

func xdg(env, def string) string {
	if v := os.Getenv(env); v != "" && filepath.IsAbs(v) {
		return v
	}
	return def
}

// Ensure creates all directories with owner-only permissions.
func (l Layout) Ensure() error {
	for _, d := range []string{l.Config, l.State, l.Runtime, l.Run, l.Scratch, l.Identities(), l.Store(), l.Snapshots(), l.Audit()} {
		if err := os.MkdirAll(d, 0o700); err != nil {
			return err
		}
		if err := os.Chmod(d, 0o700); err != nil && !errors.Is(err, os.ErrPermission) {
			return err
		}
	}
	return nil
}

func (l Layout) Identities() string    { return filepath.Join(l.State, "identities") }
func (l Layout) Store() string         { return filepath.Join(l.State, "store") }
func (l Layout) Snapshots() string     { return filepath.Join(l.State, "snapshots") }
func (l Layout) Audit() string         { return filepath.Join(l.State, "audit") }
func (l Layout) ConfigFile() string    { return filepath.Join(l.Config, "reflopd.toml") }
func (l Layout) ServerKey() string     { return filepath.Join(l.Config, "server.key") }
func (l Layout) ServerCert() string    { return filepath.Join(l.Config, "server.crt") }
func (l Layout) ControlSocket() string { return filepath.Join(l.Run, "control.sock") }
func (l Layout) PIDFile() string       { return filepath.Join(l.Run, "reflopd.pid") }
func (l Layout) LogFile() string       { return filepath.Join(l.State, "reflopd.log") }

// IdentityStore is the content-addressed store of one identity.
func (l Layout) IdentityStore(identityHex string) string {
	return filepath.Join(l.Store(), identityHex)
}

// IdentitySnapshots holds one identity's snapshots.
func (l Layout) IdentitySnapshots(identityHex string) string {
	return filepath.Join(l.Snapshots(), identityHex)
}

// SessionScratch is the only directory a worker may write.
func (l Layout) SessionScratch(sessionHex string) string { return filepath.Join(l.Scratch, sessionHex) }

// Itoa is a small helper used for /proc paths.
func Itoa(i int) string { return strconv.Itoa(i) }
