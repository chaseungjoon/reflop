// SPDX-License-Identifier: Apache-2.0

// Package buildinfo carries version metadata injected at build time with
// -ldflags "-X github.com/chaseungjoon/reflop/server/reflopd/internal/buildinfo.Version=...".
package buildinfo

import (
	"runtime"
	"runtime/debug"
)

// Set via -ldflags at release time.
var (
	Version = "dev"
	Commit  = ""
	Date    = ""
)

func init() {
	if Commit != "" {
		return
	}
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return
	}
	if Version == "dev" && info.Main.Version != "" && info.Main.Version != "(devel)" {
		Version = info.Main.Version
	}
	dirty := false
	for _, s := range info.Settings {
		switch s.Key {
		case "vcs.revision":
			Commit = s.Value
		case "vcs.time":
			Date = s.Value
		case "vcs.modified":
			dirty = s.Value == "true"
		}
	}
	if dirty && Commit != "" {
		Commit += "-dirty"
	}
}

// ShortCommit returns the first 12 characters of the commit.
func ShortCommit() string {
	if len(Commit) > 12 {
		return Commit[:12]
	}
	return Commit
}

// Platform returns GOOS/GOARCH.
func Platform() string { return runtime.GOOS + "/" + runtime.GOARCH }
