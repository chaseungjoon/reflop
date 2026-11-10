// SPDX-License-Identifier: Apache-2.0
// Adapted from recom (same sole author; relicensed under Apache-2.0, ADR-007).

// Package selfupdate replaces the running reflopd binary with a published
// GitHub release, verifying the archive against the release's checksums.txt.
package selfupdate

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// DefaultBaseURL is where releases are published.
const DefaultBaseURL = "https://github.com/chaseungjoon/reflop/releases"

// InstallScript is the documented installer, for error messages.
const InstallScript = "curl -fsSL https://raw.githubusercontent.com/chaseungjoon/reflop/main/install.sh | sh -s -- --server"

const maxDownload = 256 << 20

// Updater talks to a release host.
type Updater struct {
	BaseURL string // e.g. DefaultBaseURL; REFLOP_UPDATE_URL overrides it
	OS      string
	Arch    string
	Client  *http.Client
}

// New returns an updater for this platform.
func New() *Updater {
	base := DefaultBaseURL
	if v := os.Getenv("REFLOP_UPDATE_URL"); v != "" {
		base = strings.TrimRight(v, "/")
	}
	return &Updater{BaseURL: base, OS: runtime.GOOS, Arch: runtime.GOARCH, Client: &http.Client{Timeout: 5 * time.Minute}}
}

var tagRe = regexp.MustCompile(`^v\d+\.\d+\.\d+([-+][0-9A-Za-z.-]+)?$`)

// Latest returns the tag of the newest (non-prerelease) release. It follows
// the /releases/latest redirect rather than the rate-limited GitHub API.
func (u *Updater) Latest(ctx context.Context) (string, error) {
	c := *u.Client
	c.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, u.BaseURL+"/latest", nil)
	if err != nil {
		return "", err
	}
	resp, err := c.Do(req)
	if err != nil {
		return "", fmt.Errorf("could not reach %s: %w", u.BaseURL, err)
	}
	resp.Body.Close()
	loc := resp.Header.Get("Location")
	if loc == "" {
		return "", fmt.Errorf("could not determine the latest release from %s/latest (HTTP %d)", u.BaseURL, resp.StatusCode)
	}
	tag := loc[strings.LastIndex(loc, "/")+1:]
	if !tagRe.MatchString(tag) {
		return "", fmt.Errorf("could not determine the latest release (redirected to %q)", loc)
	}
	return tag, nil
}

// Archive is the release archive name for a tag on this platform.
func (u *Updater) Archive(tag string) string {
	return fmt.Sprintf("reflopd_%s_%s_%s.tar.gz", strings.TrimPrefix(tag, "v"), u.OS, u.Arch)
}

func (u *Updater) get(ctx context.Context, url string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := u.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("download %s: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("download %s: HTTP %d", url, resp.StatusCode)
	}
	b, err := io.ReadAll(io.LimitReader(resp.Body, maxDownload+1))
	if err != nil {
		return nil, fmt.Errorf("download %s: %w", url, err)
	}
	if len(b) > maxDownload {
		return nil, fmt.Errorf("download %s: larger than %d bytes", url, maxDownload)
	}
	return b, nil
}

// Fetch downloads tag's archive for this platform, verifies its SHA-256
// against checksums.txt, and returns the reflopd binary inside it.
func (u *Updater) Fetch(ctx context.Context, tag string) ([]byte, error) {
	name := u.Archive(tag)
	base := u.BaseURL + "/download/" + tag + "/"
	sums, err := u.get(ctx, base+"checksums.txt")
	if err != nil {
		return nil, err
	}
	want := ""
	for _, line := range strings.Split(string(sums), "\n") {
		f := strings.Fields(line)
		if len(f) == 2 && strings.TrimPrefix(f[1], "*") == name {
			want = strings.ToLower(f[0])
		}
	}
	if want == "" {
		return nil, fmt.Errorf("release %s has no %s (unsupported platform %s/%s?)", tag, name, u.OS, u.Arch)
	}
	archive, err := u.get(ctx, base+name)
	if err != nil {
		return nil, err
	}
	sum := sha256.Sum256(archive)
	if got := hex.EncodeToString(sum[:]); got != want {
		return nil, fmt.Errorf("checksum mismatch for %s: expected %s, got %s", name, want, got)
	}
	return extract(archive)
}

func extract(archive []byte) ([]byte, error) {
	zr, err := gzip.NewReader(bytes.NewReader(archive))
	if err != nil {
		return nil, fmt.Errorf("archive: %w", err)
	}
	tr := tar.NewReader(zr)
	for {
		h, err := tr.Next()
		if errors.Is(err, io.EOF) {
			return nil, errors.New("archive does not contain the reflopd binary")
		}
		if err != nil {
			return nil, fmt.Errorf("archive: %w", err)
		}
		if h.Typeflag == tar.TypeReg && filepath.Base(h.Name) == "reflopd" {
			return io.ReadAll(io.LimitReader(tr, maxDownload))
		}
	}
}

// Executable returns the real path of the running binary.
func Executable() (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	return filepath.EvalSymlinks(exe)
}

// ManagedBy names a package manager that owns path, or "".
func ManagedBy(path string) string {
	if strings.Contains(path, "/Cellar/") || strings.Contains(path, "/homebrew/") || strings.Contains(path, "/linuxbrew/") {
		return "Homebrew"
	}
	return ""
}

// ErrPermission means the binary's directory is not writable.
var ErrPermission = errors.New("permission denied")

// Install atomically replaces exe with bin after checking that bin runs
// and reports wantVersion.
func Install(exe string, bin []byte, wantVersion string) error {
	dir := filepath.Dir(exe)
	f, err := os.CreateTemp(dir, ".reflopd-update-*")
	if err != nil {
		if errors.Is(err, os.ErrPermission) {
			return fmt.Errorf("%w: cannot write to %s", ErrPermission, dir)
		}
		return err
	}
	tmp := f.Name()
	defer os.Remove(tmp)
	if _, err := f.Write(bin); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Chmod(tmp, 0o755); err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, tmp, "version").CombinedOutput()
	if err != nil {
		return fmt.Errorf("the downloaded binary does not run on this machine: %v (%s)", err, strings.TrimSpace(string(out)))
	}
	if v := strings.TrimPrefix(wantVersion, "v"); !strings.Contains(string(out), v) {
		return fmt.Errorf("the downloaded binary reports %q, expected version %s", firstLine(string(out)), v)
	}
	if err := os.Rename(tmp, exe); err != nil {
		if errors.Is(err, os.ErrPermission) {
			return fmt.Errorf("%w: cannot replace %s", ErrPermission, exe)
		}
		return err
	}
	return nil
}

func firstLine(s string) string {
	s, _, _ = strings.Cut(strings.TrimSpace(s), "\n")
	return s
}

// Version is a parsed release version (pre-release suffixes ignored).
type Version struct{ Major, Minor, Patch int }

var versionRe = regexp.MustCompile(`^v?(\d+)\.(\d+)\.(\d+)`)

// Parse reads "v0.2.0", "0.2.0", "v0.2.0-3-gabc-dirty", ...
func Parse(s string) (Version, bool) {
	m := versionRe.FindStringSubmatch(strings.TrimSpace(s))
	if m == nil {
		return Version{}, false
	}
	a, _ := strconv.Atoi(m[1])
	b, _ := strconv.Atoi(m[2])
	c, _ := strconv.Atoi(m[3])
	return Version{a, b, c}, true
}

// Less reports whether v is older than w.
func (v Version) Less(w Version) bool {
	if v.Major != w.Major {
		return v.Major < w.Major
	}
	if v.Minor != w.Minor {
		return v.Minor < w.Minor
	}
	return v.Patch < w.Patch
}

func (v Version) String() string { return fmt.Sprintf("%d.%d.%d", v.Major, v.Minor, v.Patch) }
