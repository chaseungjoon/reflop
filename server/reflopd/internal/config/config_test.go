// SPDX-License-Identifier: Apache-2.0

package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseBytes(t *testing.T) {
	for in, want := range map[string]Bytes{
		"4GiB": 4 << 30, "4gib": 4 << 30, "512 MiB": 512 << 20, "20GB": 20e9, "1000000": 1e6, "4G": 4 << 30, "1.5KiB": 1536,
	} {
		got, err := ParseBytes(in)
		if err != nil || got != want {
			t.Errorf("ParseBytes(%q) = %d, %v; want %d", in, got, err, want)
		}
	}
	for _, bad := range []string{"", "GiB", "-1", "4 XB", "1e3"} {
		if _, err := ParseBytes(bad); err == nil {
			t.Errorf("ParseBytes(%q) accepted", bad)
		}
	}
}

func TestDefaultsValidate(t *testing.T) {
	if err := Defaults().Validate(); err != nil {
		t.Fatal(err)
	}
}

// SEC_SWEEP L5: no value turns a rail or limit off.
func TestZeroNeverDisablesProtection(t *testing.T) {
	mut := map[string]func(*Config){
		"disk percent 0":    func(c *Config) { c.Rails.DiskFreeFloorPercent = 0 },
		"disk bytes 0":      func(c *Config) { c.Rails.DiskFreeFloorBytes = 0 },
		"ram 0":             func(c *Config) { c.Rails.HostRAMFloorBytes = 0 },
		"ram below minimum": func(c *Config) { c.Rails.HostRAMFloorBytes = MinHostRAMFloorBytes - 1 },
		"pids 0":            func(c *Config) { c.Rails.MaxPIDs = 0 },
		"max_nodes 0":       func(c *Config) { c.Limits.MaxNodes = 0 },
		"max_connections 0": func(c *Config) { c.Limits.MaxConnections = 0 },
		"max_fetch 0":       func(c *Config) { c.Limits.MaxFetchBytes = 0 },
		"max_handles 0":     func(c *Config) { c.Limits.MaxHandlesPerSession = 0 },
		"max_graphs 0":      func(c *Config) { c.Limits.MaxGraphsPerSession = 0 },
		"manifest 0":        func(c *Config) { c.Limits.MaxManifestBytes = 0 },
		"numel 0":           func(c *Config) { c.Limits.MaxTensorNumel = 0 },
		"list 0":            func(c *Config) { c.Limits.MaxListLength = 0 },
		"inputs 0":          func(c *Config) { c.Limits.MaxGraphInputs = 0 },
		"inline 0":          func(c *Config) { c.Limits.MaxInlineTensorBytes = 0 },
		"constant 0":        func(c *Config) { c.Limits.MaxConstantBytes = 0 },
		"retention 0":       func(c *Config) { c.Audit.RetentionDays = 0 },
		"retention < 0":     func(c *Config) { c.Audit.RetentionDays = -1 },
	}
	for name, f := range mut {
		c := Defaults()
		f(&c)
		if err := c.Validate(); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	r := DefaultRails()
	r.HostRAMFloorBytes = MinHostRAMFloorBytes
	if len(r.BelowDefaults()) != 1 {
		t.Errorf("BelowDefaults = %v", r.BelowDefaults())
	}
}

// SEC_SWEEP L5: `device_memory = 0` in reflopd.toml would have meant
// "unlimited"; it is rejected, and omitting the key still means no limit.
func TestQuotaZeroInTOMLRejected(t *testing.T) {
	dir := t.TempDir()
	cfg := filepath.Join(dir, "reflopd.toml")
	os.WriteFile(cfg, []byte("[quotas.alice]\ndevice_memory = \"0\"\nmax_sessions = 1\n"), 0o600)
	if _, err := Load(cfg, filepath.Join(dir, "settings.json")); err == nil || !strings.Contains(err.Error(), "device_memory") {
		t.Fatalf("got %v", err)
	}
	os.WriteFile(cfg, []byte("[quotas.alice]\ndevice_memory = \"6GiB\"\n"), 0o600)
	c, err := Load(cfg, filepath.Join(dir, "settings.json"))
	if err != nil || c.Quotas["alice"].DeviceMemoryBytes != 6<<30 || c.Quotas["alice"].MaxSessions != 0 {
		t.Fatalf("got %+v, %v", c.Quotas, err)
	}
}
