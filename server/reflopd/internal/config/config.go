// SPDX-License-Identifier: Apache-2.0

// Package config holds reflopd settings. No file is required: Defaults()
// is a complete, safe configuration. An optional reflopd.toml overrides
// values; `reflopd rails set` and `reflopd quota set` persist their changes
// in the state directory (settings.json), which overrides both.
package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/BurntSushi/toml"
)

// DefaultPort is reflopd's TCP port.
const DefaultPort = 7460

// Bytes is a byte count that accepts "20GiB"-style strings in TOML/JSON.
type Bytes uint64

var bytesRe = regexp.MustCompile(`^\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGTP]i?B?|B)?\s*$`)

// ParseBytes parses "4GiB", "512 MiB", "1000000", "2TB".
func ParseBytes(s string) (Bytes, error) {
	m := bytesRe.FindStringSubmatch(strings.ToUpper(strings.TrimSpace(s)))
	if m == nil {
		return 0, fmt.Errorf("invalid size %q (examples: 4GiB, 512MiB, 1000000)", s)
	}
	f, err := strconv.ParseFloat(m[1], 64)
	if err != nil {
		return 0, err
	}
	unit := strings.TrimSuffix(strings.TrimSuffix(m[2], "B"), "I")
	mult := map[string]float64{"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40, "P": 1 << 50}[unit]
	if strings.Contains(m[2], "B") && !strings.Contains(m[2], "I") && unit != "" {
		mult = map[string]float64{"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15}[unit]
	}
	v := f * mult
	if v < 0 || v > 1<<63 {
		return 0, fmt.Errorf("size %q out of range", s)
	}
	return Bytes(v), nil
}

func (b Bytes) String() string {
	units := []string{"B", "KiB", "MiB", "GiB", "TiB", "PiB"}
	v := float64(b)
	i := 0
	for v >= 1024 && i < len(units)-1 {
		v /= 1024
		i++
	}
	if i == 0 {
		return fmt.Sprintf("%d B", uint64(b))
	}
	s := strconv.FormatFloat(v, 'f', 2, 64)
	s = strings.TrimRight(strings.TrimRight(s, "0"), ".")
	return s + " " + units[i]
}

func (b *Bytes) UnmarshalText(t []byte) error {
	v, err := ParseBytes(string(t))
	*b = v
	return err
}

func (b Bytes) MarshalText() ([]byte, error) { return []byte(strconv.FormatUint(uint64(b), 10)), nil }

// Duration accepts Go duration strings.
type Duration time.Duration

func (d *Duration) UnmarshalText(t []byte) error {
	v, err := time.ParseDuration(string(t))
	*d = Duration(v)
	return err
}

func (d Duration) MarshalText() ([]byte, error) { return []byte(time.Duration(d).String()), nil }

// Rails are always-on limits that keep the machine alive (PLAN.md §9.6).
type Rails struct {
	DiskFreeFloorPercent uint32 `toml:"disk_free_floor_percent" json:"disk_free_floor_percent"`
	DiskFreeFloorBytes   Bytes  `toml:"disk_free_floor_bytes" json:"disk_free_floor_bytes"`
	HostRAMFloorBytes    Bytes  `toml:"host_ram_floor_bytes" json:"host_ram_floor_bytes"`
	MaxPIDs              uint32 `toml:"max_pids" json:"max_pids"`
}

// Limits are protocol sanity limits (robustness, not usage limits).
type Limits struct {
	MaxFrameBytes        Bytes  `toml:"max_frame_bytes" json:"max_frame_bytes"`
	MaxNodes             uint32 `toml:"max_nodes" json:"max_nodes"`
	MaxGraphInputs       uint32 `toml:"max_graph_inputs" json:"max_graph_inputs"`
	MaxDims              uint32 `toml:"max_dims" json:"max_dims"`
	MaxTensorNumel       uint64 `toml:"max_tensor_numel" json:"max_tensor_numel"`
	MaxInlineTensorBytes Bytes  `toml:"max_inline_tensor_bytes" json:"max_inline_tensor_bytes"`
	MaxConstantBytes     Bytes  `toml:"max_constant_bytes" json:"max_constant_bytes"`
	MaxFetchBytes        Bytes  `toml:"max_fetch_bytes" json:"max_fetch_bytes"`
	MaxListLength        uint32 `toml:"max_list_length" json:"max_list_length"`
	UploadChunkBytes     Bytes  `toml:"upload_chunk_bytes" json:"upload_chunk_bytes"`
	MaxManifestBytes     Bytes  `toml:"max_manifest_bytes" json:"max_manifest_bytes"`
	MaxGraphsPerSession  uint32 `toml:"max_graphs_per_session" json:"max_graphs_per_session"`
	MaxHandlesPerSession uint32 `toml:"max_handles_per_session" json:"max_handles_per_session"`
	MaxConnections       uint32 `toml:"max_connections" json:"max_connections"`
}

// Quota is an optional per-identity fairness limit. Zero means unset.
type Quota struct {
	DeviceMemoryBytes Bytes  `toml:"device_memory" json:"device_memory,omitempty"`
	MaxSessions       uint32 `toml:"max_sessions" json:"max_sessions,omitempty"`
	ExecuteWallMS     uint64 `toml:"execute_wall_ms" json:"execute_wall_ms,omitempty"`
	ExecuteFLOPs      uint64 `toml:"execute_flops" json:"execute_flops,omitempty"`
	StorageBytes      Bytes  `toml:"storage" json:"storage,omitempty"`
	UploadBytesPerSec Bytes  `toml:"upload_bandwidth" json:"upload_bandwidth,omitempty"`
}

// IsZero reports whether no quota is set.
func (q Quota) IsZero() bool { return q == Quota{} }

// Numerics are server numeric defaults; clients may request changes within
// this allowlist.
type Numerics struct {
	TF32Matmul     bool `toml:"tf32_matmul" json:"tf32_matmul"`
	TF32CuDNN      bool `toml:"tf32_cudnn" json:"tf32_cudnn"`
	Deterministic  bool `toml:"deterministic" json:"deterministic"`
	CuDNNBenchmark bool `toml:"cudnn_benchmark" json:"cudnn_benchmark"`
}

// Config is the complete reflopd configuration.
type Config struct {
	Listen    []string `toml:"listen"`    // host:port; default 127.0.0.1:7460
	Advertise []string `toml:"advertise"` // addresses written into invites
	Rails     Rails    `toml:"rails"`
	Limits    Limits   `toml:"limits"`
	Numerics  Numerics `toml:"numerics"`
	Session   struct {
		TTL Duration `toml:"ttl"`
	} `toml:"session"`
	Audit struct {
		RetentionDays int `toml:"retention_days"`
	} `toml:"audit"`
	Sandbox struct {
		AcceptWeakIsolation bool `toml:"accept_weak_isolation"`
		// TryUserNamespaces adds user+network namespaces when the kernel
		// allows unprivileged ones (not on Ubuntu 24.04 defaults).
		TryUserNamespaces bool `toml:"try_user_namespaces"`
	} `toml:"sandbox"`
	Devices struct {
		CUDA []int `toml:"cuda"` // physical ordinals to expose; empty = all
		CPU  bool  `toml:"cpu"`
	} `toml:"devices"`
	Ops struct {
		Disabled     []string        `toml:"disabled"`
		Capabilities map[string]bool `toml:"capabilities"` // Tier 2 toggles
	} `toml:"ops"`
	Yield struct {
		Auto     bool     `toml:"auto"`
		Interval Duration `toml:"interval"`
	} `toml:"yield"`
	Quotas map[string]Quota `toml:"quotas"`
}

// DefaultRails are the user-confirmed defaults: keep max(10%, 20 GiB) of
// the store filesystem free and 4 GiB of host RAM free.
func DefaultRails() Rails {
	return Rails{
		DiskFreeFloorPercent: 10,
		DiskFreeFloorBytes:   20 << 30,
		HostRAMFloorBytes:    4 << 30,
		MaxPIDs:              4096,
	}
}

// Defaults returns the built-in configuration.
func Defaults() Config {
	var c Config
	c.Listen = []string{fmt.Sprintf("127.0.0.1:%d", DefaultPort)}
	c.Rails = DefaultRails()
	c.Limits = Limits{
		MaxFrameBytes:        64 << 20,
		MaxNodes:             1_000_000,
		MaxGraphInputs:       65_536,
		MaxDims:              16,
		MaxTensorNumel:       1 << 40,
		MaxInlineTensorBytes: 1 << 20,
		MaxConstantBytes:     64 << 10,
		MaxFetchBytes:        16 << 30,
		MaxListLength:        65_536,
		UploadChunkBytes:     4 << 20,
		MaxManifestBytes:     1 << 20,
		MaxGraphsPerSession:  65_536,
		MaxHandlesPerSession: 1 << 22,
		MaxConnections:       256,
	}
	c.Numerics = Numerics{TF32Matmul: false, TF32CuDNN: true}
	c.Session.TTL = Duration(30 * time.Minute)
	c.Audit.RetentionDays = 30
	c.Sandbox.TryUserNamespaces = true
	c.Devices.CPU = true
	c.Ops.Capabilities = map[string]bool{
		"linalg": true, "complex": true, "fp8": true, "ctc": true, "cdist_backward": true,
		"rnn_fused": false, // O11: off until the _cudnn_rnn validator has M4 tests
	}
	c.Yield.Interval = Duration(2 * time.Second)
	c.Quotas = map[string]Quota{}
	return c
}

// Settings are values changed at runtime by Owner commands.
type Settings struct {
	Rails  *Rails           `json:"rails,omitempty"`
	Quotas map[string]Quota `json:"quotas,omitempty"`
	Yield  *bool            `json:"yield_auto,omitempty"`
}

// Load reads the optional config file and settings file over Defaults().
func Load(configFile, settingsFile string) (Config, error) {
	c := Defaults()
	if b, err := os.ReadFile(configFile); err == nil {
		md, err := toml.Decode(string(b), &c)
		if err != nil {
			return c, fmt.Errorf("%s: %w", configFile, err)
		}
		if und := md.Undecoded(); len(und) > 0 {
			return c, fmt.Errorf("%s: unknown keys: %v", configFile, und)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return c, err
	}
	s, err := LoadSettings(settingsFile)
	if err != nil {
		return c, err
	}
	if s.Rails != nil {
		c.Rails = *s.Rails
	}
	for k, v := range s.Quotas {
		if c.Quotas == nil {
			c.Quotas = map[string]Quota{}
		}
		c.Quotas[k] = v
	}
	if s.Yield != nil {
		c.Yield.Auto = *s.Yield
	}
	return c, c.Validate()
}

// Validate rejects nonsensical values.
func (c Config) Validate() error {
	if c.Rails.DiskFreeFloorPercent > 90 {
		return errors.New("rails.disk_free_floor_percent must be ≤ 90")
	}
	if c.Limits.MaxFrameBytes < 1<<20 || c.Limits.MaxFrameBytes > 1<<30 {
		return errors.New("limits.max_frame_bytes must be between 1 MiB and 1 GiB")
	}
	if c.Limits.UploadChunkBytes == 0 || c.Limits.UploadChunkBytes > c.Limits.MaxFrameBytes/2 {
		return errors.New("limits.upload_chunk_bytes must be positive and at most half of max_frame_bytes")
	}
	if c.Limits.MaxDims == 0 || c.Limits.MaxDims > 64 {
		return errors.New("limits.max_dims must be 1..64")
	}
	if c.Rails.MaxPIDs < 64 {
		return errors.New("rails.max_pids must be ≥ 64")
	}
	return nil
}

// LoadSettings reads settings.json (missing file = empty settings).
func LoadSettings(path string) (Settings, error) {
	var s Settings
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return s, nil
	}
	if err != nil {
		return s, err
	}
	dec := json.NewDecoder(strings.NewReader(string(b)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&s); err != nil {
		return s, fmt.Errorf("%s: %w", path, err)
	}
	return s, nil
}

// SaveSettings writes settings.json atomically with mode 0600.
func SaveSettings(path string, s Settings) error {
	b, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	return WriteFileAtomic(path, append(b, '\n'), 0o600)
}

// WriteFileAtomic writes via a temp file and rename.
func WriteFileAtomic(path string, data []byte, mode os.FileMode) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	f, err := os.CreateTemp(dir, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tmp := f.Name()
	defer os.Remove(tmp)
	if _, err := f.Write(data); err != nil {
		f.Close()
		return err
	}
	if err := f.Chmod(mode); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
