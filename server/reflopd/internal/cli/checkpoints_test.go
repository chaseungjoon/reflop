// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/reflopai/reflop/server/reflopd/internal/config"
	"github.com/reflopai/reflop/server/reflopd/internal/gpu"
)

// The warning gives the worst case for the configured interval.
func TestCheckpointWarningFigures(t *testing.T) {
	g := []gpu.GPU{{Name: "NVIDIA GeForce RTX 5070", MemoryMiB: 12 * 1024}}
	for _, c := range []struct {
		iv   time.Duration
		want []string
	}{
		{config.DefaultCheckpointInterval, []string{"up to 72 GiB written per hour", "1.7 TiB per day", "writes 6 GiB per hour"}},
		{config.MinCheckpointInterval, []string{"up to 144 GiB written per hour", "3.4 TiB per day", "writes 12 GiB per hour"}},
		{time.Hour, []string{"up to 12 GiB written per hour", "288 GiB per day", "writes 1 GiB per hour"}},
	} {
		w := checkpointWarning(c.iv, g)
		for _, s := range append(c.want, "wear out SSDs", "doubles") {
			if !strings.Contains(w, s) {
				t.Errorf("interval %s: warning lacks %q:\n%s", c.iv, s, w)
			}
		}
	}
	if w := checkpointWarning(config.DefaultCheckpointInterval, nil); !strings.Contains(w, "writes 6 GiB per hour") {
		t.Errorf("without a GPU: %s", w)
	}
}

// Output that is not a terminal gets no escape codes.
func TestRedOnlyOnTerminals(t *testing.T) {
	var b bytes.Buffer
	if s := red(&b, "x"); s != "x" {
		t.Fatalf("colored for a buffer: %q", s)
	}
}

// The floor is hard: the command refuses an interval below it and leaves
// the setting unchanged.
func TestCheckpointsIntervalFloor(t *testing.T) {
	t.Setenv("REFLOP_HOME", t.TempDir())
	cmd := checkpointsCmd()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"on", "--interval", "4m59s"})
	if err := cmd.Execute(); err == nil || !strings.Contains(err.Error(), "at least 5m0s") {
		t.Fatalf("an interval below 5m was accepted: %v", err)
	}
	l, _ := layout()
	s, err := config.LoadSettings(settingsPath(l))
	if err != nil {
		t.Fatal(err)
	}
	if s.Checkpoints != nil && s.Checkpoints.Enabled {
		t.Fatal("checkpoints were turned on by a refused command")
	}
	cmd = checkpointsCmd()
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"on"})
	if err := cmd.Execute(); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "WARNING: checkpoints wear out SSDs") || !strings.Contains(out.String(), "every 10m0s") {
		t.Fatalf("on: %s", out.String())
	}
	cfg, err := config.Load(l.ConfigFile(), settingsPath(l))
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.Checkpoints.Enabled || time.Duration(cfg.Checkpoints.Interval) != config.DefaultCheckpointInterval {
		t.Fatalf("saved %+v", cfg.Checkpoints)
	}
}
