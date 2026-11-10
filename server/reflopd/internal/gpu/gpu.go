// SPDX-License-Identifier: Apache-2.0

// Package gpu detects NVIDIA GPUs and the driver without loading CUDA in
// reflopd (reflopd never touches the GPU, PLAN.md §9.1). It reads
// /proc/driver/nvidia and, when present, asks nvidia-smi. Detection runs
// only at Owner-triggered times (startup, doctor, runtime install), never
// in response to a Guest request.
package gpu

import (
	"bufio"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// GPU is one detected device.
type GPU struct {
	Index       int    `json:"index"`
	UUID        string `json:"uuid"`
	Name        string `json:"name"`
	MemoryMiB   uint64 `json:"memory_mib"`
	CCMajor     int    `json:"cc_major"` // 0 if unknown
	CCMinor     int    `json:"cc_minor"`
	BusID       string `json:"bus_id,omitempty"`
	DeviceMinor int    `json:"device_minor"` // /dev/nvidia<N>; -1 if unknown
}

// Info is the detection result.
type Info struct {
	Driver      string `json:"driver"` // e.g. "595.71.05"; "" if none
	DriverMajor int    `json:"driver_major"`
	DriverMinor int    `json:"driver_minor"`
	Open        bool   `json:"open_kernel_module"`
	WSL         bool   `json:"wsl"`
	GPUs        []GPU  `json:"gpus"`
	Notes       []string
}

var driverRe = regexp.MustCompile(`(\d+)\.(\d+)(?:\.(\d+))?`)

// Detect inspects the machine.
func Detect() Info {
	var inf Info
	if b, err := os.ReadFile("/proc/driver/nvidia/version"); err == nil {
		s := string(b)
		inf.Open = strings.Contains(s, "Open Kernel Module")
		if m := driverRe.FindStringSubmatch(firstLine(s)); m != nil {
			inf.Driver = strings.Trim(m[0], ".")
		}
	}
	if _, err := os.Stat("/dev/dxg"); err == nil {
		inf.WSL = true
	}
	smi := findSMI(inf.WSL)
	if smi != "" {
		inf.GPUs, inf.Driver = querySMI(smi, inf.Driver, &inf.Notes)
	}
	if len(inf.GPUs) == 0 {
		inf.GPUs = fromProc()
	}
	if m := driverRe.FindStringSubmatch(inf.Driver); m != nil {
		inf.DriverMajor, _ = strconv.Atoi(m[1])
		inf.DriverMinor, _ = strconv.Atoi(m[2])
	}
	if inf.Driver == "" && len(inf.GPUs) == 0 {
		inf.Notes = append(inf.Notes, "no NVIDIA driver found; only the CPU device will be served")
	}
	return inf
}

func firstLine(s string) string { l, _, _ := strings.Cut(s, "\n"); return l }

func findSMI(wsl bool) string {
	cands := []string{"/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi", "/bin/nvidia-smi"}
	if wsl {
		cands = append([]string{"/usr/lib/wsl/lib/nvidia-smi"}, cands...)
	}
	for _, c := range cands {
		if st, err := os.Stat(c); err == nil && st.Mode().IsRegular() {
			return c
		}
	}
	if p, err := exec.LookPath("nvidia-smi"); err == nil {
		return p
	}
	return ""
}

func querySMI(smi, driver string, notes *[]string) ([]GPU, string) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, smi,
		"--query-gpu=index,uuid,name,memory.total,compute_cap,driver_version,pci.bus_id",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		*notes = append(*notes, "nvidia-smi failed: "+err.Error())
		return nil, driver
	}
	var gs []GPU
	sc := bufio.NewScanner(strings.NewReader(string(out)))
	for sc.Scan() {
		f := strings.Split(sc.Text(), ",")
		if len(f) < 7 {
			continue
		}
		for i := range f {
			f[i] = strings.TrimSpace(f[i])
		}
		g := GPU{UUID: f[1], Name: f[2], BusID: f[6], DeviceMinor: -1}
		g.Index, _ = strconv.Atoi(f[0])
		g.MemoryMiB, _ = strconv.ParseUint(f[3], 10, 64)
		if a, b, ok := strings.Cut(f[4], "."); ok {
			g.CCMajor, _ = strconv.Atoi(a)
			g.CCMinor, _ = strconv.Atoi(b)
		}
		if driver == "" {
			driver = f[5]
		}
		g.DeviceMinor = minorForBus(g.BusID)
		gs = append(gs, g)
	}
	return gs, driver
}

// minorForBus maps a PCI bus id to the /dev/nvidiaN minor number using
// /proc/driver/nvidia/gpus/<bus>/information.
func minorForBus(bus string) int {
	bus = strings.ToLower(bus)
	ents, _ := os.ReadDir("/proc/driver/nvidia/gpus")
	for _, e := range ents {
		n := strings.ToLower(e.Name())
		if !strings.HasSuffix(bus, strings.TrimPrefix(n, "0000")) && !strings.HasSuffix(n, strings.TrimPrefix(bus, "0000")) {
			continue
		}
		if m := readMinor(filepath.Join("/proc/driver/nvidia/gpus", e.Name(), "information")); m >= 0 {
			return m
		}
	}
	return -1
}

func readMinor(path string) int {
	b, err := os.ReadFile(path)
	if err != nil {
		return -1
	}
	for _, l := range strings.Split(string(b), "\n") {
		k, v, ok := strings.Cut(l, ":")
		if ok && strings.TrimSpace(k) == "Device Minor" {
			n, err := strconv.Atoi(strings.TrimSpace(v))
			if err == nil {
				return n
			}
		}
	}
	return -1
}

func fromProc() []GPU {
	ents, _ := os.ReadDir("/proc/driver/nvidia/gpus")
	var gs []GPU
	for i, e := range ents {
		p := filepath.Join("/proc/driver/nvidia/gpus", e.Name(), "information")
		b, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		g := GPU{Index: i, BusID: e.Name(), DeviceMinor: readMinor(p)}
		for _, l := range strings.Split(string(b), "\n") {
			k, v, ok := strings.Cut(l, ":")
			if !ok {
				continue
			}
			switch strings.TrimSpace(k) {
			case "Model":
				g.Name = strings.TrimSpace(v)
			case "GPU UUID":
				g.UUID = strings.TrimSpace(v)
			}
		}
		gs = append(gs, g)
	}
	return gs
}

// DeviceNodes lists the /dev nodes a worker needs for the given GPUs.
func DeviceNodes(gs []GPU, wsl bool) []string {
	if wsl {
		return []string{"/dev/dxg"}
	}
	nodes := []string{"/dev/nvidiactl", "/dev/nvidia-uvm"}
	for _, g := range gs {
		if g.DeviceMinor >= 0 {
			nodes = append(nodes, "/dev/nvidia"+strconv.Itoa(g.DeviceMinor))
		}
	}
	return nodes
}
