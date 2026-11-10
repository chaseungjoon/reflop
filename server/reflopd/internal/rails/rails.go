// SPDX-License-Identifier: Apache-2.0

// Package rails implements the always-on safety rails of PLAN.md §9.6:
// a disk free floor on the store filesystem and a host RAM floor.
package rails

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"

	"golang.org/x/sys/unix"

	"github.com/chaseungjoon/reflop/server/reflopd/internal/config"
)

// DiskUsage describes a filesystem.
type DiskUsage struct {
	Total, Free uint64
}

// Disk returns usage of the filesystem holding path (space available to
// unprivileged users).
func Disk(path string) (DiskUsage, error) {
	var st unix.Statfs_t
	if err := unix.Statfs(path, &st); err != nil {
		return DiskUsage{}, err
	}
	bs := uint64(st.Bsize)
	return DiskUsage{Total: st.Blocks * bs, Free: st.Bavail * bs}, nil
}

// DiskFloor is the free space that must remain: the larger of the percent
// and the absolute floor.
func DiskFloor(r config.Rails, total uint64) uint64 {
	pct := total / 100 * uint64(r.DiskFreeFloorPercent)
	if uint64(r.DiskFreeFloorBytes) > pct {
		return uint64(r.DiskFreeFloorBytes)
	}
	return pct
}

// CheckDiskWrite fails if writing n more bytes under path would leave less
// free space than the floor.
func CheckDiskWrite(r config.Rails, path string, n uint64) error {
	u, err := Disk(path)
	if err != nil {
		return err
	}
	floor := DiskFloor(r, u.Total)
	if u.Free < n || u.Free-n < floor {
		return fmt.Errorf("disk safety rail: writing %s would leave %s free on the store filesystem; the floor is %s",
			config.Bytes(n), config.Bytes(sub(u.Free, n)), config.Bytes(floor))
	}
	return nil
}

// DiskBudget is how many bytes may still be written under path.
func DiskBudget(r config.Rails, path string) uint64 {
	u, err := Disk(path)
	if err != nil {
		return 0
	}
	return sub(u.Free, DiskFloor(r, u.Total))
}

func sub(a, b uint64) uint64 {
	if a < b {
		return 0
	}
	return a - b
}

// Mem is a /proc/meminfo snapshot in bytes.
type Mem struct {
	Total, Available uint64
}

// Memory reads /proc/meminfo.
func Memory() (Mem, error) {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return Mem{}, err
	}
	defer f.Close()
	var m Mem
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		k, v, ok := strings.Cut(sc.Text(), ":")
		if !ok {
			continue
		}
		fs := strings.Fields(v)
		if len(fs) == 0 {
			continue
		}
		n, _ := strconv.ParseUint(fs[0], 10, 64)
		n *= 1024
		switch k {
		case "MemTotal":
			m.Total = n
		case "MemAvailable":
			m.Available = n
		}
	}
	if m.Total == 0 {
		return m, fmt.Errorf("could not parse /proc/meminfo")
	}
	return m, nil
}

// RAMFloor returns the host RAM that must stay available. The configured
// floor never exceeds half of physical RAM, so small machines still work.
func RAMFloor(r config.Rails, total uint64) uint64 {
	f := uint64(r.HostRAMFloorBytes)
	if f > total/2 {
		f = total / 2
	}
	return f
}

// WorkerBudget is how much host memory all workers together may use now,
// given what they already use: keep RAMFloor available for the Owner.
func WorkerBudget(r config.Rails, m Mem, workersUsing uint64) uint64 {
	return sub(m.Available+workersUsing, RAMFloor(r, m.Total))
}
