// SPDX-License-Identifier: Apache-2.0

package server

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	pb "github.com/chaseungjoon/reflop/pkg/reflopv1"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/auth"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/gpu"
)

// The worker error from `reflopd up` on 0.1.9 when the desktop held 11.4 GiB
// of a 12 GiB GPU. The self-test failed and reflopd did not start at all.
const ctxRetainOOM = `worker initialization failed: CUDA error: out of memory
Search for ` + "`cudaErrorMemoryAllocation'" + ` in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
The CUDA driver logged these messages, which may provide useful details:
Returning 2 (CUDA_ERROR_OUT_OF_MEMORY) from cuDevicePrimaryCtxRetain`

func TestGPUFullIsRecognized(t *testing.T) {
	g := &DeviceSlot{Name: "cuda:0", GPU: &gpu.GPU{}}
	cpu := &DeviceSlot{Name: "cpu"}
	if !gpuFull(g, errors.New(ctxRetainOOM)) {
		t.Fatal("a CUDA context OOM on a GPU slot is not recognized")
	}
	for _, c := range []struct {
		slot *DeviceSlot
		err  string
	}{
		{cpu, ctxRetainOOM}, // never skip the CPU, which proves the sandbox works
		{g, "worker initialization failed: CUDA error: no kernel image is available for execution on the device"},
		{g, "seccomp filter: program too long"},
		{g, "host memory safety rail (OOM in worker cgroup)"},
		{g, "landlock: out of memory"}, // not CUDA
	} {
		if gpuFull(c.slot, errors.New(c.err)) {
			t.Errorf("%s: %q was taken for a full GPU", c.slot.Name, c.err)
		}
	}
}

func TestGPUFullNote(t *testing.T) {
	g := &DeviceSlot{Name: "cuda:0", GPU: &gpu.GPU{UUID: "GPU-x"}}
	n := gpuFullNote(g, func(gpu.GPU) (uint64, uint64, bool) { return 11723, 12227, true })
	if !strings.Contains(n, "11723 of 12227 MiB in use") || !strings.Contains(n, "nvidia-smi") {
		t.Fatalf("note: %s", n)
	}
	if n := gpuFullNote(g, func(gpu.GPU) (uint64, uint64, bool) { return 0, 0, false }); strings.Contains(n, "MiB") {
		t.Fatalf("note without nvidia-smi: %s", n)
	}
}

// A GPU that was full at startup has no probed operator table, so it must not
// be served (spawnWorker would start its workers without the Owner's op
// filter) or advertised until a probe succeeds.
func TestPendingGPUIsNotServed(t *testing.T) {
	s := testServer(8)
	g := &DeviceSlot{PB: &pb.Device{Kind: pb.DeviceKind_DEVICE_CUDA}, Name: "cuda:0", GPU: &gpu.GPU{}}
	c := &DeviceSlot{PB: &pb.Device{Kind: pb.DeviceKind_DEVICE_CPU}, Name: "cpu", Enabled: []uint32{1}}
	s.allSlots = []*DeviceSlot{g, c}
	s.pending = map[*DeviceSlot]bool{g: true}
	s.publishSlots()

	if d := s.devices(); len(d) != 1 || d[0] != c {
		t.Fatalf("served: %v", d)
	}
	if sl := s.slotFor(nil); sl != c {
		t.Fatalf("default device: %v, want cpu", sl)
	}
	if sl := s.slotFor(g.PB); sl != nil {
		t.Fatal("the pending GPU can be opened")
	}
	for _, d := range s.capabilitiesFor(&auth.Identity{Name: "alice"}).Devices {
		if d.GetDevice().GetKind() == pb.DeviceKind_DEVICE_CUDA {
			t.Fatal("the pending GPU is advertised")
		}
	}
	if p := s.pendingDevices(); len(p) != 1 || p[0] != "cuda:0" {
		t.Fatalf("pending: %v", p)
	}

	// Not due yet: no probe is started (a probe would need a runtime and panic here).
	s.lastRetry = time.Now()
	s.retryPending(context.Background())
	if s.probing.Load() {
		t.Fatal("probed before probeRetryInterval")
	}

	// Once probed, it is published first again, as at startup.
	delete(s.pending, g)
	s.publishSlots()
	if d := s.devices(); len(d) != 2 || d[0] != g || s.slotFor(nil) != g {
		t.Fatalf("after the probe: %v", d)
	}
}
