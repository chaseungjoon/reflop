// SPDX-License-Identifier: Apache-2.0

package server

import (
	"errors"
	"strings"
	"testing"

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
	if !strings.Contains(n, "11723 of 12227 MiB in use") || !strings.Contains(n, "nvidia-smi") ||
		!strings.Contains(n, "OUT_OF_MEMORY") {
		t.Fatalf("note: %s", n)
	}
	if n := gpuFullNote(g, func(gpu.GPU) (uint64, uint64, bool) { return 0, 0, false }); strings.Contains(n, "MiB") {
		t.Fatalf("note without nvidia-smi: %s", n)
	}
}
