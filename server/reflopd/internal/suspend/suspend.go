// SPDX-License-Identifier: Apache-2.0

// Package suspend moves a worker's GPU memory to host RAM and back with the
// CUDA driver's process checkpoint API (ADR-030). reflopd is a static Go
// binary without libcuda, so a small embedded helper runs with the pinned
// runtime's Python. It inherits reflopd's sandbox and acts only on reflopd's
// own workers.
package suspend

import (
	"bytes"
	"context"
	_ "embed"
	"errors"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

//go:embed helper.py
var helperPy string

// LockTimeout is how long suspend waits for running GPU work to finish
// before it gives up (the worker then keeps running).
const LockTimeout = 10 * time.Second

// Errors a caller tells apart.
var (
	ErrUnsupported = errors.New("the NVIDIA driver cannot suspend processes (driver 550 or newer is needed)")
	ErrNotWorker   = errors.New("not a reflop worker of this reflopd")
	ErrLockTimeout = errors.New("GPU work did not finish in time")
)

// Helper runs the embedded helper with python.
type Helper struct {
	Python string
	// GPUUUID limits the helper's own CUDA view to the worker's GPU.
	GPUUUID string
}

func (h Helper) run(op string, pid int) error {
	ctx, cancel := context.WithTimeout(context.Background(), LockTimeout+2*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, h.Python, "-I", "-B", "-c", helperPy, op, strconv.Itoa(pid),
		strconv.FormatInt(LockTimeout.Milliseconds(), 10))
	cmd.Env = []string{"LANG=C.UTF-8", "CUDA_DEVICE_ORDER=PCI_BUS_ID"}
	if h.GPUUUID != "" {
		cmd.Env = append(cmd.Env, "CUDA_VISIBLE_DEVICES="+h.GPUUUID)
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	err := cmd.Run()
	if err == nil {
		return nil
	}
	msg := strings.TrimSpace(stderr.String())
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		switch ee.ExitCode() {
		case 3:
			return fmt.Errorf("%w: %s", ErrUnsupported, msg)
		case 4:
			return fmt.Errorf("%w: %s", ErrNotWorker, msg)
		case 5:
			return fmt.Errorf("%w: %s", ErrLockTimeout, msg)
		}
	}
	if msg == "" {
		msg = err.Error()
	}
	return fmt.Errorf("suspend helper (%s): %s", op, msg)
}

// Suspend locks the worker's CUDA calls and moves its GPU memory to its own
// host memory. On ErrLockTimeout the worker keeps running unchanged.
func (h Helper) Suspend(pid int) error { return h.run("suspend", pid) }

// Resume moves the memory back to the GPU and unlocks the worker.
func (h Helper) Resume(pid int) error { return h.run("resume", pid) }
