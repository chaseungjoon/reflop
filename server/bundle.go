// SPDX-License-Identifier: Apache-2.0

// Package server embeds the Python worker into reflopd so that a single
// binary carries everything the Owner needs (PLAN.md §4.2).
package server

import (
	"embed"
	"io/fs"
)

//go:embed all:worker/reflop_worker
var workerFS embed.FS

// Worker returns the reflop_worker package tree (rooted at reflop_worker/).
func Worker() fs.FS {
	sub, err := fs.Sub(workerFS, "worker")
	if err != nil {
		panic(err)
	}
	return sub
}
