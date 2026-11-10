// SPDX-License-Identifier: Apache-2.0

//go:build reflop_probes

package server

import _ "embed"

// Probes are compiled in only with -tags reflop_probes (test builds, §17.5).
//
//go:embed worker/testprobes/_probes.py
var probes []byte

func init() { Probes = probes }
