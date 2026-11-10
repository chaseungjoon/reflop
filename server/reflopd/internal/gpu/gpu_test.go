// SPDX-License-Identifier: Apache-2.0

package gpu

import "testing"

func TestDetectDoesNotPanic(t *testing.T) {
	inf := Detect()
	t.Logf("%+v", inf)
}
