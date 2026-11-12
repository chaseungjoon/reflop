// SPDX-License-Identifier: Apache-2.0

package gpu

import "testing"

func TestDetectDoesNotPanic(t *testing.T) {
	inf := Detect()
	t.Logf("%+v", inf)
}

func TestParseMemoryUse(t *testing.T) {
	if u, tot, ok := parseMemoryUse("11723, 12227\n"); !ok || u != 11723 || tot != 12227 {
		t.Fatalf("got %d %d %v", u, tot, ok)
	}
	for _, s := range []string{"", "[N/A], [N/A]", "11723", "1, 0"} {
		if _, _, ok := parseMemoryUse(s); ok {
			t.Errorf("%q parsed", s)
		}
	}
}
