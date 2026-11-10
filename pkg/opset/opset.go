// SPDX-License-Identifier: Apache-2.0

// Package opset exposes opset v1 (generated from opset/v1/opset.yaml, the
// single source of truth, invariant I10) to Go code: reflopd and libreflop.
package opset

// Security flags (PLAN.md §8.3).
const (
	IDX        = 1
	RNG        = 2
	DDS        = 4
	ALLOC      = 8
	ENUM       = 16
	UNINIT     = 32
	FUNC       = 64
	RESTRICTED = 128
	CLIENT     = 256
	BANNED     = 512
)

// Arg is one schema argument.
type Arg struct {
	Name      string
	Type      string // native schema type, e.g. "Tensor?", "SymInt[]"
	KwargOnly bool
}

// Op is one opset entry (one fully qualified overload).
type Op struct {
	ID         uint32
	Name       string
	Tier       int
	Capability string
	Flags      uint32
	Args       []Arg
	Returns    []string
	Enums      map[string][]string
}

var (
	byID   = map[uint32]*Op{}
	byName = map[string]*Op{}
)

func init() {
	for i := range table {
		o := &table[i]
		byID[o.ID] = o
		byName[o.Name] = o
	}
}

// ByID looks up an op id.
func ByID(id uint32) (*Op, bool) { o, ok := byID[id]; return o, ok }

// ByName looks up a fully qualified name such as "aten.mm.default".
func ByName(name string) (*Op, bool) { o, ok := byName[name]; return o, ok }

// All returns every entry in id order.
func All() []Op { return table }
