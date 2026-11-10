// SPDX-License-Identifier: Apache-2.0

package server

import (
	"sync"
	"time"

	pb "github.com/chaseungjoon/reflop/server/reflopd/internal/pb/reflopv1"
)

// broadcaster delivers informational events to subscribed connections.
// Events have a fixed schema and never cause client actions (§6.2, §12).
type broadcaster struct {
	mu   sync.Mutex
	subs map[*clientConn]bool
}

func newBroadcaster() *broadcaster { return &broadcaster{subs: map[*clientConn]bool{}} }

func (b *broadcaster) subscribe(c *clientConn) {
	b.mu.Lock()
	b.subs[c] = true
	b.mu.Unlock()
}

func (b *broadcaster) unsubscribe(c *clientConn) {
	b.mu.Lock()
	delete(b.subs, c)
	b.mu.Unlock()
}

func (b *broadcaster) publish(kind pb.Event_Kind, text string) {
	if len(text) > 1000 {
		text = text[:1000]
	}
	ev := &pb.Event{Kind: kind, Text: text, TimeUnixMs: time.Now().UnixMilli()}
	b.mu.Lock()
	subs := make([]*clientConn, 0, len(b.subs))
	for c := range b.subs {
		subs = append(subs, c)
	}
	b.mu.Unlock()
	for _, c := range subs {
		_ = c.send(&pb.Envelope{RequestId: c.eventsRID.Load(), Body: &pb.Envelope_Event{Event: ev}})
	}
}
