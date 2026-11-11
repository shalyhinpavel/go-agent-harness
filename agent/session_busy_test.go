package agent

import (
	"context"
	"testing"
	"time"

	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// A turn that outlives the session TTL keeps its session: a second request
// with the same id must not reset the history under the running turn.
func TestGetOrCreateSession_TTLDoesNotEvictABusySession(t *testing.T) {
	ctx := context.Background()
	sm := NewSessionManager(WithSessionTTL(time.Millisecond))
	if _, err := sm.GetOrCreateSession(ctx, "s", "u"); err != nil {
		t.Fatal(err)
	}
	release, ok := sm.BeginTurn("s")
	if !ok {
		t.Fatal("the first turn must start")
	}
	defer release()
	if err := sm.AppendMessage(ctx, "s", llm.UserText("kept")); err != nil {
		t.Fatal(err)
	}

	time.Sleep(10 * time.Millisecond) // well past the TTL
	if _, err := sm.GetOrCreateSession(ctx, "s", "u"); err != nil {
		t.Fatal(err)
	}
	if n := len(sm.GetHistory("s")); n != 1 {
		t.Fatalf("history = %d messages after a second request mid-turn, want 1", n)
	}
}
