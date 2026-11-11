package agent

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// toolLoop builds a session holding a task followed by n tool exchanges
// (assistant call + tool result), the shape of a long agentic run.
func toolLoop(t *testing.T, sm *SessionManager, id string, n int, result string) {
	t.Helper()
	sm.GetOrCreateSession(context.Background(), id, "u")
	sm.AppendMessage(context.Background(), id, llm.UserText("TASK"))
	for i := 0; i < n; i++ {
		appendExchange(sm, id, i, result)
	}
}

func appendExchange(sm *SessionManager, id string, i int, result string) {
	callID := "c" + strings.Repeat("x", i%3) + string(rune('a'+i%26))
	sm.AppendMessage(context.Background(), id, &llm.Message{Role: llm.RoleAssistant, Parts: []llm.Part{
		llm.ThinkingPart("why"), llm.ToolCallPart(llm.ToolCall{ID: callID, Name: "t"})}})
	sm.AppendMessage(context.Background(), id, &llm.Message{Role: llm.RoleTool, Parts: []llm.Part{
		llm.ToolResultPart(llm.ToolResultData{CallID: callID, Content: result})}})
}

func checkChains(t *testing.T, w []llm.Message) {
	t.Helper()
	for i := range w {
		if isToolResult(&w[i]) && (i == 0 || !hasToolCall(&w[i-1]) && !isToolResult(&w[i-1])) {
			t.Fatalf("tool result at %d has no call before it", i)
		}
		if hasToolCall(&w[i]) && (i+1 >= len(w) || !isToolResult(&w[i+1])) {
			t.Fatalf("tool call at %d has no result after it", i)
		}
	}
}

// Regression: a history that is one long tool chain used to defeat the
// window entirely (the cut walked back over every tool message to the
// first user message), so the request grew until the context overflowed.
func TestWindow_LongToolChainIsBounded(t *testing.T) {
	sm := NewSessionManager()
	toolLoop(t, sm, "s", 200, "ok")

	if got := len(sm.CompactHistory("s", 10)); got > 10 {
		t.Fatalf("CompactHistory returned %d messages for a cap of 10", got)
	}
	w := sm.Window("s", WindowPolicy{MaxItems: 10, TrimTo: 1, PinTask: true})
	if len(w) > 11 {
		t.Fatalf("window = %d messages, want at most task + 10", len(w))
	}
	if w[0].Role != llm.RoleUser || w[0].Parts[0].Text != "TASK" {
		t.Fatalf("pinned task missing: %+v", w[0])
	}
	checkChains(t, w[1:])
	for _, m := range w {
		for _, p := range m.Parts {
			if p.Kind == llm.PartThinking {
				t.Fatal("a cut window must not replay thinking")
			}
		}
	}
}

func TestWindow_HysteresisKeepsPrefixStable(t *testing.T) {
	sm := NewSessionManager()
	toolLoop(t, sm, "s", 10, "ok") // task + 20 messages
	p := WindowPolicy{MaxItems: 20, TrimTo: 0.5, PinTask: true}

	first := sm.Window("s", p) // fits: nothing dropped
	if len(first) != 21 {
		t.Fatalf("fitting history must pass whole, got %d", len(first))
	}

	appendExchange(sm, "s", 10, "ok") // 22 > 20: trim to <= 10
	trimmed := sm.Window("s", p)
	if n := len(trimmed) - 1; n > 10 || n < 8 {
		t.Fatalf("trim kept %d messages, want about half of 20", n)
	}
	checkChains(t, trimmed[1:])

	// Growing back towards the cap keeps the same prefix byte for byte.
	for i := 11; i < 15; i++ {
		appendExchange(sm, "s", i, "ok")
		next := sm.Window("s", p)
		if !reflect.DeepEqual(next[:len(trimmed)], trimmed) {
			t.Fatalf("prefix changed at exchange %d without an overflow", i)
		}
		trimmed = next
	}
}

func TestWindow_CharCapBoundsLargeResults(t *testing.T) {
	sm := NewSessionManager()
	big := strings.Repeat("r", 5000)
	toolLoop(t, sm, "s", 40, big)

	w := sm.Window("s", WindowPolicy{MaxItems: 1000, MaxChars: 30000, TrimTo: 0.75, PinTask: true})
	size := 0
	for i := 1; i < len(w); i++ {
		size += messageChars(&w[i])
	}
	if size > 30000 {
		t.Fatalf("window holds %d chars, cap is 30000", size)
	}
	if size < 15000 {
		t.Fatalf("window trimmed too far: %d chars", size)
	}
	checkChains(t, w[1:])
}

func TestWindow_SingleOversizedExchangeStillSent(t *testing.T) {
	sm := NewSessionManager()
	toolLoop(t, sm, "s", 3, strings.Repeat("r", 50000))
	w := sm.Window("s", WindowPolicy{MaxChars: 1000, PinTask: true})
	// Nothing fits: the newest exchange goes alone after the task and
	// the context guard decides whether the request may be sent.
	if len(w) != 3 || !hasToolCall(&w[1]) || !isToolResult(&w[2]) {
		t.Fatalf("want task + last exchange, got %d messages", len(w))
	}
}

func TestWindow_ResetsWhenSessionRestarts(t *testing.T) {
	sm := NewSessionManager()
	toolLoop(t, sm, "s", 30, "ok")
	sm.Window("s", WindowPolicy{MaxItems: 10, PinTask: true})
	// A shorter history (fresh session under the same id) must not reuse
	// a stale start beyond its end.
	sm.mu.Lock()
	sm.sessions["s"].History = sm.sessions["s"].History[:5]
	sm.mu.Unlock()
	if w := sm.Window("s", WindowPolicy{MaxItems: 10, PinTask: true}); len(w) != 5 {
		t.Fatalf("window over a 5-message history = %d", len(w))
	}
}
