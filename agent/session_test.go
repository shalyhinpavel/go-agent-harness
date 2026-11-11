package agent

import (
	"context"
	"testing"
	"time"

	"github.com/shalyhinpavel/go-agent-harness/llm"
)

func TestSessionManager_GetOrCreate(t *testing.T) {
	sm := NewSessionManager()

	s1, _ := sm.GetOrCreateSession(context.Background(), "s1", "u1")
	if s1.ID != "s1" || s1.UserID != "u1" {
		t.Fatalf("unexpected session: %+v", s1)
	}
	s2, _ := sm.GetOrCreateSession(context.Background(), "s1", "u1")
	if s1 != s2 {
		t.Fatal("same id must return the same session pointer")
	}
}

func TestSessionManager_EmptyIDGetsUUID(t *testing.T) {
	sm := NewSessionManager()
	s, _ := sm.GetOrCreateSession(context.Background(), "", "u1")
	if s.ID == "" {
		t.Fatal("generated id must not be empty")
	}
}

func TestSessionManager_AppendAndHistory(t *testing.T) {
	sm := NewSessionManager()
	sm.GetOrCreateSession(context.Background(), "s1", "u1")

	sm.AppendMessage(context.Background(), "s1", llm.UserText("hello"))
	sm.AppendMessage(context.Background(), "s1", llm.AssistantText("hi"))

	history := sm.GetHistory("s1")
	if len(history) != 2 {
		t.Fatalf("history = %d messages, want 2", len(history))
	}
	if history[0].Parts[0].Text != "hello" {
		t.Fatalf("unexpected first message: %+v", history[0])
	}

	// Returned slice must be a copy.
	history[0] = *llm.AssistantText("tampered")
	if sm.GetHistory("s1")[0].Parts[0].Text != "hello" {
		t.Fatal("GetHistory must return a defensive copy")
	}
}

func TestSessionManager_AppendNonExistent(t *testing.T) {
	sm := NewSessionManager()
	sm.AppendMessage(context.Background(), "ghost", llm.UserText("x")) // must not panic
	if h := sm.GetHistory("ghost"); h != nil {
		t.Fatal("history for unknown session must be nil")
	}
}

func TestSessionManager_CompactHistory(t *testing.T) {
	sm := NewSessionManager()
	sm.GetOrCreateSession(context.Background(), "s1", "u1")

	for i := 0; i < 10; i++ {
		sm.AppendMessage(context.Background(), "s1", llm.UserText("msg"))
	}
	compacted := sm.CompactHistory("s1", 4)
	if len(compacted) != 4 {
		t.Fatalf("compacted = %d, want 4", len(compacted))
	}

	// Must be a copy, not a sub-slice of history.
	compacted[0] = *llm.AssistantText("tampered")
	if sm.GetHistory("s1")[6].Parts[0].Text != "msg" {
		t.Fatal("CompactHistory must return an independent copy")
	}
}

func TestSessionManager_CompactHistoryPreservesToolChains(t *testing.T) {
	sm := NewSessionManager()
	sm.GetOrCreateSession(context.Background(), "s1", "u1")

	// 3 plain messages + tool_call + tool_result + 3 plain.
	sm.AppendMessage(context.Background(), "s1", llm.UserText("1"))
	sm.AppendMessage(context.Background(), "s1", llm.AssistantText("2"))
	sm.AppendMessage(context.Background(), "s1", llm.UserText("3"))
	sm.AppendMessage(context.Background(), "s1", &llm.Message{Role: llm.RoleAssistant, Parts: []llm.Part{llm.ToolCallPart(llm.ToolCall{ID: "c1", Name: "t"})}})
	sm.AppendMessage(context.Background(), "s1", &llm.Message{Role: llm.RoleTool, Parts: []llm.Part{llm.ToolResultPart(llm.ToolResultData{CallID: "c1"})}})
	sm.AppendMessage(context.Background(), "s1", llm.AssistantText("4"))
	sm.AppendMessage(context.Background(), "s1", llm.UserText("5"))

	// Cut lands inside the chain (indices 4..7 window starts at tool_result) —
	// the cut must move back to keep the call+result pair intact.
	compacted := sm.CompactHistory("s1", 4)
	first := compacted[0]
	if isToolResult(&first) {
		t.Fatal("compacted history must not start with a dangling tool_result")
	}
	// Verify chain integrity: any tool_call in the window has its result right after.
	for i := range compacted {
		if hasToolCall(&compacted[i]) && i+1 < len(compacted) && !isToolResult(&compacted[i+1]) {
			t.Fatalf("tool_call at %d not followed by result", i)
		}
	}
}

func TestSessionManager_TTLReset(t *testing.T) {
	sm := NewSessionManager(WithSessionTTL(5 * time.Millisecond))
	sm.GetOrCreateSession(context.Background(), "s1", "u1")
	sm.AppendMessage(context.Background(), "s1", llm.UserText("old"))

	time.Sleep(10 * time.Millisecond)

	s, _ := sm.GetOrCreateSession(context.Background(), "s1", "u1")
	if len(s.History) != 0 {
		t.Fatalf("expired session must reset, got %d messages", len(s.History))
	}
}

func TestSessionManager_MaxSessionsEviction(t *testing.T) {
	sm := NewSessionManager(WithMaxSessions(2))
	sm.GetOrCreateSession(context.Background(), "a", "u")
	time.Sleep(1 * time.Millisecond)
	sm.GetOrCreateSession(context.Background(), "b", "u")
	time.Sleep(1 * time.Millisecond)
	sm.GetOrCreateSession(context.Background(), "c", "u") // evicts "a" (oldest)

	if h := sm.GetHistory("a"); h != nil {
		t.Fatal("oldest session must be evicted")
	}
	if h := sm.GetHistory("c"); h == nil {
		t.Fatal("newest session must survive")
	}
}

// hasToolCall reports whether the message requests tool invocations.
func hasToolCall(msg *llm.Message) bool {
	if msg == nil {
		return false
	}
	for _, p := range msg.Parts {
		if p.Kind == llm.PartToolCall {
			return true
		}
	}
	return false
}
