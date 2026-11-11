package agent

import (
	"context"
	"errors"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// fixedCompactor summarizes in one line and spends nothing.
func fixedCompactor(context.Context, []llm.Message) (string, llm.Usage, error) {
	return "S", llm.Usage{}, nil
}

func callMsg(id string) *llm.Message {
	return &llm.Message{Role: llm.RoleAssistant, Parts: []llm.Part{llm.ToolCallPart(llm.ToolCall{ID: id, Name: "t"})}}
}

func resultMsg(id string) *llm.Message {
	return &llm.Message{Role: llm.RoleTool, Parts: []llm.Part{llm.ToolResultPart(llm.ToolResultData{CallID: id})}}
}

// The cut never starts on a tool result: it moves back onto the call that
// produced it, so the rewritten history stays a valid conversation.
func TestCompactSession_CutLandsOnACallNotItsResult(t *testing.T) {
	ctx := context.Background()
	sessions := NewSessionManager()
	sess, err := sessions.GetOrCreateSession(ctx, "s", "u")
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range []*llm.Message{
		llm.UserText("a"), llm.AssistantText("b"), llm.UserText("c"),
		callMsg("c1"), resultMsg("c1"), llm.AssistantText("d"),
		llm.UserText("e"), callMsg("c2"), resultMsg("c2"), llm.UserText("f"),
	} {
		if err := sessions.AppendMessage(ctx, "s", m); err != nil {
			t.Fatal(err)
		}
	}
	// Ten messages: keeping six would cut at index 4, a tool result.
	e := NewAgentEngine(AgentConfig{Compactor: fixedCompactor}, nil, sessions)
	if _, err := e.compactSession(ctx, sess, "t"); err != nil {
		t.Fatal(err)
	}

	h := sessions.GetHistory("s")
	if h[0].Role != llm.RoleUser || !strings.Contains(h[0].Parts[0].Text, "S") {
		t.Fatalf("the summary must come first, got %+v", h[0])
	}
	if !hasToolCall(&h[1]) || !isToolResult(&h[2]) {
		t.Fatalf("the window must start with the call and then its result, got %+v", h[1:3])
	}
	for i := range h {
		if isToolResult(&h[i]) && (i == 0 || !hasToolCall(&h[i-1])) {
			t.Fatalf("tool result at %d has no call in front of it", i)
		}
	}
}

// When the cut lands on a user turn, the summary joins that turn so the roles
// keep alternating.
func TestCompactSession_SummaryJoinsAUserTurn(t *testing.T) {
	ctx := context.Background()
	sessions := NewSessionManager()
	sess, err := sessions.GetOrCreateSession(ctx, "s", "u")
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 8; i++ {
		msg := llm.UserText(string(rune('a' + i)))
		if i%2 == 1 {
			msg = llm.AssistantText(string(rune('a' + i)))
		}
		if err := sessions.AppendMessage(ctx, "s", msg); err != nil {
			t.Fatal(err)
		}
	}
	e := NewAgentEngine(AgentConfig{Compactor: fixedCompactor}, nil, sessions)
	if _, err := e.compactSession(ctx, sess, "t"); err != nil {
		t.Fatal(err)
	}

	h := sessions.GetHistory("s")
	if len(h) != 6 {
		t.Fatalf("history = %d messages, want 6: the summary joins the first kept user turn", len(h))
	}
	if h[0].Role != llm.RoleUser || len(h[0].Parts) != 2 || !strings.HasSuffix(h[0].Parts[0].Text, "\n\n") || h[0].Parts[1].Text != "c" {
		t.Fatalf("the first kept user turn must carry the summary separated from its own text, got %+v", h[0].Parts)
	}
}

// The rewrite reaches the disk: a reopened store holds the same history and
// the same owner.
func TestCompactSession_RewriteSurvivesARestart(t *testing.T) {
	ctx := context.Background()
	store, err := NewFileSessionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	sessions := NewSessionManager(WithSessionStore(store))
	sess, err := sessions.GetOrCreateSession(ctx, "s", "u1")
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 8; i++ {
		msg := llm.UserText("q")
		if i%2 == 1 {
			msg = llm.AssistantText("a")
		}
		if err := sessions.AppendMessage(ctx, "s", msg); err != nil {
			t.Fatal(err)
		}
	}
	e := NewAgentEngine(AgentConfig{Compactor: fixedCompactor}, nil, sessions)
	if _, err := e.compactSession(ctx, sess, "t"); err != nil {
		t.Fatal(err)
	}
	want := sessions.GetHistory("s")

	reopened := NewSessionManager(WithSessionStore(store))
	got, err := reopened.GetOrCreateSession(ctx, "s", "someone-else")
	if err != nil {
		t.Fatal(err)
	}
	if got.UserID != "u1" {
		t.Fatalf("owner = %q, want u1: the rewrite must keep the owner", got.UserID)
	}
	if len(got.History) != len(want) {
		t.Fatalf("reopened history = %d messages, want %d", len(got.History), len(want))
	}
	for i := range want {
		if got.History[i].Role != want[i].Role || got.History[i].Parts[0].Text != want[i].Parts[0].Text {
			t.Fatalf("message %d differs after reopen: %+v vs %+v", i, got.History[i], want[i])
		}
	}
}

// refusingStore lets every write through except the compaction rewrite.
type refusingStore struct{ SessionStore }

func (refusingStore) Compact(context.Context, string, []llm.Message) error {
	return errors.New("disk full")
}

// A rewrite the store refuses leaves the in-memory history as it was: memory
// never runs ahead of what is durable.
func TestCompactSession_RefusedRewriteKeepsTheHistory(t *testing.T) {
	ctx := context.Background()
	file, err := NewFileSessionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	sessions := NewSessionManager(WithSessionStore(refusingStore{file}))
	sess, err := sessions.GetOrCreateSession(ctx, "s", "u")
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 8; i++ {
		if err := sessions.AppendMessage(ctx, "s", llm.UserText("q")); err != nil {
			t.Fatal(err)
		}
	}
	e := NewAgentEngine(AgentConfig{Compactor: fixedCompactor}, nil, sessions)
	if _, err := e.compactSession(ctx, sess, "t"); err == nil || !strings.Contains(err.Error(), "disk full") {
		t.Fatalf("the refused rewrite must be reported, got %v", err)
	}
	if h := sessions.GetHistory("s"); len(h) != 8 {
		t.Fatalf("history = %d messages after a refused rewrite, want 8", len(h))
	}
}

// tailUTF8 never returns a broken character.
func TestTailUTF8KeepsCharacterBoundaries(t *testing.T) {
	s := "aéb—c"
	for n := 0; n <= len(s); n++ {
		if out := tailUTF8(s, n); !utf8.ValidString(out) {
			t.Fatalf("tailUTF8(%q, %d) = %q is not valid UTF-8", s, n, out)
		}
	}
}
