package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// A rebinding the store refuses leaves the reasoning in memory: memory never
// runs ahead of what is durable.
func TestBindReasoning_RefusedRewriteKeepsTheReasoning(t *testing.T) {
	ctx := context.Background()
	file, err := NewFileSessionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	sessions := NewSessionManager(WithSessionStore(refusingStore{file}))
	if _, err := sessions.GetOrCreateSession(ctx, "s", "u"); err != nil {
		t.Fatal(err)
	}
	reply := &llm.Message{Role: llm.RoleAssistant, Parts: []llm.Part{
		{Kind: llm.PartThinking, Text: "r", Signature: "S"}, llm.TextPart("a"),
	}}
	if err := sessions.AppendMessage(ctx, "s", reply); err != nil {
		t.Fatal(err)
	}

	err = sessions.BindReasoning(ctx, "s", "new-binding")
	if err == nil || !strings.Contains(err.Error(), "disk full") {
		t.Fatalf("the refused rebinding must be reported, got %v", err)
	}
	if !hasThinking(sessions.GetHistory("s")) {
		t.Fatal("a refused rebinding must keep the reasoning in memory")
	}
}

// A compaction rewrites the prefix every kept block was produced under, so
// the rewritten history holds no reasoning at all.
func TestCompactSession_StripsReasoning(t *testing.T) {
	ctx := context.Background()
	sessions := NewSessionManager()
	sess, err := sessions.GetOrCreateSession(ctx, "s", "u")
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 8; i++ {
		msg := llm.UserText("q")
		if i%2 == 1 {
			msg = &llm.Message{Role: llm.RoleAssistant, Parts: []llm.Part{
				{Kind: llm.PartThinking, Text: "why", Signature: "S"}, llm.TextPart("a"),
			}}
		}
		if err := sessions.AppendMessage(ctx, "s", msg); err != nil {
			t.Fatal(err)
		}
	}
	e := NewAgentEngine(AgentConfig{Compactor: fixedCompactor}, nil, sessions)
	if _, err := e.compactSession(ctx, sess, "t"); err != nil {
		t.Fatal(err)
	}
	if h := sessions.GetHistory("s"); hasThinking(h) {
		t.Fatalf("the rewritten history still carries reasoning: %+v", h)
	}
}
