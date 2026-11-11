package agent_test

import (
	"context"
	"errors"
	"testing"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// A turn refused by the context guard stores nothing: the user message is
// written only once the turn is admitted, so retries do not pile up unanswered
// user messages in the history.
func TestGuard_RefusedTurnStoresNothing(t *testing.T) {
	provider := llm.NewMockProvider(
		llm.MockTurn{Text: "first answer", Usage: &llm.Usage{InputTokens: 950, OutputTokens: 50}},
		llm.MockTurn{Text: "never reached"},
	)
	engine, _, sessions := newGuardEngine(t, provider, func(c *agent.AgentConfig) {
		c.ContextLimit = 1000
		c.MaxTokens = 100
	})
	ctx := context.Background()
	if _, err := engine.Chat(ctx, agent.ChatRequest{UserID: "u1", SessionID: "s1", Message: "one"}); err != nil {
		t.Fatal(err)
	}
	before := len(sessions.GetHistory("s1"))

	for i := 0; i < 2; i++ {
		if _, err := engine.Chat(ctx, agent.ChatRequest{UserID: "u1", SessionID: "s1", Message: "two"}); err == nil {
			t.Fatal("the refused turn must report the context limit")
		}
	}
	if after := len(sessions.GetHistory("s1")); after != before {
		t.Fatalf("history = %d messages after refused turns, want %d", after, before)
	}
}

// A turn the caller halts through Proceed before its first call stores nothing
// either.
func TestProceed_HaltedTurnStoresNothing(t *testing.T) {
	provider := llm.NewMockProvider(llm.MockTurn{Text: "never reached"})
	engine, _, sessions := newGuardEngine(t, provider, nil)
	_, err := engine.Chat(context.Background(), agent.ChatRequest{
		UserID: "u1", SessionID: "s1", Message: "hi",
		Proceed: func() error { return errors.New("halted by the caller") },
	})
	if err == nil {
		t.Fatal("the halted turn must report the halt")
	}
	if n := len(sessions.GetHistory("s1")); n != 0 {
		t.Fatalf("history = %d messages after a halted turn, want 0", n)
	}
	if n := len(provider.Requests()); n != 0 {
		t.Fatalf("provider calls = %d after a halted turn, want 0", n)
	}
}
