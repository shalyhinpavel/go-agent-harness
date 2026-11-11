package agent_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// prefill gives session s1 an earlier history. Compaction keeps the six newest
// messages verbatim, so a shorter history has nothing to condense.
func prefill(t *testing.T, sessions *agent.SessionManager, n int) {
	t.Helper()
	ctx := context.Background()
	if _, err := sessions.GetOrCreateSession(ctx, "s1", "u1"); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		msg := llm.UserText("earlier question")
		if i%2 == 1 {
			msg = llm.AssistantText("earlier answer")
		}
		if err := sessions.AppendMessage(ctx, "s1", msg); err != nil {
			t.Fatal(err)
		}
	}
}

// A turn that reaches the context limit is compacted and then answered. The
// summarizer's own spend is part of the turn's accounting, and the next request
// carries the summary in place of the dropped messages.
func TestCompaction_TurnProceedsAtTheContextLimit(t *testing.T) {
	provider := llm.NewMockProvider(
		llm.MockTurn{Text: "first answer", Usage: &llm.Usage{InputTokens: 950, OutputTokens: 50}},
		llm.MockTurn{Text: "summary text", Usage: &llm.Usage{InputTokens: 10, OutputTokens: 10}},
		llm.MockTurn{Text: "second answer", Usage: &llm.Usage{InputTokens: 20, OutputTokens: 5}},
	)
	engine, _, sessions := newGuardEngine(t, provider, func(c *agent.AgentConfig) {
		c.ContextLimit = 1000
		c.MaxTokens = 100
		c.AutoCompact = true
	})
	prefill(t, sessions, 6)
	ctx := context.Background()
	if _, err := engine.Chat(ctx, agent.ChatRequest{UserID: "u1", SessionID: "s1", Message: "one"}); err != nil {
		t.Fatal(err)
	}

	stream, err := engine.StreamChat(ctx, agent.ChatRequest{UserID: "u1", SessionID: "s1", Message: "two"})
	if err != nil {
		t.Fatal(err)
	}
	var done *agent.StreamChunk
	for _, c := range collectChunks(t, stream) {
		if c.Type == agent.ChunkTypeDone {
			done = &c
		}
	}
	if done == nil {
		t.Fatal("the turn after compaction must complete")
	}
	// Turn accounting: summarizer 10/10 plus the answer 20/5.
	if done.Usage == nil || done.Usage.InputTokens != 30 || done.Usage.OutputTokens != 15 {
		t.Fatalf("turn usage = %+v, want input 30 output 15 (summarizer included)", done.Usage)
	}
	if total := sessions.SessionUsage("s1"); total.InputTokens != 980 {
		t.Fatalf("session input = %d, want 980 (950 + 10 + 20)", total.InputTokens)
	}

	reqs := provider.Requests()
	if len(reqs) != 3 {
		t.Fatalf("provider calls = %d, want 3 (answer, summary, answer)", len(reqs))
	}
	if first := reqs[2].Messages[0]; !strings.Contains(first.Parts[0].Text, "summary text") {
		t.Fatalf("the answer must start from the summary, got %+v", first)
	}
}

// A failing summarizer ends the turn with the cause named, leaves the history
// alone, and still books what the failed call spent.
func TestCompaction_FailureNamesTheCauseAndKeepsItsSpend(t *testing.T) {
	provider := llm.NewMockProvider(
		llm.MockTurn{Text: "first answer", Usage: &llm.Usage{InputTokens: 950, OutputTokens: 50}},
		llm.MockTurn{Text: "never reached"},
	)
	engine, _, sessions := newGuardEngine(t, provider, func(c *agent.AgentConfig) {
		c.ContextLimit = 1000
		c.MaxTokens = 100
		c.Compactor = func(context.Context, []llm.Message) (string, llm.Usage, error) {
			return "", llm.Usage{InputTokens: 5}, errors.New("summarizer down")
		}
	})
	prefill(t, sessions, 6)
	ctx := context.Background()
	if _, err := engine.Chat(ctx, agent.ChatRequest{UserID: "u1", SessionID: "s1", Message: "one"}); err != nil {
		t.Fatal(err)
	}

	_, err := engine.Chat(ctx, agent.ChatRequest{UserID: "u1", SessionID: "s1", Message: "two"})
	if err == nil || !strings.Contains(err.Error(), "compaction failed") || !strings.Contains(err.Error(), "summarizer down") {
		t.Fatalf("the error must name the compaction failure, got %v", err)
	}
	if n := len(provider.Requests()); n != 1 {
		t.Fatalf("provider calls = %d, want 1: the answer must not be requested", n)
	}
	// The refused question is not stored: a turn the guard refuses leaves the
	// history as it was.
	if h := sessions.GetHistory("s1"); len(h) != 8 {
		t.Fatalf("history = %d messages, want 8 (unchanged: 6 prefilled + 2 turns)", len(h))
	}
	if total := sessions.SessionUsage("s1"); total.InputTokens != 955 {
		t.Fatalf("session input = %d, want 955: the failed summarizer call is still spend", total.InputTokens)
	}
}

// AutoCompact off means the guard refuses instead of compacting.
func TestCompaction_OffMeansTheGuardRefuses(t *testing.T) {
	provider := llm.NewMockProvider(
		llm.MockTurn{Text: "first answer", Usage: &llm.Usage{InputTokens: 950, OutputTokens: 50}},
		llm.MockTurn{Text: "never reached"},
	)
	engine, _, sessions := newGuardEngine(t, provider, func(c *agent.AgentConfig) {
		c.ContextLimit = 1000
		c.MaxTokens = 100
	})
	prefill(t, sessions, 6)
	ctx := context.Background()
	if _, err := engine.Chat(ctx, agent.ChatRequest{UserID: "u1", SessionID: "s1", Message: "one"}); err != nil {
		t.Fatal(err)
	}
	_, err := engine.Chat(ctx, agent.ChatRequest{UserID: "u1", SessionID: "s1", Message: "two"})
	if err == nil || !strings.Contains(err.Error(), "context limit exceeded") || strings.Contains(err.Error(), "compaction failed") {
		t.Fatalf("without compaction the guard must refuse, got %v", err)
	}
}
