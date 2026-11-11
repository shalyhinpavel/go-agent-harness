package agent_test

import (
	"context"
	"testing"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// signed is a reasoning block as a signing provider returns it.
func signed(text, sig string) llm.Part {
	return llm.Part{Kind: llm.PartThinking, Text: text, Signature: sig}
}

// thinkingIn lists the reasoning parts a request carries.
func thinkingIn(req llm.ChatRequest) []llm.Part {
	var out []llm.Part
	for _, m := range req.Messages {
		for _, p := range m.Parts {
			if p.Kind == llm.PartThinking {
				out = append(out, p)
			}
		}
	}
	return out
}

func newReasoningEngine(provider llm.LLMProvider, sessions *agent.SessionManager, mut func(*agent.AgentConfig)) *agent.AgentEngine {
	cfg := agent.AgentConfig{Provider: provider, SystemPrompt: "test", PreserveReasoning: true}
	if mut != nil {
		mut(&cfg)
	}
	return agent.NewAgentEngine(cfg, agent.NewToolRegistry(), sessions)
}

func chatOnce(t *testing.T, engine *agent.AgentEngine, sessionID, msg string) {
	t.Helper()
	if _, err := engine.Chat(context.Background(), agent.ChatRequest{UserID: "u", SessionID: sessionID, Message: msg}); err != nil {
		t.Fatal(err)
	}
}

// The reasoning of one turn comes back on the next request, signature intact.
func TestReasoning_BlocksReturnOnTheNextTurn(t *testing.T) {
	provider := llm.NewMockProvider(
		llm.MockTurn{Text: "first", Reasoning: []llm.Part{signed("plan", "S1")}},
		llm.MockTurn{Text: "second"},
	)
	engine := newReasoningEngine(provider, agent.NewSessionManager(), nil)
	chatOnce(t, engine, "s", "one")
	chatOnce(t, engine, "s", "two")

	got := thinkingIn(provider.Requests()[1])
	if len(got) != 1 || got[0] != signed("plan", "S1") {
		t.Fatalf("second request carries %+v, want the signed block S1", got)
	}
}

// The augmenter may change the system prompt between turns. Reasoning signed
// under the old prompt is no longer valid, so it is stripped from the stored
// history; the reasoning made under the new prompt comes back normally.
func TestReasoning_ChangedSystemPromptStripsTheStoredBlocks(t *testing.T) {
	provider := llm.NewMockProvider(
		llm.MockTurn{Text: "a1", Reasoning: []llm.Part{signed("r1", "S1")}},
		llm.MockTurn{Text: "a2", Reasoning: []llm.Part{signed("r2", "S2")}},
		llm.MockTurn{Text: "a3"},
	)
	system := "memory v1"
	engine := newReasoningEngine(provider, agent.NewSessionManager(), func(c *agent.AgentConfig) {
		c.PromptAugmenter = func(_ context.Context, _, _, _ string) (string, error) { return system, nil }
	})
	chatOnce(t, engine, "s", "one")
	system = "memory v2"
	chatOnce(t, engine, "s", "two")
	chatOnce(t, engine, "s", "three")

	reqs := provider.Requests()
	if got := thinkingIn(reqs[1]); len(got) != 0 {
		t.Fatalf("the turn under a new prompt carries %+v; S1 is bound to the old one", got)
	}
	if got := thinkingIn(reqs[2]); len(got) != 1 || got[0].Signature != "S2" {
		t.Fatalf("the next turn carries %+v, want only S2", got)
	}
}

// Stored reasoning survives in the store with its signature and redacted
// payload. After a restart the session has no recorded binding, so its old
// reasoning is dropped once rather than trusted.
func TestReasoning_StoredAndDroppedOnceAfterARestart(t *testing.T) {
	ctx := context.Background()
	store, err := agent.NewFileSessionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	provider := llm.NewMockProvider(
		llm.MockTurn{Text: "a1", Reasoning: []llm.Part{{Kind: llm.PartThinking, Redacted: "ENC"}, signed("r1", "S1")}},
		llm.MockTurn{Text: "a2"},
	)
	before := newReasoningEngine(provider, agent.NewSessionManager(agent.WithSessionStore(store)), nil)
	chatOnce(t, before, "s", "one")

	stored, err := store.Load(ctx, "s")
	if err != nil || stored == nil {
		t.Fatalf("load: %v, %v", stored, err)
	}
	var kept []llm.Part
	for _, m := range stored.History {
		for _, p := range m.Parts {
			if p.Kind == llm.PartThinking {
				kept = append(kept, p)
			}
		}
	}
	if len(kept) != 2 || kept[0].Redacted != "ENC" || kept[1] != signed("r1", "S1") {
		t.Fatalf("stored reasoning = %+v, want the redacted block and S1", kept)
	}

	after := newReasoningEngine(provider, agent.NewSessionManager(agent.WithSessionStore(store)), nil)
	chatOnce(t, after, "s", "two")
	if got := thinkingIn(provider.Requests()[1]); len(got) != 0 {
		t.Fatalf("after a restart the request carries %+v, want none", got)
	}
}
