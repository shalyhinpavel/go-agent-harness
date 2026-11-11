package agent_test

import (
	"context"
	"testing"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// A reply that is reasoning and nothing else is silence even when its thinking
// text is not visible (display omitted, the Opus default): the model is nudged
// once to answer.
func TestReasoningOnlyReply_IsNudgedToAnswer(t *testing.T) {
	provider := llm.NewMockProvider(
		llm.MockTurn{Reasoning: []llm.Part{signed("plan", "S1")}},
		llm.MockTurn{Text: "answer"},
	)
	engine := newReasoningEngine(provider, agent.NewSessionManager(), nil)
	got, err := engine.Chat(context.Background(), agent.ChatRequest{UserID: "u", SessionID: "s", Message: "q"})
	if err != nil {
		t.Fatal(err)
	}
	if got != "answer" {
		t.Fatalf("reply = %q, want the answer given after the nudge", got)
	}
	if n := len(provider.Requests()); n != 2 {
		t.Fatalf("provider calls = %d, want 2 (the silent reply, then the nudged one)", n)
	}
}

// A second reasoning-only reply ends the turn without an answer, and its
// reasoning is not kept: a block with no answer behind it must not go back to
// the model as if it were one.
func TestReasoningOnlyFinalReply_IsNotKept(t *testing.T) {
	provider := llm.NewMockProvider(
		llm.MockTurn{Reasoning: []llm.Part{signed("plan", "S1")}},
		llm.MockTurn{Reasoning: []llm.Part{signed("again", "S2")}},
	)
	sessions := agent.NewSessionManager()
	engine := newReasoningEngine(provider, sessions, nil)
	if _, err := engine.Chat(context.Background(), agent.ChatRequest{UserID: "u", SessionID: "s", Message: "q"}); err != nil {
		t.Fatal(err)
	}
	if hasReasoning(sessions.GetHistory("s")) {
		t.Fatalf("the reasoning-only reply was kept: %+v", sessions.GetHistory("s"))
	}
}

func hasReasoning(msgs []llm.Message) bool {
	for _, m := range msgs {
		for _, p := range m.Parts {
			if p.Kind == llm.PartThinking {
				return true
			}
		}
	}
	return false
}
