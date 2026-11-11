package agent_test

import (
	"context"
	"strings"
	"testing"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// newToolEngine wires an engine with a single "lookup" tool.
func newToolEngine(t *testing.T, provider llm.LLMProvider) (*agent.AgentEngine, *agent.SessionManager) {
	t.Helper()
	registry := agent.NewToolRegistry()
	if err := registry.Register(&agent.Tool{
		Name:       "lookup",
		Parameters: llm.Schema{"type": "object"},
		Execute: func(ctx context.Context, tc *agent.ToolContext, args map[string]any) (*agent.ToolResult, error) {
			return &agent.ToolResult{Data: map[string]any{"verdict": "B"}}, nil
		},
	}); err != nil {
		t.Fatalf("register: %v", err)
	}
	sessions := agent.NewSessionManager()
	engine := agent.NewAgentEngine(agent.AgentConfig{
		Provider:     provider,
		SystemPrompt: "You have access to tools. Use a tool whenever the task needs computation.",
	}, registry, sessions)
	return engine, sessions
}

func toolCall() llm.MockTurn {
	return llm.MockTurn{ToolCalls: []llm.ToolCall{{ID: "c1", Name: "lookup", Args: map[string]any{"q": "x"}}}}
}

func finalText(chunks []agent.StreamChunk) string {
	var b strings.Builder
	for _, c := range chunks {
		if c.Type == agent.ChunkTypeText {
			b.WriteString(c.Content)
		}
	}
	return b.String()
}

// A turn that ends on reasoning alone gets one nudge for the answer.
func TestThinkingOnlyTurnGetsOneNudge(t *testing.T) {
	provider := llm.NewMockProvider(
		toolCall(),
		llm.MockTurn{Thinking: "Let me verify with Python computation"}, // no text, no call
		llm.MockTurn{Text: "Answer: B"},
	)
	engine, _ := newToolEngine(t, provider)
	chunks := runTurn(t, engine, "q")

	reqs := provider.Requests()
	if len(reqs) != 3 {
		t.Fatalf("expected 3 model calls (tool, empty, nudged), got %d", len(reqs))
	}
	if note := lastUserText(reqs[2].Messages); !strings.Contains(note, "final answer") {
		t.Fatalf("the retry must carry the nudge, last message: %q", note)
	}
	if got := finalText(chunks); got != "Answer: B" {
		t.Fatalf("final text = %q, want the nudged answer", got)
	}
	last := chunks[len(chunks)-1]
	if last.Type != agent.ChunkTypeDone {
		t.Fatalf("last chunk = %+v, want done", last)
	}
}

// One nudge, not a loop: a model that stays silent ends the turn as before
// (an honest empty answer, scored as no_answer by the eval).
func TestThinkingOnlyTurnNudgedOnlyOnce(t *testing.T) {
	provider := llm.NewMockProvider(
		toolCall(),
		llm.MockTurn{Thinking: "still thinking"},
	)
	engine, _ := newToolEngine(t, provider)
	chunks := runTurn(t, engine, "q")

	if n := len(provider.Requests()); n != 3 {
		t.Fatalf("expected 3 model calls (tool, empty, one nudged retry), got %d", n)
	}
	if got := finalText(chunks); got != "" {
		t.Fatalf("a silent model must stay silent, got %q", got)
	}
	if last := chunks[len(chunks)-1]; last.Type != agent.ChunkTypeDone {
		t.Fatalf("last chunk = %+v, want done", last)
	}
}

// A max_tokens cut is not a silent stop: the answer is truncated, and a
// nudge would only spend more tokens on a model that has none left.
func TestThinkingOnlyMaxTokensIsNotNudged(t *testing.T) {
	provider := llm.NewMockProvider(
		toolCall(),
		llm.MockTurn{Thinking: "long thought", StopReason: llm.StopMaxTokens},
	)
	engine, _ := newToolEngine(t, provider)
	runTurn(t, engine, "q")

	if n := len(provider.Requests()); n != 2 {
		t.Fatalf("a max_tokens stop must not be nudged: %d model calls", n)
	}
}
