package agent_test

import (
	"context"
	"strings"
	"testing"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// runTurn executes one chat turn and drains its chunks.
func runTurn(t *testing.T, engine *agent.AgentEngine, msg string) []agent.StreamChunk {
	t.Helper()
	return collectChunks(t, mustStream(t, engine, msg))
}

// lastUserText returns the text of the last user message in the request.
func lastUserText(msgs []llm.Message) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role != llm.RoleUser {
			continue
		}
		var b strings.Builder
		for _, p := range msgs[i].Parts {
			b.WriteString(p.Text)
		}
		return b.String()
	}
	return ""
}

// deltaProvider streams text in many small events, like a real backend.
type deltaProvider struct{ deltas []string }

func (p *deltaProvider) Name() string { return "delta" }
func (p *deltaProvider) StreamChat(ctx context.Context, req *llm.ChatRequest) (<-chan llm.StreamEvent, error) {
	out := make(chan llm.StreamEvent, len(p.deltas)+1)
	for _, d := range p.deltas {
		out <- llm.StreamEvent{Kind: llm.EventText, Text: d}
	}
	out <- llm.StreamEvent{Kind: llm.EventDone}
	close(out)
	return out, nil
}
func (p *deltaProvider) Generate(ctx context.Context, req *llm.ChatRequest) (string, llm.Usage, error) {
	return "", llm.Usage{}, nil
}
func (p *deltaProvider) GenerateStructured(ctx context.Context, req *llm.ChatRequest, schema llm.Schema, out any) (llm.Usage, error) {
	return llm.Usage{}, nil
}

func TestStreamedTextIsOnePart(t *testing.T) {
	// One text part per iteration, not one per delta: a part per token,
	// replayed newline-joined by the OpenAI adapter, fed the model its own
	// answer with a newline between every token and it imitated the damage.
	provider := &deltaProvider{deltas: []string{"The ", "answer ", "is ", "B."}}
	engine, sessions := newToolEngine(t, provider)
	runTurn(t, engine, "q")

	history := sessions.GetHistory("s1")
	assistant := history[len(history)-1]
	var texts []string
	for _, p := range assistant.Parts {
		if p.Kind == llm.PartText {
			texts = append(texts, p.Text)
		}
	}
	if len(texts) != 1 || texts[0] != "The answer is B." {
		t.Fatalf("assistant text parts = %q, want one joined part", texts)
	}
}
