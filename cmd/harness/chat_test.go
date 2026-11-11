package main

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/llm"
)

func TestTurn_PrintsToolsTextAndBill(t *testing.T) {
	registry := agent.NewToolRegistry()
	if err := registry.Register(&agent.Tool{
		Name:       "lookup",
		Parameters: llm.Schema{"type": "object"},
		Execute: func(context.Context, *agent.ToolContext, map[string]any) (*agent.ToolResult, error) {
			return &agent.ToolResult{Data: "go.dev"}, nil
		},
	}); err != nil {
		t.Fatal(err)
	}
	mock := llm.NewMockProvider(
		llm.MockTurn{ToolCalls: []llm.ToolCall{{ID: "1", Name: "lookup", Args: map[string]any{"q": "go"}}}},
		llm.MockTurn{Text: "Found it.", Usage: &llm.Usage{InputTokens: 1500, OutputTokens: 20}},
	)
	engine := agent.NewAgentEngine(agent.AgentConfig{Provider: mock}, registry, agent.NewSessionManager())

	var buf bytes.Buffer
	p := &printer{w: &buf}
	if err := p.turn(engine, "s", "find go"); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	for _, want := range []string{`→ lookup {"q":"go"}`, "Found it.", "(1.5k in · 20 out"} {
		if !strings.Contains(out, want) {
			t.Errorf("output lacks %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "\x1b[") {
		t.Error("no color codes when the output is not a terminal")
	}
}

func TestTurn_ErrorIsReturned(t *testing.T) {
	mock := llm.NewMockProvider(llm.MockTurn{Err: errors.New("backend down")})
	engine := agent.NewAgentEngine(agent.AgentConfig{Provider: mock}, agent.NewToolRegistry(), agent.NewSessionManager())
	p := &printer{w: &bytes.Buffer{}}
	if err := p.turn(engine, "s", "hi"); err == nil {
		t.Fatal("a failed turn must come back as an error")
	}
}

func TestTurn_ApprovePromptsAndForwardsDenialReason(t *testing.T) {
	var ran int
	registry := agent.NewToolRegistry()
	_ = registry.Register(&agent.Tool{
		Name:       "write_file",
		ReadOnly:   false,
		Parameters: llm.Schema{"type": "object"},
		Execute: func(context.Context, *agent.ToolContext, map[string]any) (*agent.ToolResult, error) {
			ran++
			return &agent.ToolResult{Data: "ok"}, nil
		},
	})
	mock := llm.NewMockProvider(
		llm.MockTurn{ToolCalls: []llm.ToolCall{{ID: "1", Name: "write_file", Args: map[string]any{"path": "draft.md"}}}},
		llm.MockTurn{ToolCalls: []llm.ToolCall{{ID: "2", Name: "write_file", Args: map[string]any{"path": "final.md"}}}},
		llm.MockTurn{Text: "Done."},
	)
	engine := agent.NewAgentEngine(agent.AgentConfig{Provider: mock}, registry, agent.NewSessionManager())

	var buf bytes.Buffer
	p := &printer{
		w:       &buf,
		approve: true,
		in:      readLines(strings.NewReader("use final.md instead\ny\n")),
	}
	if err := p.turn(engine, "s", "write report"); err != nil {
		t.Fatal(err)
	}
	if ran != 1 {
		t.Fatalf("write_file ran %d times, want 1", ran)
	}
	out := buf.String()
	for _, want := range []string{"? allow write_file? [y/N]: ", "✗ write_file: tool call denied by user: use final.md instead", "✓ write_file", "Done."} {
		if !strings.Contains(out, want) {
			t.Errorf("output lacks %q:\n%s", want, out)
		}
	}
}
