package agent

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"testing"

	"github.com/shalyhinpavel/go-agent-harness/llm"
)

func testContext() *ToolContext {
	return &ToolContext{UserID: "u1", SessionID: "s1"}
}

func TestExecuteToolsParallel_Single(t *testing.T) {
	registry := NewToolRegistry()
	if err := registry.Register(&Tool{
		Name: "echo",
		Execute: func(ctx context.Context, tc *ToolContext, args map[string]any) (*ToolResult, error) {
			return &ToolResult{Data: "pong"}, nil
		},
	}); err != nil {
		t.Fatal(err)
	}
	e := &AgentEngine{registry: registry}

	parts, _, err := e.executeToolsParallel(context.Background(), testContext(), HookContext{}, nil,
		[]llm.ToolCall{{ID: "c1", Name: "echo"}}, nil)
	if err != nil {
		t.Fatal(err)
	}

	if len(parts) != 1 {
		t.Fatalf("parts = %d, want 1", len(parts))
	}
	res := parts[0].ToolResult
	if res == nil || res.IsError || res.Content != "pong" {
		t.Fatalf("unexpected result: %+v", parts[0])
	}
	if res.CallID != "c1" {
		t.Fatalf("call_id = %q, want c1", res.CallID)
	}
}

func TestExecuteToolsParallel_Multiple(t *testing.T) {
	registry := NewToolRegistry()
	for _, name := range []string{"a", "b", "c"} {
		n := name
		if err := registry.Register(&Tool{
			Name: n,
			Execute: func(ctx context.Context, tc *ToolContext, args map[string]any) (*ToolResult, error) {
				return &ToolResult{Data: n}, nil
			},
		}); err != nil {
			t.Fatal(err)
		}
	}
	e := &AgentEngine{registry: registry}

	parts, _, err := e.executeToolsParallel(context.Background(), testContext(), HookContext{}, nil,
		[]llm.ToolCall{
			{ID: "1", Name: "a"}, {ID: "2", Name: "b"}, {ID: "3", Name: "c"},
		}, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Results must stay in input order.
	for i, want := range []string{"a", "b", "c"} {
		if got := parts[i].ToolResult.Content; got != want {
			t.Fatalf("parts[%d] = %v, want %s", i, got, want)
		}
	}
}

func TestExecuteToolsParallel_ToolNotFound(t *testing.T) {
	e := &AgentEngine{registry: NewToolRegistry()}

	parts, _, _ := e.executeToolsParallel(context.Background(), testContext(), HookContext{}, nil,
		[]llm.ToolCall{{ID: "c1", Name: "ghost"}}, nil)

	if !parts[0].ToolResult.IsError {
		t.Fatal("missing tool must yield error result")
	}
	if !strings.Contains(parts[0].ToolResult.Content.(map[string]any)["error"].(string), "ghost") {
		t.Fatalf("error should name the tool: %+v", parts[0])
	}
}

func TestExecuteToolsParallel_ExecutionError(t *testing.T) {
	registry := NewToolRegistry()
	if err := registry.Register(&Tool{
		Name: "boom",
		Execute: func(ctx context.Context, tc *ToolContext, args map[string]any) (*ToolResult, error) {
			return nil, errors.New("exploded")
		},
	}); err != nil {
		t.Fatal(err)
	}
	e := &AgentEngine{registry: registry}

	parts, _, _ := e.executeToolsParallel(context.Background(), testContext(), HookContext{}, nil,
		[]llm.ToolCall{{ID: "c1", Name: "boom"}}, nil)

	if !parts[0].ToolResult.IsError {
		t.Fatal("failed execution must yield error result")
	}
}

func TestExecuteToolsParallel_NilPartNever(t *testing.T) {
	// Even on a cancelled context every call must produce a non-nil part.
	registry := NewToolRegistry()
	if err := registry.Register(&Tool{
		Name: "slow",
		Execute: func(ctx context.Context, tc *ToolContext, args map[string]any) (*ToolResult, error) {
			<-ctx.Done()
			return &ToolResult{Data: "late"}, nil
		},
	}); err != nil {
		t.Fatal(err)
	}
	e := &AgentEngine{registry: registry}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	parts, _, _ := e.executeToolsParallel(ctx, testContext(), HookContext{}, nil,
		[]llm.ToolCall{{ID: "c1", Name: "slow"}}, nil)

	if parts[0].ToolResult == nil {
		t.Fatal("nil part on cancellation — regression of the nil-part bug")
	}
}

func TestExecuteToolsParallel_ChunksToStream(t *testing.T) {
	registry := NewToolRegistry()
	if err := registry.Register(&Tool{
		Name: "echo",
		Execute: func(ctx context.Context, tc *ToolContext, args map[string]any) (*ToolResult, error) {
			return &ToolResult{Data: "x"}, nil
		},
	}); err != nil {
		t.Fatal(err)
	}
	e := &AgentEngine{registry: registry}

	out := make(chan StreamChunk, 8)
	parts, _, _ := e.executeToolsParallel(context.Background(), testContext(), HookContext{}, nil,
		[]llm.ToolCall{{ID: "c1", Name: "echo"}}, out)
	close(out)

	if len(parts) != 1 {
		t.Fatalf("parts = %d", len(parts))
	}
	var calls, results int
	for c := range out {
		switch c.Type {
		case ChunkTypeToolCall:
			calls++
		case ChunkTypeToolResult:
			results++
		}
	}
	if calls != 1 || results != 1 {
		t.Fatalf("stream chunks: calls=%d results=%d, want 1/1", calls, results)
	}
}

func TestToolPanicBecomesAnErrorResult(t *testing.T) {
	registry := NewToolRegistry()
	if err := registry.Register(&Tool{
		Name:       "boom",
		Parameters: llm.Schema{"type": "object"},
		Execute: func(context.Context, *ToolContext, map[string]any) (*ToolResult, error) {
			var m map[string]int
			m["x"] = 1 // nil map write
			return nil, nil
		},
	}); err != nil {
		t.Fatal(err)
	}
	mock := llm.NewMockProvider(
		llm.MockTurn{ToolCalls: []llm.ToolCall{{ID: "1", Name: "boom"}}},
		llm.MockTurn{Text: "recovered"},
	)
	engine := NewAgentEngine(AgentConfig{Provider: mock}, registry, NewSessionManager())
	text, err := engine.Chat(context.Background(), ChatRequest{UserID: "u", SessionID: "s", Message: "go"})
	if err != nil || text != "recovered" {
		t.Fatalf("a panicking tool must fail its call, not the turn: %q %v", text, err)
	}
	reqs := mock.Requests()
	last := reqs[len(reqs)-1].Messages
	res := last[len(last)-1].Parts[0].ToolResult
	if !res.IsError || !strings.Contains(fmt.Sprint(res.Content), "panicked") {
		t.Fatalf("the model must see the failure: %+v", res)
	}
}

func TestAgentConfigLoggerReceivesDiagnostics(t *testing.T) {
	registry := NewToolRegistry()
	_ = registry.Register(&Tool{
		Name: "boom", Parameters: llm.Schema{"type": "object"},
		Execute: func(context.Context, *ToolContext, map[string]any) (*ToolResult, error) { panic("bang") },
	})
	var buf strings.Builder
	logger := slog.New(slog.NewTextHandler(&buf, nil))
	mock := llm.NewMockProvider(
		llm.MockTurn{ToolCalls: []llm.ToolCall{{ID: "1", Name: "boom"}}},
		llm.MockTurn{Text: "ok"},
	)
	engine := NewAgentEngine(AgentConfig{Provider: mock, Logger: logger}, registry, NewSessionManager())
	if _, err := engine.Chat(context.Background(), ChatRequest{UserID: "u", SessionID: "s", Message: "go"}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "tool panicked") || !strings.Contains(buf.String(), "bang") {
		t.Fatalf("the configured logger saw nothing: %q", buf.String())
	}
}
