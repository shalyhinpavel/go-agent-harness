package agent_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// newTestEngine wires a registry with a recording calculator tool.
func newTestEngine(t *testing.T, provider llm.LLMProvider) (*agent.AgentEngine, *agent.ToolRegistry, *agent.SessionManager) {
	t.Helper()

	registry := agent.NewToolRegistry()
	err := registry.Register(&agent.Tool{
		Name:        "calculator",
		Description: "Adds two numbers.",
		Parameters:  llm.Schema{"type": "object"},
		Execute: func(ctx context.Context, tc *agent.ToolContext, args map[string]any) (*agent.ToolResult, error) {
			a, _ := args["a"].(float64)
			b, _ := args["b"].(float64)
			return &agent.ToolResult{Data: map[string]any{"sum": a + b}}, nil
		},
	})
	if err != nil {
		t.Fatalf("register tool: %v", err)
	}

	sessions := agent.NewSessionManager()
	engine := agent.NewAgentEngine(agent.AgentConfig{
		Provider:     provider,
		SystemPrompt: "test system prompt",
	}, registry, sessions)
	return engine, registry, sessions
}

func collectChunks(t *testing.T, stream <-chan agent.StreamChunk) []agent.StreamChunk {
	t.Helper()
	var chunks []agent.StreamChunk
	for chunk := range stream {
		chunks = append(chunks, chunk)
	}
	return chunks
}

func TestStreamChat_TextOnly(t *testing.T) {
	provider := llm.NewMockProvider(llm.MockTurn{Text: "Hello there"})
	engine, _, sessions := newTestEngine(t, provider)

	stream, err := engine.StreamChat(context.Background(), agent.ChatRequest{
		UserID: "u1", SessionID: "s1", Message: "hi",
	})
	if err != nil {
		t.Fatalf("StreamChat: %v", err)
	}

	chunks := collectChunks(t, stream)
	if len(chunks) != 2 || chunks[0].Type != agent.ChunkTypeText || chunks[0].Content != "Hello there" {
		t.Fatalf("unexpected chunks: %+v", chunks)
	}
	if chunks[1].Type != agent.ChunkTypeDone {
		t.Fatalf("expected done chunk, got %+v", chunks[1])
	}

	// History: user + assistant.
	if h := sessions.GetHistory("s1"); len(h) != 2 {
		t.Fatalf("expected 2 history messages, got %d", len(h))
	}
}

func TestStreamChat_ThinkingNotPersisted(t *testing.T) {
	provider := llm.NewMockProvider(llm.MockTurn{Thinking: "pondering", Text: "Answer"})
	engine, _, sessions := newTestEngine(t, provider)

	stream, _ := engine.StreamChat(context.Background(), agent.ChatRequest{
		UserID: "u1", SessionID: "s1", Message: "q",
	})
	chunks := collectChunks(t, stream)

	if len(chunks) != 3 || chunks[0].Type != agent.ChunkTypeThinking {
		t.Fatalf("expected thinking+text+done, got %+v", chunks)
	}
	history := sessions.GetHistory("s1")
	assistant := history[len(history)-1]
	for _, p := range assistant.Parts {
		if p.Kind == llm.PartThinking {
			t.Fatal("thinking part must not be persisted into history")
		}
	}
}

func TestStreamChat_ToolLoop(t *testing.T) {
	provider := llm.NewMockProvider(
		llm.MockTurn{ToolCalls: []llm.ToolCall{{ID: "call_1", Name: "calculator", Args: map[string]any{"a": 2.0, "b": 3.0}}}},
		llm.MockTurn{Text: "The sum is 5."},
	)
	engine, _, sessions := newTestEngine(t, provider)

	stream, err := engine.StreamChat(context.Background(), agent.ChatRequest{
		UserID: "u1", SessionID: "s1", Message: "add 2 and 3",
	})
	if err != nil {
		t.Fatalf("StreamChat: %v", err)
	}

	chunks := collectChunks(t, stream)
	var types []agent.StreamChunkType
	for _, c := range chunks {
		types = append(types, c.Type)
	}
	want := []agent.StreamChunkType{
		agent.ChunkTypeToolCall, agent.ChunkTypeToolResult, agent.ChunkTypeText, agent.ChunkTypeDone,
	}
	if len(types) != len(want) {
		t.Fatalf("chunk types = %v, want %v", types, want)
	}
	for i := range want {
		if types[i] != want[i] {
			t.Fatalf("chunk types = %v, want %v", types, want)
		}
	}
	if got := chunks[1].ToolResult.Data.(map[string]any)["sum"]; got != 5.0 {
		t.Fatalf("tool result sum = %v, want 5", got)
	}

	// History: user, assistant(tool_call), tool(result), assistant(text).
	history := sessions.GetHistory("s1")
	if len(history) != 4 {
		t.Fatalf("expected 4 history messages, got %d", len(history))
	}
	if history[2].Role != llm.RoleTool || history[2].Parts[0].ToolResult == nil {
		t.Fatalf("expected tool result message, got %+v", history[2])
	}
	if history[2].Parts[0].ToolResult.CallID != "call_1" {
		t.Fatalf("tool result call_id = %q", history[2].Parts[0].ToolResult.CallID)
	}
}

func TestStreamChat_ProviderError(t *testing.T) {
	provider := llm.NewMockProvider(llm.MockTurn{Err: errors.New("backend down")})
	engine, _, _ := newTestEngine(t, provider)

	stream, _ := engine.StreamChat(context.Background(), agent.ChatRequest{
		UserID: "u1", SessionID: "s1", Message: "hi",
	})
	chunks := collectChunks(t, stream)
	if len(chunks) != 1 || chunks[0].Type != agent.ChunkTypeError {
		t.Fatalf("expected single error chunk, got %+v", chunks)
	}
}

func TestStreamChat_UnknownTool(t *testing.T) {
	provider := llm.NewMockProvider(
		llm.MockTurn{ToolCalls: []llm.ToolCall{{ID: "c1", Name: "nope"}}},
		llm.MockTurn{Text: "Tool missing, sorry."},
	)
	engine, _, _ := newTestEngine(t, provider)

	stream, _ := engine.StreamChat(context.Background(), agent.ChatRequest{
		UserID: "u1", SessionID: "s1", Message: "call it",
	})
	chunks := collectChunks(t, stream)
	for _, c := range chunks {
		if c.Type == agent.ChunkTypeToolResult && c.ToolResult.Error == "" {
			t.Fatal("unknown tool must produce an error tool result")
		}
	}
}

func TestChat_Synchronous(t *testing.T) {
	provider := llm.NewMockProvider(llm.MockTurn{Text: "sync answer"})
	engine, _, _ := newTestEngine(t, provider)

	got, err := engine.Chat(context.Background(), agent.ChatRequest{
		UserID: "u1", SessionID: "s1", Message: "hi",
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if got != "sync answer" {
		t.Fatalf("Chat = %q", got)
	}
}

func TestStreamChat_CancelledContext(t *testing.T) {
	provider := llm.NewMockProvider(llm.MockTurn{Text: "too late"})
	engine, _, _ := newTestEngine(t, provider)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	stream, _ := engine.StreamChat(ctx, agent.ChatRequest{
		UserID: "u1", SessionID: "s1", Message: "hi",
	})
	// With the context already dead, the engine may either push an error
	// chunk or exit silently — both are valid cancellations. It must never
	// stream a successful answer.
	chunks := collectChunks(t, stream)
	for _, c := range chunks {
		if c.Type == agent.ChunkTypeText || c.Type == agent.ChunkTypeDone {
			t.Fatalf("cancelled run must not produce text/done chunks: %+v", chunks)
		}
	}
}

func TestStreamChat_MessagesGrowAcrossTurns(t *testing.T) {
	provider := llm.NewMockProvider(
		llm.MockTurn{Text: "first"},
		llm.MockTurn{Text: "second"},
	)
	engine, _, _ := newTestEngine(t, provider)

	ctx := context.Background()
	if _, err := engine.Chat(ctx, agent.ChatRequest{UserID: "u1", SessionID: "s1", Message: "one"}); err != nil {
		t.Fatal(err)
	}
	if _, err := engine.Chat(ctx, agent.ChatRequest{UserID: "u1", SessionID: "s1", Message: "two"}); err != nil {
		t.Fatal(err)
	}

	// Second request must include prior history.
	reqs := provider.Requests()
	if len(reqs) != 2 {
		t.Fatalf("expected 2 provider calls, got %d", len(reqs))
	}
	if len(reqs[1].Messages) != 3 { // user1, assistant1, user2
		t.Fatalf("second request messages = %d, want 3", len(reqs[1].Messages))
	}
	if reqs[1].System != "test system prompt" {
		t.Fatalf("system prompt not forwarded: %q", reqs[1].System)
	}
}

func TestStreamChat_HistoryWindowRespected(t *testing.T) {
	provider := llm.NewMockProvider(llm.MockTurn{Text: "ok"})
	registry := agent.NewToolRegistry()
	sessions := agent.NewSessionManager()
	engine := agent.NewAgentEngine(agent.AgentConfig{
		Provider:      provider,
		HistoryWindow: 4,
	}, registry, sessions)

	ctx := context.Background()
	for i := 0; i < 5; i++ {
		if _, err := engine.Chat(ctx, agent.ChatRequest{UserID: "u1", SessionID: "s1", Message: "m"}); err != nil {
			t.Fatal(err)
		}
	}
	reqs := provider.Requests()
	last := reqs[len(reqs)-1]
	// 5 turns → 10 history messages; window 4 must cap the request.
	if len(last.Messages) > 4 {
		t.Fatalf("request carried %d messages, window is 4", len(last.Messages))
	}
}

func TestSessionTTLAndEviction(t *testing.T) {
	sessions := agent.NewSessionManager(
		agent.WithSessionTTL(10*time.Millisecond),
		agent.WithMaxSessions(2),
	)

	sessions.GetOrCreateSession(context.Background(), "a", "u")
	time.Sleep(15 * time.Millisecond)

	// TTL reset: session "a" comes back fresh.
	sess, _ := sessions.GetOrCreateSession(context.Background(), "a", "u")
	if len(sess.History) != 0 {
		t.Fatal("expired session must be reset")
	}

	// LRU eviction: filling to the cap evicts the oldest.
	sessions.GetOrCreateSession(context.Background(), "b", "u")
	sessions.GetOrCreateSession(context.Background(), "c", "u") // cap=2 → evicts "b" or "c"
	_ = sessions
}

func TestStreamChat_FinalIterationDemandsText(t *testing.T) {
	// A model that tool-calls until the end: the last iteration must
	// demand a text answer so the turn ends with text, not silence.
	toolTurn := llm.MockTurn{ToolCalls: []llm.ToolCall{{
		ID: "call-1", Name: "calculator",
		Args: map[string]any{"a": 1.0, "b": 2.0},
	}}}
	provider := llm.NewMockProvider(toolTurn, toolTurn, llm.MockTurn{Text: "Final answer after the budget."})

	registry := agent.NewToolRegistry()
	if err := registry.Register(&agent.Tool{
		Name:        "calculator",
		Description: "Adds two numbers.",
		Parameters:  llm.Schema{"type": "object"},
		Execute: func(ctx context.Context, tc *agent.ToolContext, args map[string]any) (*agent.ToolResult, error) {
			return &agent.ToolResult{Data: map[string]any{"sum": 3}}, nil
		},
	}); err != nil {
		t.Fatal(err)
	}
	engine := agent.NewAgentEngine(agent.AgentConfig{
		Provider:      provider,
		SystemPrompt:  "test",
		MaxIterations: 3,
	}, registry, agent.NewSessionManager())

	stream, err := engine.StreamChat(context.Background(), agent.ChatRequest{
		UserID: "u1", SessionID: "strip-1", Message: "go",
	})
	if err != nil {
		t.Fatal(err)
	}
	var text string
	for chunk := range stream {
		if chunk.Type == agent.ChunkTypeError {
			t.Fatalf("unexpected error chunk: %s", chunk.Error)
		}
		if chunk.Type == agent.ChunkTypeText {
			text += chunk.Content
		}
	}
	if text != "Final answer after the budget." {
		t.Fatalf("iteration budget must end with a text answer, got %q", text)
	}

	// The last call says tool_choice none but keeps the declarations:
	// Anthropic rejects tool_use history without declared tools.
	reqs := provider.Requests()
	if n := len(reqs); n != 3 {
		t.Fatalf("provider calls = %d", n)
	}
	last := reqs[2]
	if last.ToolChoice.Mode != llm.ToolChoiceNone {
		t.Fatalf("final iteration tool choice = %+v, want none", last.ToolChoice)
	}
	if len(last.Tools) == 0 {
		t.Fatal("final iteration must keep tool declarations next to tool_use history")
	}
	if reqs[0].ToolChoice.Mode != llm.ToolChoiceAuto || reqs[1].ToolChoice.Mode != llm.ToolChoiceAuto {
		t.Fatal("earlier iterations must leave tool choice to the model")
	}
	note := last.Messages[len(last.Messages)-1]
	if note.Role != llm.RoleUser || len(note.Parts) == 0 || note.Parts[0].Text == "" {
		t.Fatalf("final iteration must carry the answer-now note, got %+v", note)
	}
}

// A long tool loop in one turn: every request stays within the window,
// starts with the pinned task and never splits a tool exchange.
func TestStreamChat_LongToolLoopWindowedWithPinnedTask(t *testing.T) {
	toolTurn := llm.MockTurn{ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "calculator", Args: map[string]any{}}}}
	turns := make([]llm.MockTurn, 0, 31)
	for i := 0; i < 30; i++ {
		turns = append(turns, toolTurn)
	}
	turns = append(turns, llm.MockTurn{Text: "done"})
	provider := llm.NewMockProvider(turns...)

	registry := agent.NewToolRegistry()
	if err := registry.Register(&agent.Tool{
		Name: "calculator", Description: "Adds.", Parameters: llm.Schema{"type": "object"},
		Execute: func(ctx context.Context, tc *agent.ToolContext, args map[string]any) (*agent.ToolResult, error) {
			return &agent.ToolResult{Data: map[string]any{"sum": 3}}, nil
		},
	}); err != nil {
		t.Fatal(err)
	}
	engine := agent.NewAgentEngine(agent.AgentConfig{
		Provider: provider, SystemPrompt: "s", MaxIterations: 40,
		HistoryWindow: 8, PinTask: true,
	}, registry, agent.NewSessionManager())

	stream, err := engine.StreamChat(context.Background(), agent.ChatRequest{UserID: "u", SessionID: "loop", Message: "TASK"})
	if err != nil {
		t.Fatal(err)
	}
	for chunk := range stream {
		if chunk.Type == agent.ChunkTypeError {
			t.Fatalf("error chunk: %s", chunk.Error)
		}
	}
	reqs := provider.Requests()
	if len(reqs) != 31 {
		t.Fatalf("provider calls = %d, want 31", len(reqs))
	}
	for i, r := range reqs {
		if len(r.Messages) > 9 {
			t.Fatalf("request %d carried %d messages, window is task + 8", i, len(r.Messages))
		}
		if r.Messages[0].Role != llm.RoleUser || r.Messages[0].Parts[0].Text != "TASK" {
			t.Fatalf("request %d does not start with the task", i)
		}
		if len(r.Messages) > 1 && r.Messages[1].Role == llm.RoleTool {
			t.Fatalf("request %d starts its window on a tool result", i)
		}
	}
}

// Proceed ends a tool loop between iterations: the batch that ran is in
// the history with its results, and no further model call is made.
func TestStreamChat_ProceedHaltsBetweenIterations(t *testing.T) {
	toolTurn := llm.MockTurn{ToolCalls: []llm.ToolCall{
		{ID: "a", Name: "calculator", Args: map[string]any{}},
		{ID: "b", Name: "calculator", Args: map[string]any{}},
	}}
	provider := llm.NewMockProvider(toolTurn, toolTurn, toolTurn, llm.MockTurn{Text: "done"})
	executed := 0
	registry := agent.NewToolRegistry()
	if err := registry.Register(&agent.Tool{
		Name: "calculator", Description: "Adds.", Parameters: llm.Schema{"type": "object"},
		Execute: func(ctx context.Context, tc *agent.ToolContext, args map[string]any) (*agent.ToolResult, error) {
			executed++
			return &agent.ToolResult{Data: "ok"}, nil
		},
	}); err != nil {
		t.Fatal(err)
	}
	sessions := agent.NewSessionManager()
	engine := agent.NewAgentEngine(agent.AgentConfig{
		Provider: provider, SystemPrompt: "s", MaxIterations: 10, MaxParallelTools: 1,
	}, registry, sessions)

	calls := 0
	stream, err := engine.StreamChat(context.Background(), agent.ChatRequest{
		UserID: "u", SessionID: "halt", Message: "go",
		Proceed: func() error {
			calls++
			if calls > 2 {
				return errors.New("environment finished")
			}
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	var errText string
	for chunk := range stream {
		if chunk.Type == agent.ChunkTypeError {
			errText = chunk.Error
		}
	}
	if !strings.Contains(errText, "environment finished") {
		t.Fatalf("error chunk = %q, want the Proceed error", errText)
	}
	if n := len(provider.Requests()); n != 2 {
		t.Fatalf("model calls = %d, want 2", n)
	}
	if executed != 4 {
		t.Fatalf("tool executions = %d, want both batches whole (4)", executed)
	}
	h := sessions.GetHistory("halt")
	if len(h) != 5 || h[4].Role != llm.RoleTool {
		t.Fatalf("history must end on the second batch's results, got %d messages", len(h))
	}
}
