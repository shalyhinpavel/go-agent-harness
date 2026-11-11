package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/llm"
	"github.com/shalyhinpavel/go-agent-harness/trace"
)

// newGuardEngine builds an engine with explicit guard settings.
func newGuardEngine(t *testing.T, provider llm.LLMProvider, mut func(*agent.AgentConfig)) (*agent.AgentEngine, *agent.ToolRegistry, *agent.SessionManager) {
	t.Helper()

	registry := agent.NewToolRegistry()
	sessions := agent.NewSessionManager()
	cfg := agent.AgentConfig{Provider: provider, SystemPrompt: "test"}
	if mut != nil {
		mut(&cfg)
	}
	return agent.NewAgentEngine(cfg, registry, sessions), registry, sessions
}

func mustRegister(t *testing.T, r *agent.ToolRegistry, tool *agent.Tool) {
	t.Helper()
	if err := r.Register(tool); err != nil {
		t.Fatal(err)
	}
}

func TestStreamChat_UsageAccumulatedAndReported(t *testing.T) {
	provider := llm.NewMockProvider(
		llm.MockTurn{
			ToolCalls: []llm.ToolCall{{ID: "c1", Name: "calc"}},
			Usage:     &llm.Usage{InputTokens: 100, OutputTokens: 20},
		},
		llm.MockTurn{
			Text:  "done",
			Usage: &llm.Usage{InputTokens: 150, OutputTokens: 30, CacheReadTokens: 40},
		},
	)
	engine, registry, sessions := newGuardEngine(t, provider, nil)
	mustRegister(t, registry, &agent.Tool{
		Name: "calc", Description: "calc", Parameters: llm.Schema{"type": "object"},
		Execute: func(ctx context.Context, tc *agent.ToolContext, args map[string]any) (*agent.ToolResult, error) {
			return &agent.ToolResult{Data: map[string]any{"ok": true}}, nil
		},
	})

	chunks := collectChunks(t, mustStream(t, engine, "hi"))

	var done *agent.StreamChunk
	for i := range chunks {
		if chunks[i].Type == agent.ChunkTypeDone {
			done = &chunks[i]
		}
	}
	if done == nil || done.Usage == nil {
		t.Fatalf("done chunk must carry usage totals: %+v", done)
	}
	if done.Usage.InputTokens != 250 || done.Usage.OutputTokens != 50 || done.Usage.CacheReadTokens != 40 {
		t.Fatalf("turn usage = %+v, want {250 50 cache:40}", done.Usage)
	}

	total := sessions.SessionUsage("s1")
	if total.InputTokens != 250 || total.OutputTokens != 50 {
		t.Fatalf("session usage = %+v, want {250 50}", total)
	}
}

func TestStreamChat_ToolResultTruncatedForModelNotClient(t *testing.T) {
	big := strings.Repeat("x", 5000)
	provider := llm.NewMockProvider(
		llm.MockTurn{ToolCalls: []llm.ToolCall{{ID: "c1", Name: "big", Args: map[string]any{}}}},
		llm.MockTurn{Text: "ok"},
	)
	engine, registry, sessions := newGuardEngine(t, provider, func(c *agent.AgentConfig) {
		c.MaxToolResultBytes = 200
	})
	mustRegister(t, registry, &agent.Tool{
		Name: "big", Description: "big", Parameters: llm.Schema{"type": "object"},
		Execute: func(ctx context.Context, tc *agent.ToolContext, args map[string]any) (*agent.ToolResult, error) {
			return &agent.ToolResult{Data: map[string]any{"blob": big}}, nil
		},
	})

	chunks := collectChunks(t, mustStream(t, engine, "go"))

	// Client stream: full payload.
	var streamedFull bool
	for _, c := range chunks {
		if c.Type == agent.ChunkTypeToolResult {
			blob := c.ToolResult.Data.(map[string]any)["blob"].(string)
			streamedFull = blob == big
		}
	}
	if !streamedFull {
		t.Fatal("client stream must carry the full tool result")
	}

	// Model-facing history: truncated with a marker.
	history := sessions.GetHistory("s1")
	var truncated bool
	for _, msg := range history {
		for _, p := range msg.Parts {
			if p.Kind == llm.PartToolResult {
				s, ok := p.ToolResult.Content.(string)
				if ok && strings.Contains(s, "[truncated: first") && len(s) < 5000 {
					truncated = true
				}
			}
		}
	}
	if !truncated {
		t.Fatal("history tool result must be truncated with a marker")
	}
}

func TestExecuteToolsParallel_ConcurrencyLimit(t *testing.T) {
	var inFlight, maxInFlight int64
	var mu sync.Mutex

	provider := llm.NewMockProvider(
		llm.MockTurn{ToolCalls: []llm.ToolCall{
			{ID: "1", Name: "hold"}, {ID: "2", Name: "hold"}, {ID: "3", Name: "hold"},
			{ID: "4", Name: "hold"}, {ID: "5", Name: "hold"}, {ID: "6", Name: "hold"},
		}},
		llm.MockTurn{Text: "ok"},
	)
	engine, registry, _ := newGuardEngine(t, provider, func(c *agent.AgentConfig) {
		c.MaxParallelTools = 2
	})
	mustRegister(t, registry, &agent.Tool{
		Name: "hold", Description: "hold", Parameters: llm.Schema{"type": "object"},
		Execute: func(ctx context.Context, tc *agent.ToolContext, args map[string]any) (*agent.ToolResult, error) {
			cur := atomic.AddInt64(&inFlight, 1)
			mu.Lock()
			if cur > maxInFlight {
				maxInFlight = cur
			}
			mu.Unlock()
			time.Sleep(5 * time.Millisecond)
			atomic.AddInt64(&inFlight, -1)
			return &agent.ToolResult{Data: map[string]any{"ok": true}}, nil
		},
	})

	chunks := collectChunks(t, mustStream(t, engine, "go"))
	results := 0
	for _, c := range chunks {
		if c.Type == agent.ChunkTypeToolResult {
			results++
		}
	}
	if results != 6 {
		t.Fatalf("tool results = %d, want 6", results)
	}
	if maxInFlight > 2 {
		t.Fatalf("observed %d concurrent executions, limit is 2", maxInFlight)
	}
}

func TestStreamChat_ContextLimitGuard(t *testing.T) {
	provider := llm.NewMockProvider(
		llm.MockTurn{Text: "ok", Usage: &llm.Usage{InputTokens: 950, OutputTokens: 50}},
		llm.MockTurn{Text: "should not run"},
	)
	engine, _, _ := newGuardEngine(t, provider, func(c *agent.AgentConfig) {
		c.ContextLimit = 1000
		c.MaxTokens = 100
	})

	ctx := context.Background()
	if _, err := engine.Chat(ctx, agent.ChatRequest{UserID: "u1", SessionID: "s1", Message: "one"}); err != nil {
		t.Fatal(err)
	}

	_, err := engine.Chat(ctx, agent.ChatRequest{UserID: "u1", SessionID: "s1", Message: "two"})
	if err == nil || !strings.Contains(err.Error(), "context limit") {
		t.Fatalf("second turn must fail with a context-limit error, got %v", err)
	}
	if got := len(provider.Requests()); got != 1 {
		t.Fatalf("provider called %d times, want 1 (guard fired before the second call)", got)
	}
}

func TestStreamChat_TraceWritten(t *testing.T) {
	dir := t.TempDir()
	sink, err := trace.NewDirSink(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer sink.Close()

	provider := llm.NewMockProvider(
		llm.MockTurn{
			Thinking:  "hmm",
			ToolCalls: []llm.ToolCall{{ID: "c1", Name: "calc", Args: map[string]any{"a": 1}}},
			Usage:     &llm.Usage{InputTokens: 10, OutputTokens: 5},
		},
		llm.MockTurn{Text: "answer", Usage: &llm.Usage{InputTokens: 20, OutputTokens: 7}},
	)
	engine, registry, _ := newGuardEngine(t, provider, func(c *agent.AgentConfig) {
		c.TraceSink = sink
	})
	mustRegister(t, registry, &agent.Tool{
		Name: "calc", Description: "calc", Parameters: llm.Schema{"type": "object"},
		Execute: func(ctx context.Context, tc *agent.ToolContext, args map[string]any) (*agent.ToolResult, error) {
			return &agent.ToolResult{Data: map[string]any{"ok": true}}, nil
		},
	})

	if _, err := engine.Chat(context.Background(), agent.ChatRequest{UserID: "u1", SessionID: "s1", Message: "go"}); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(filepath.Join(dir, "s1.jsonl"))
	if err != nil {
		t.Fatalf("trace file: %v", err)
	}

	types := map[string]bool{}
	var sawUsage, sawLatency bool
	for _, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		var ev struct {
			Type      string     `json:"type"`
			Usage     *llm.Usage `json:"usage"`
			LatencyMS int64      `json:"latency_ms"`
		}
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			t.Fatalf("bad JSONL line %q: %v", line, err)
		}
		types[ev.Type] = true
		if ev.Usage != nil && ev.Usage.InputTokens > 0 {
			sawUsage = true
		}
		if ev.LatencyMS >= 0 && ev.Type == "llm_call" {
			sawLatency = true
		}
	}
	for _, want := range []string{
		trace.EventTurnStart, trace.EventThinking, trace.EventToolCall,
		trace.EventToolResult, trace.EventLLMCall, trace.EventTurnEnd,
	} {
		if !types[want] {
			t.Fatalf("trace missing %q event; got %v", want, types)
		}
	}
	if !sawUsage || !sawLatency {
		t.Fatal("llm_call events must carry usage and latency")
	}
}

// TestTraceRequestsRecordOnlyTheAnsweringCall: trace.requests stores the
// request that produced the final answer, once per turn — the tool-decision
// turn before it would only repeat the same history.
func TestTraceRequestsRecordOnlyTheAnsweringCall(t *testing.T) {
	dir := t.TempDir()
	sink, err := trace.NewDirSink(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer sink.Close()

	provider := llm.NewMockProvider(
		llm.MockTurn{ToolCalls: []llm.ToolCall{{ID: "c1", Name: "calc", Args: map[string]any{"a": 1}}},
			Usage: &llm.Usage{InputTokens: 10, OutputTokens: 5}},
		llm.MockTurn{Text: "answer", Usage: &llm.Usage{InputTokens: 20, OutputTokens: 7}},
	)
	engine, registry, _ := newGuardEngine(t, provider, func(c *agent.AgentConfig) {
		c.TraceSink = sink
		c.TraceRequests = true
	})
	mustRegister(t, registry, &agent.Tool{
		Name: "calc", Description: "calc", Parameters: llm.Schema{"type": "object"},
		Execute: func(ctx context.Context, tc *agent.ToolContext, args map[string]any) (*agent.ToolResult, error) {
			return &agent.ToolResult{Data: map[string]any{"ok": true}}, nil
		},
	})
	if _, err := engine.Chat(context.Background(), agent.ChatRequest{UserID: "u1", SessionID: "s1", Message: "go"}); err != nil {
		t.Fatal(err)
	}
	sink.Close()

	raw, err := os.ReadFile(filepath.Join(dir, "s1.jsonl"))
	if err != nil {
		t.Fatalf("trace file: %v", err)
	}
	calls, withRequest := 0, 0
	for _, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		var ev struct {
			Type    string          `json:"type"`
			Request json.RawMessage `json:"request"`
		}
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			t.Fatalf("bad JSONL line %q: %v", line, err)
		}
		if ev.Type != trace.EventLLMCall {
			continue
		}
		calls++
		if len(ev.Request) > 0 {
			withRequest++
			var req struct {
				Messages []json.RawMessage `json:"messages"`
			}
			if err := json.Unmarshal(ev.Request, &req); err != nil || len(req.Messages) == 0 {
				t.Fatalf("request must carry the messages the answer came from: %s (%v)", ev.Request, err)
			}
		}
	}
	if calls != 2 || withRequest != 1 {
		t.Fatalf("llm_call events = %d, with request = %d; want 2 and 1", calls, withRequest)
	}
}

func mustStream(t *testing.T, engine *agent.AgentEngine, msg string) <-chan agent.StreamChunk {
	t.Helper()
	stream, err := engine.StreamChat(context.Background(), agent.ChatRequest{
		UserID: "u1", SessionID: "s1", Message: msg,
	})
	if err != nil {
		t.Fatal(err)
	}
	return stream
}

func TestStreamChat_ImageAttachmentsReachProvider(t *testing.T) {
	provider := llm.NewMockProvider(llm.MockTurn{Text: "a chart"})
	engine, _, sessions := newGuardEngine(t, provider, nil)

	req := agent.ChatRequest{
		UserID: "u1", SessionID: "s1", Message: "what is this?",
		Attachments: []agent.Attachment{
			{Name: "chart.png", MediaType: "image/png", Data: "AAAA"},
			{Name: "notes.txt", MediaType: "text/plain", Data: "aGVsbG8="},
		},
	}
	if _, err := engine.Chat(context.Background(), req); err != nil {
		t.Fatal(err)
	}

	first := provider.Requests()[0].Messages[0]
	if first.Role != llm.RoleUser {
		t.Fatalf("first message role = %q", first.Role)
	}
	var images int
	for _, p := range first.Parts {
		if p.Kind == llm.PartImage {
			images++
		}
	}
	if images != 1 {
		t.Fatalf("image parts = %d, want 1 (only image/* attachments)", images)
	}

	// Persisted history keeps the image part for later turns.
	history := sessions.GetHistory("s1")
	if history[0].Parts[1].Kind != llm.PartImage {
		t.Fatalf("history lost the image part: %+v", history[0])
	}
}

func TestStreamChat_ThinkingPreservedWhenEnabled(t *testing.T) {
	provider := llm.NewMockProvider(llm.MockTurn{Thinking: "chain of thought", Text: "answer"})
	engine, _, sessions := newGuardEngine(t, provider, func(c *agent.AgentConfig) {
		c.PreserveReasoning = true
	})

	if _, err := engine.Chat(context.Background(), agent.ChatRequest{
		UserID: "u1", SessionID: "s1", Message: "q",
	}); err != nil {
		t.Fatal(err)
	}

	history := sessions.GetHistory("s1")
	assistant := history[len(history)-1]
	var thinking string
	for _, p := range assistant.Parts {
		if p.Kind == llm.PartThinking {
			thinking = p.Text
		}
	}
	if thinking != "chain of thought" {
		t.Fatalf("thinking not preserved: %+v", assistant.Parts)
	}

	// The provider request must ask for reasoning replay.
	if !provider.Requests()[0].PreserveReasoning {
		t.Fatal("ChatRequest.PreserveReasoning not set")
	}
}

func TestCompactHistory_StripsThinkingOnlyWhenCut(t *testing.T) {
	sessions := agent.NewSessionManager()
	sessions.GetOrCreateSession(context.Background(), "s", "u1")
	sessions.AppendMessage(context.Background(), "s", llm.UserText("one"))
	sessions.AppendMessage(context.Background(), "s", &llm.Message{
		Role:  llm.RoleAssistant,
		Parts: []llm.Part{llm.ThinkingPart("old thought"), llm.TextPart("old")},
	})
	sessions.AppendMessage(context.Background(), "s", llm.UserText("two"))
	sessions.AppendMessage(context.Background(), "s", &llm.Message{
		Role:  llm.RoleAssistant,
		Parts: []llm.Part{llm.ThinkingPart("recent thought"), llm.TextPart("recent")},
	})

	// No cut: everything stays, thinking included.
	full := sessions.CompactHistory("s", 10)
	thoughts := 0
	for _, m := range full {
		for _, p := range m.Parts {
			if p.Kind == llm.PartThinking {
				thoughts++
			}
		}
	}
	if thoughts != 2 {
		t.Fatalf("no-cut window must keep thinking parts, got %d", thoughts)
	}

	// Cut: the prefix changed, thinking must be stripped.
	cut := sessions.CompactHistory("s", 2)
	for _, m := range cut {
		for _, p := range m.Parts {
			if p.Kind == llm.PartThinking {
				t.Fatal("cut window must not carry thinking parts")
			}
		}
	}
}

// Parallel calls to one tool must be told apart: the chunks and the trace
// carry the call id, and the trace names the agent.
func TestToolCallIDPairsCallsWithResults(t *testing.T) {
	dir := t.TempDir()
	sink, err := trace.NewDirSink(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer sink.Close()

	provider := llm.NewMockProvider(
		llm.MockTurn{ToolCalls: []llm.ToolCall{
			{ID: "c1", Name: "calc", Args: map[string]any{"a": 1}},
			{ID: "c2", Name: "calc", Args: map[string]any{"a": 2}},
		}},
		llm.MockTurn{Text: "done"},
	)
	engine, registry, _ := newGuardEngine(t, provider, func(c *agent.AgentConfig) {
		c.TraceSink = sink
		c.Name = "researcher"
	})
	mustRegister(t, registry, &agent.Tool{
		Name: "calc", Description: "calc", Parameters: llm.Schema{"type": "object"},
		Execute: func(context.Context, *agent.ToolContext, map[string]any) (*agent.ToolResult, error) {
			return &agent.ToolResult{Data: "ok"}, nil
		},
	})

	stream, err := engine.StreamChat(context.Background(), agent.ChatRequest{UserID: "u", SessionID: "s1", Message: "go"})
	if err != nil {
		t.Fatal(err)
	}
	calls, results := map[string]bool{}, map[string]bool{}
	for c := range stream {
		switch c.Type {
		case agent.ChunkTypeToolCall:
			calls[c.ToolCallID] = true
		case agent.ChunkTypeToolResult:
			results[c.ToolCallID] = true
		}
	}
	if !calls["c1"] || !calls["c2"] || !results["c1"] || !results["c2"] {
		t.Fatalf("call ids missing: calls %v results %v", calls, results)
	}

	sink.Close()
	raw, _ := os.ReadFile(filepath.Join(dir, "s1.jsonl"))
	var withID, named int
	for _, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		var ev struct {
			Type       string `json:"type"`
			ToolCallID string `json:"tool_call_id"`
			Agent      string `json:"agent"`
		}
		_ = json.Unmarshal([]byte(line), &ev)
		if (ev.Type == trace.EventToolCall || ev.Type == trace.EventToolResult) && ev.ToolCallID != "" {
			withID++
		}
		if ev.Agent == "researcher" {
			named++
		}
	}
	if withID != 4 || named == 0 {
		t.Fatalf("trace: %d events with a call id (want 4), %d naming the agent", withID, named)
	}
}

func TestCustomDataReachesTools(t *testing.T) {
	var got map[string]any
	provider := llm.NewMockProvider(
		llm.MockTurn{ToolCalls: []llm.ToolCall{{ID: "c1", Name: "peek"}}},
		llm.MockTurn{Text: "done"},
	)
	engine, registry, _ := newGuardEngine(t, provider, nil)
	mustRegister(t, registry, &agent.Tool{
		Name: "peek", Description: "peek", Parameters: llm.Schema{"type": "object"},
		Execute: func(_ context.Context, tc *agent.ToolContext, _ map[string]any) (*agent.ToolResult, error) {
			got = tc.CustomData
			return &agent.ToolResult{Data: "ok"}, nil
		},
	})
	req := agent.ChatRequest{UserID: "u", SessionID: "s", Message: "go", CustomData: map[string]any{"tenant": "acme"}}
	if _, err := engine.Chat(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if got["tenant"] != "acme" {
		t.Fatalf("the tool did not see the request's data: %v", got)
	}
}

func TestPromptAugmenterErrorEndsTheTurnBeforeTheMessageIsStored(t *testing.T) {
	provider := llm.NewMockProvider(llm.MockTurn{Text: "never asked"})
	engine, _, sessions := newGuardEngine(t, provider, func(c *agent.AgentConfig) {
		c.PromptAugmenter = func(context.Context, string, string, string) (string, error) {
			return "", errors.New("store down")
		}
	})
	_, err := engine.Chat(context.Background(), agent.ChatRequest{UserID: "u", SessionID: "s", Message: "hi"})
	if err == nil || !strings.Contains(err.Error(), "store down") {
		t.Fatalf("the augmenter's error must end the turn, got %v", err)
	}
	if h := sessions.GetHistory("s"); len(h) != 0 {
		t.Fatalf("an unanswered user message was left in the history: %d messages", len(h))
	}
	if n := len(provider.Requests()); n != 0 {
		t.Fatalf("the model was called %d times despite the failure", n)
	}
}
