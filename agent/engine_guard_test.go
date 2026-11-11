package agent_test

import (
	"context"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// countingEngine wires one tool that counts its executions and reports
// the given usage, over a scripted provider.
func countingEngine(t *testing.T, provider llm.LLMProvider, maxIter int, toolUsage *llm.Usage) (*agent.AgentEngine, *atomic.Int32, *agent.SessionManager) {
	t.Helper()
	var runs atomic.Int32
	registry := agent.NewToolRegistry()
	if err := registry.Register(&agent.Tool{
		Name:       "probe",
		Parameters: llm.Schema{"type": "object"},
		Execute: func(ctx context.Context, tc *agent.ToolContext, args map[string]any) (*agent.ToolResult, error) {
			runs.Add(1)
			return &agent.ToolResult{Data: "ok", Usage: toolUsage}, nil
		},
	}); err != nil {
		t.Fatal(err)
	}
	sessions := agent.NewSessionManager()
	engine := agent.NewAgentEngine(agent.AgentConfig{Provider: provider, MaxIterations: maxIter}, registry, sessions)
	return engine, &runs, sessions
}

func run(t *testing.T, engine *agent.AgentEngine, session string) (text string, done *agent.StreamChunk, errs []string) {
	t.Helper()
	stream, err := engine.StreamChat(context.Background(), agent.ChatRequest{UserID: "u", SessionID: session, Message: "go"})
	if err != nil {
		t.Fatal(err)
	}
	for c := range stream {
		switch c.Type {
		case agent.ChunkTypeText:
			text += c.Content
		case agent.ChunkTypeError:
			errs = append(errs, c.Error)
		case agent.ChunkTypeDone:
			cc := c
			done = &cc
		}
	}
	return
}

var callTurn = llm.MockTurn{ToolCalls: []llm.ToolCall{{ID: "c1", Name: "probe"}}}

func TestFinalIteration_IgnoredToolChoiceWithText(t *testing.T) {
	// The backend ignores tool_choice none and sends text plus a call on
	// the last iteration: the text is the answer, the call never runs.
	final := llm.MockTurn{Text: "Answer anyway.", ToolCalls: []llm.ToolCall{{ID: "c2", Name: "probe"}}}
	engine, runs, sessions := countingEngine(t, llm.NewMockProvider(callTurn, final), 2, nil)

	text, done, errs := run(t, engine, "s1")
	if len(errs) > 0 || done == nil {
		t.Fatalf("turn must end cleanly, errors: %v", errs)
	}
	if text != "Answer anyway." {
		t.Fatalf("text = %q", text)
	}
	if runs.Load() != 1 {
		t.Fatalf("tool runs = %d, want 1 (the final-iteration call must not execute)", runs.Load())
	}
	// History stays well-formed: no dangling tool call without a result.
	hist := sessions.GetHistory("s1")
	for _, p := range hist[len(hist)-1].Parts {
		if p.Kind == llm.PartToolCall {
			t.Fatal("dropped tool call leaked into history")
		}
	}
}

func TestFinalIteration_IgnoredToolChoiceWithoutText(t *testing.T) {
	// Only a tool call on the last iteration: no answer exists, and the
	// turn says so instead of ending in silence.
	engine, runs, _ := countingEngine(t, llm.NewMockProvider(callTurn, callTurn), 2, nil)

	_, done, errs := run(t, engine, "s2")
	if done != nil || len(errs) != 1 || !strings.Contains(errs[0], "tool budget exhausted") {
		t.Fatalf("want one honest error, got done=%v errs=%v", done, errs)
	}
	if runs.Load() != 1 {
		t.Fatalf("tool runs = %d, want 1", runs.Load())
	}
}

func TestTurnUsageIncludesToolSpend(t *testing.T) {
	// A tool that calls models itself (a sub-agent tool) reports its
	// spend; the turn total must include it.
	provider := llm.NewMockProvider(
		llm.MockTurn{ToolCalls: callTurn.ToolCalls, Usage: &llm.Usage{InputTokens: 10, OutputTokens: 1}},
		llm.MockTurn{Text: "done", Usage: &llm.Usage{InputTokens: 20, OutputTokens: 2}},
	)
	engine, _, sessions := countingEngine(t, provider, 5, &llm.Usage{InputTokens: 300, OutputTokens: 30})

	_, done, errs := run(t, engine, "s3")
	if len(errs) > 0 || done == nil {
		t.Fatalf("errors: %v", errs)
	}
	if done.Usage.InputTokens != 330 || done.Usage.OutputTokens != 33 {
		t.Fatalf("turn usage = %+v, want agent 30/3 + tool 300/30", done.Usage)
	}
	if got := sessions.SessionUsage("s3"); got.InputTokens != 330 {
		t.Fatalf("session usage = %+v", got)
	}
	// The context guard keeps measuring the agent's own context only.
	if in, _ := sessions.LastUsage("s3"); in != 20 {
		t.Fatalf("last call input = %d, want 20 (tool spend must not inflate it)", in)
	}
}

func TestFailedTurnReportsSpend(t *testing.T) {
	// The turn dies on its last iteration after a paid call and a paid
	// tool: the error chunk carries what was spent, so a caller that
	// retries (a sub-agent tool) can account for the lost attempt.
	provider := llm.NewMockProvider(
		llm.MockTurn{ToolCalls: callTurn.ToolCalls, Usage: &llm.Usage{InputTokens: 10, OutputTokens: 1}},
		llm.MockTurn{ToolCalls: callTurn.ToolCalls, Usage: &llm.Usage{InputTokens: 20, OutputTokens: 2}},
	)
	engine, _, _ := countingEngine(t, provider, 2, &llm.Usage{InputTokens: 300, OutputTokens: 30})

	stream, err := engine.StreamChat(context.Background(), agent.ChatRequest{UserID: "u", SessionID: "s7", Message: "go"})
	if err != nil {
		t.Fatal(err)
	}
	var failed *agent.StreamChunk
	for c := range stream {
		if c.Type == agent.ChunkTypeError {
			cc := c
			failed = &cc
		}
	}
	if failed == nil {
		t.Fatal("turn must fail: no answer on the final iteration")
	}
	if failed.Usage == nil || failed.Usage.InputTokens != 330 || failed.Usage.OutputTokens != 33 {
		t.Fatalf("error chunk usage = %+v, want 330/33", failed.Usage)
	}
}

func TestMaxTokensStop(t *testing.T) {
	t.Run("cut tool calls are not executed", func(t *testing.T) {
		provider := llm.NewMockProvider(
			llm.MockTurn{ToolCalls: callTurn.ToolCalls, StopReason: llm.StopMaxTokens},
			llm.MockTurn{Text: "retrying shorter"},
		)
		engine, runs, sessions := countingEngine(t, provider, 5, nil)
		if _, _, errs := run(t, engine, "s4"); len(errs) > 0 {
			t.Fatalf("errors: %v", errs)
		}
		if runs.Load() != 0 {
			t.Fatalf("a call cut by max_tokens ran %d time(s)", runs.Load())
		}
		var sawError bool
		for _, m := range sessions.GetHistory("s4") {
			for _, p := range m.Parts {
				if p.Kind == llm.PartToolResult && p.ToolResult.IsError {
					sawError = true
				}
			}
		}
		if !sawError {
			t.Fatal("the model must get an error result explaining the cut")
		}
	})

	t.Run("truncated answer is flagged on done", func(t *testing.T) {
		engine, _, _ := countingEngine(t, llm.NewMockProvider(llm.MockTurn{Text: "The answer is", StopReason: llm.StopMaxTokens}), 5, nil)
		_, done, _ := run(t, engine, "s5")
		if done == nil || done.StopReason != llm.StopMaxTokens {
			t.Fatalf("done must carry stop_reason max_tokens, got %+v", done)
		}
	})
}

func TestConcurrentTurnOnSameSessionRefused(t *testing.T) {
	sessions := agent.NewSessionManager()
	release, ok := sessions.BeginTurn("busy")
	if !ok {
		t.Fatal("first turn must be admitted")
	}
	engine := agent.NewAgentEngine(agent.AgentConfig{Provider: llm.NewMockProvider(llm.MockTurn{Text: "hi"})}, agent.NewToolRegistry(), sessions)

	_, done, errs := run(t, engine, "busy")
	if done != nil || len(errs) != 1 || !strings.Contains(errs[0], "already has a turn in progress") {
		t.Fatalf("concurrent turn must be refused, got done=%v errs=%v", done, errs)
	}
	if len(sessions.GetHistory("busy")) != 0 {
		t.Fatal("a refused turn must not touch the history")
	}

	release()
	if _, done, errs := run(t, engine, "busy"); done == nil || len(errs) > 0 {
		t.Fatalf("after release the session must accept turns: %v", errs)
	}
}

func TestNonImageAttachmentIsNamedForTheModel(t *testing.T) {
	provider := llm.NewMockProvider(llm.MockTurn{Text: "ok"})
	engine := agent.NewAgentEngine(agent.AgentConfig{Provider: provider}, agent.NewToolRegistry(), agent.NewSessionManager())
	stream, _ := engine.StreamChat(context.Background(), agent.ChatRequest{
		UserID: "u", SessionID: "att", Message: "summarize",
		Attachments: []agent.Attachment{{Name: "report.pdf", MediaType: "application/pdf", Data: "JVBERi0="}},
	})
	for range stream {
	}
	user := provider.Requests()[0].Messages[0]
	var found bool
	for _, p := range user.Parts {
		found = found || (p.Kind == llm.PartText && strings.Contains(p.Text, "report.pdf"))
	}
	if !found {
		t.Fatalf("non-image attachment must be named for the model: %+v", user.Parts)
	}
}

// hangingProvider blocks every call until the caller's context ends.
type hangingProvider struct{ llm.LLMProvider }

func (hangingProvider) StreamChat(ctx context.Context, req *llm.ChatRequest) (<-chan llm.StreamEvent, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

func TestCancelledTurnEndsWithError(t *testing.T) {
	// A turn cut off by its deadline must still say so to a consumer that
	// keeps reading: a bare closed channel is indistinguishable from a
	// turn that produced nothing. (The old send raced the dead context
	// and dropped the terminal chunk about half the time.)
	engine := agent.NewAgentEngine(agent.AgentConfig{Provider: hangingProvider{}}, agent.NewToolRegistry(), agent.NewSessionManager())
	for i := 0; i < 20; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
		stream, err := engine.StreamChat(ctx, agent.ChatRequest{UserID: "u", SessionID: fmt.Sprintf("c%d", i), Message: "go"})
		if err != nil {
			t.Fatal(err)
		}
		var last *agent.StreamChunk
		for c := range stream {
			cc := c
			last = &cc
		}
		cancel()
		if last == nil || last.Type != agent.ChunkTypeError || !strings.Contains(last.Error, "deadline") {
			t.Fatalf("run %d: want a terminal deadline error, got %+v", i, last)
		}
	}
}

func TestSession_BelongsToItsUser(t *testing.T) {
	provider := llm.NewMockProvider(llm.MockTurn{Text: "mine"})
	engine, _, sessions := countingEngine(t, provider, 3, nil)

	if _, _, errs := run(t, engine, "shared-id"); len(errs) > 0 {
		t.Fatalf("the owner's own turn failed: %v", errs)
	}
	stream, err := engine.StreamChat(context.Background(), agent.ChatRequest{UserID: "mallory", SessionID: "shared-id", Message: "what did they say?"})
	if err != nil {
		t.Fatal(err)
	}
	var errMsg string
	for c := range stream {
		if c.Type == agent.ChunkTypeError {
			errMsg = c.Error
		}
		if c.Type == agent.ChunkTypeText {
			t.Fatalf("another user got text from someone else's session: %q", c.Content)
		}
	}
	if !strings.Contains(errMsg, "another user") {
		t.Fatalf("expected a refusal, got %q", errMsg)
	}
	shared, _ := sessions.GetOrCreateSession(context.Background(), "shared-id", "u")
	if n := len(shared.History); n == 0 || n > 3 {
		t.Fatalf("the refused request must not touch the history (len %d)", n)
	}
}

func TestEstimatedUsageKeepsTheContextGuardArmed(t *testing.T) {
	sessions := agent.NewSessionManager()
	sessions.GetOrCreateSession(context.Background(), "s", "u")
	sessions.AddUsage("s", llm.Usage{InputTokens: 900, OutputTokens: 50})
	// A cancelled stream reports an estimate with no input count.
	sessions.AddUsage("s", llm.Usage{OutputTokens: 12, Estimated: true})
	if in, out := sessions.LastUsage("s"); in != 900 || out != 50 {
		t.Fatalf("last usage = %d/%d, want the real call's 900/50", in, out)
	}
}

func TestChat_CutOffStreamIsAnError(t *testing.T) {
	// A provider that blocks until the context ends: the turn is cut off.
	engine, _, _ := countingEngine(t, blockingProvider{}, 3, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	text, err := engine.Chat(ctx, agent.ChatRequest{UserID: "u", SessionID: "s", Message: "go"})
	if err == nil {
		t.Fatalf("an interrupted turn must not look like an answer (got %q)", text)
	}
}

func TestNewAgentEngine_NilPartsAreDefaulted(t *testing.T) {
	engine := agent.NewAgentEngine(agent.AgentConfig{Provider: llm.NewMockProvider(llm.MockTurn{Text: "ok"})}, nil, nil)
	if text, err := engine.Chat(context.Background(), agent.ChatRequest{UserID: "u", Message: "hi"}); err != nil || text != "ok" {
		t.Fatalf("a nil registry and sessions must work: %q %v", text, err)
	}
	noProvider := agent.NewAgentEngine(agent.AgentConfig{}, nil, nil)
	if _, err := noProvider.StreamChat(context.Background(), agent.ChatRequest{UserID: "u", Message: "hi"}); err == nil {
		t.Fatal("a missing provider is an error from StreamChat, not a panic in a goroutine")
	}
}

// blockingProvider streams nothing until the context is done.
type blockingProvider struct{}

func (blockingProvider) Name() string { return "blocking" }
func (blockingProvider) StreamChat(ctx context.Context, _ *llm.ChatRequest) (<-chan llm.StreamEvent, error) {
	ch := make(chan llm.StreamEvent)
	go func() { <-ctx.Done(); close(ch) }()
	return ch, nil
}
func (blockingProvider) Generate(context.Context, *llm.ChatRequest) (string, llm.Usage, error) {
	return "", llm.Usage{}, nil
}
func (blockingProvider) GenerateStructured(context.Context, *llm.ChatRequest, llm.Schema, any) (llm.Usage, error) {
	return llm.Usage{}, nil
}
