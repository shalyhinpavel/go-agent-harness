package agent_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/llm"
)

func TestHooks_BeforeModelMutatesAndAborts(t *testing.T) {
	mock := llm.NewMockProvider(llm.MockTurn{Text: "ok"})
	var seenIter int
	var seenTenant any
	engine := agent.NewAgentEngine(agent.AgentConfig{
		Provider:     mock,
		SystemPrompt: "base",
		Hooks: agent.Hooks{
			BeforeModel: func(_ context.Context, hc agent.HookContext, req *llm.ChatRequest) error {
				seenIter = hc.Iteration
				seenTenant = hc.CustomData["tenant"]
				req.System += " + hook"
				return nil
			},
		},
	}, nil, nil)

	got, err := engine.Chat(context.Background(), agent.ChatRequest{
		UserID: "u1", SessionID: "s1", Message: "hi",
		CustomData: map[string]any{"tenant": "acme"},
	})
	if err != nil || got != "ok" {
		t.Fatalf("Chat = %q, %v", got, err)
	}
	if seenIter != 0 || seenTenant != "acme" {
		t.Fatalf("HookContext = iter %d tenant %v", seenIter, seenTenant)
	}
	if sys := mock.Requests()[0].System; sys != "base + hook" {
		t.Fatalf("provider saw system %q, want %q", sys, "base + hook")
	}

	// Returning an error aborts the turn before calling the provider.
	abortEngine := agent.NewAgentEngine(agent.AgentConfig{
		Provider: mock,
		Hooks: agent.Hooks{
			BeforeModel: func(context.Context, agent.HookContext, *llm.ChatRequest) error {
				return errors.New("budget exceeded")
			},
		},
	}, nil, nil)
	if _, err := abortEngine.Chat(context.Background(), agent.ChatRequest{UserID: "u1", SessionID: "s1", Message: "hi"}); err == nil || !strings.Contains(err.Error(), "budget exceeded") {
		t.Fatalf("expected BeforeModel error, got %v", err)
	}
	if n := len(mock.Requests()); n != 1 {
		t.Fatalf("provider calls = %d, want 1", n)
	}
}

func TestHooks_BeforeToolSoftDenyContinuesTurnAndSkipsApprover(t *testing.T) {
	var executed, askedApproval atomic.Bool
	reg := agent.NewToolRegistry()
	_ = reg.Register(&agent.Tool{
		Name:       "rm_file",
		Parameters: llm.Schema{"type": "object"},
		Execute: func(context.Context, *agent.ToolContext, map[string]any) (*agent.ToolResult, error) {
			executed.Store(true)
			return &agent.ToolResult{Data: "deleted"}, nil
		},
	})

	mock := llm.NewMockProvider(
		llm.MockTurn{ToolCalls: []llm.ToolCall{{ID: "c1", Name: "rm_file", Args: map[string]any{"path": "/etc/hosts"}}}},
		llm.MockTurn{Text: "Understood, skipping deletion."},
	)
	engine := agent.NewAgentEngine(agent.AgentConfig{
		Provider: mock,
		Hooks: agent.Hooks{
			BeforeTool: func(_ context.Context, _ agent.HookContext, call llm.ToolCall) (agent.ToolDecision, error) {
				if call.Args["path"] == "/etc/hosts" {
					return agent.ToolDecision{Deny: true, Reason: "protected path"}, nil
				}
				return agent.ToolDecision{}, nil
			},
		},
		Approver: func(context.Context, agent.ApprovalRequest) (agent.ApprovalDecision, error) {
			askedApproval.Store(true)
			return agent.ApprovalDecision{Approved: true}, nil
		},
	}, reg, nil)

	ans, err := engine.Chat(context.Background(), agent.ChatRequest{UserID: "u1", SessionID: "s1", Message: "clean up"})
	if err != nil || !strings.Contains(ans, "skipping") {
		t.Fatalf("Chat = %q, %v", ans, err)
	}
	if executed.Load() || askedApproval.Load() {
		t.Fatalf("denied tool must neither ask approval (%v) nor execute (%v)", askedApproval.Load(), executed.Load())
	}
	tr := mock.Requests()[1].Messages[len(mock.Requests()[1].Messages)-1].Parts[0].ToolResult
	if !tr.IsError || !strings.Contains(fmt.Sprint(tr.Content), "protected path") {
		t.Fatalf("model must see denial reason, got %+v", tr)
	}
}

func TestHooks_BeforeToolFatalErrorCancelsParallelSiblings(t *testing.T) {
	siblingStarted := make(chan struct{})
	var siblingCancelled atomic.Bool

	reg := agent.NewToolRegistry()
	_ = reg.Register(&agent.Tool{
		Name:       "slow",
		ReadOnly:   true,
		Parameters: llm.Schema{"type": "object"},
		Execute: func(ctx context.Context, _ *agent.ToolContext, _ map[string]any) (*agent.ToolResult, error) {
			close(siblingStarted)
			<-ctx.Done()
			siblingCancelled.Store(true)
			return nil, ctx.Err()
		},
	})
	_ = reg.Register(&agent.Tool{
		Name:       "forbidden",
		ReadOnly:   true,
		Parameters: llm.Schema{"type": "object"},
		Execute: func(context.Context, *agent.ToolContext, map[string]any) (*agent.ToolResult, error) {
			return &agent.ToolResult{Data: "never"}, nil
		},
	})

	mock := llm.NewMockProvider(
		llm.MockTurn{ToolCalls: []llm.ToolCall{
			{ID: "c1", Name: "slow"},
			{ID: "c2", Name: "forbidden"},
		}},
		llm.MockTurn{Text: "unreachable"},
	)
	engine := agent.NewAgentEngine(agent.AgentConfig{
		Provider: mock,
		Hooks: agent.Hooks{
			BeforeTool: func(_ context.Context, _ agent.HookContext, call llm.ToolCall) (agent.ToolDecision, error) {
				if call.Name == "forbidden" {
					<-siblingStarted
					return agent.ToolDecision{}, errors.New("policy engine offline")
				}
				return agent.ToolDecision{}, nil
			},
		},
	}, reg, nil)

	_, err := engine.Chat(context.Background(), agent.ChatRequest{UserID: "u1", SessionID: "s1", Message: "run"})
	if err == nil || !strings.Contains(err.Error(), "policy engine offline") {
		t.Fatalf("expected fatal BeforeTool error, got %v", err)
	}
	if !siblingCancelled.Load() {
		t.Fatal("parallel sibling tool was not cancelled on fatal hook error")
	}
}

func TestHooks_AfterToolRedactsOutputAndPreservesUsage(t *testing.T) {
	reg := agent.NewToolRegistry()
	_ = reg.Register(&agent.Tool{
		Name:       "subagent",
		ReadOnly:   true,
		Parameters: llm.Schema{"type": "object"},
		Execute: func(context.Context, *agent.ToolContext, map[string]any) (*agent.ToolResult, error) {
			return &agent.ToolResult{
				Data:  "token=sk-live-12345",
				Usage: &llm.Usage{InputTokens: 40, OutputTokens: 10},
			}, nil
		},
	})

	mock := llm.NewMockProvider(
		llm.MockTurn{ToolCalls: []llm.ToolCall{{ID: "c1", Name: "subagent"}}, Usage: &llm.Usage{InputTokens: 10, OutputTokens: 5}},
		llm.MockTurn{Text: "done", Usage: &llm.Usage{InputTokens: 20, OutputTokens: 5}},
	)
	engine := agent.NewAgentEngine(agent.AgentConfig{
		Provider: mock,
		Hooks: agent.Hooks{
			AfterTool: func(_ context.Context, _ agent.HookContext, _ llm.ToolCall, res *agent.ToolResult, _ time.Duration) (*agent.ToolResult, error) {
				clean := strings.ReplaceAll(fmt.Sprint(res.Data), "sk-live-12345", "[REDACTED]")
				return &agent.ToolResult{Data: clean}, nil
			},
		},
	}, reg, nil)

	stream, err := engine.StreamChat(context.Background(), agent.ChatRequest{UserID: "u1", SessionID: "s1", Message: "go"})
	if err != nil {
		t.Fatal(err)
	}
	var sawRedacted bool
	var doneUsage *llm.Usage
	for c := range stream {
		if c.Type == agent.ChunkTypeToolResult {
			if strings.Contains(fmt.Sprint(c.ToolResult.Data), "sk-live-12345") {
				t.Fatalf("stream chunk leaked unredacted data: %+v", c.ToolResult)
			}
			sawRedacted = strings.Contains(fmt.Sprint(c.ToolResult.Data), "[REDACTED]")
		}
		if c.Type == agent.ChunkTypeDone {
			doneUsage = c.Usage
		}
	}
	if !sawRedacted {
		t.Fatal("expected redacted tool result on stream")
	}
	if doneUsage == nil || doneUsage.InputTokens != 70 || doneUsage.OutputTokens != 20 {
		t.Fatalf("usage = %+v, want 70 in / 20 out (tool spend preserved)", doneUsage)
	}
	if sent := fmt.Sprint(mock.Requests()[1].Messages); strings.Contains(sent, "sk-live-12345") {
		t.Fatalf("model received unredacted secret: %s", sent)
	}
}

func TestApprover_ReadOnlySkipsAndWriteAsksAndRequestOverrides(t *testing.T) {
	var writes int
	reg := agent.NewToolRegistry()
	_ = reg.Register(&agent.Tool{
		Name:       "read_doc",
		ReadOnly:   true,
		Parameters: llm.Schema{"type": "object"},
		Execute: func(context.Context, *agent.ToolContext, map[string]any) (*agent.ToolResult, error) {
			return &agent.ToolResult{Data: "contents"}, nil
		},
	})
	_ = reg.Register(&agent.Tool{
		Name:       "write_doc",
		ReadOnly:   false,
		Parameters: llm.Schema{"type": "object"},
		Execute: func(context.Context, *agent.ToolContext, map[string]any) (*agent.ToolResult, error) {
			writes++
			return &agent.ToolResult{Data: "written"}, nil
		},
	})

	mock := llm.NewMockProvider(
		llm.MockTurn{ToolCalls: []llm.ToolCall{
			{ID: "r1", Name: "read_doc"},
			{ID: "w1", Name: "write_doc", Args: map[string]any{"path": "a.txt"}},
		}},
		llm.MockTurn{Text: "Write was rejected, read worked."},
		llm.MockTurn{ToolCalls: []llm.ToolCall{{ID: "w2", Name: "write_doc", Args: map[string]any{"path": "b.txt"}}}},
		llm.MockTurn{Text: "Written now."},
	)

	var asked []string
	engine := agent.NewAgentEngine(agent.AgentConfig{
		Provider: mock,
		Approver: func(_ context.Context, req agent.ApprovalRequest) (agent.ApprovalDecision, error) {
			asked = append(asked, req.ToolName+":"+req.ToolCallID)
			return agent.ApprovalDecision{Approved: false, Reason: "not yet"}, nil
		},
	}, reg, nil)

	// Turn 1: read_doc runs unattended; write_doc emits approval_required and is denied.
	stream, err := engine.StreamChat(context.Background(), agent.ChatRequest{UserID: "u1", SessionID: "s1", Message: "try"})
	if err != nil {
		t.Fatal(err)
	}
	var seq []string
	for c := range stream {
		switch c.Type {
		case agent.ChunkTypeToolCall, agent.ChunkTypeApprovalRequired, agent.ChunkTypeToolResult:
			seq = append(seq, string(c.Type)+":"+c.ToolCallID)
		}
	}
	if len(asked) != 1 || asked[0] != "write_doc:w1" {
		t.Fatalf("approver asked for %v, want [write_doc:w1]", asked)
	}
	if writes != 0 {
		t.Fatalf("denied write executed %d times", writes)
	}
	// w1 must appear in order: tool_call -> approval_required -> tool_result.
	var w1Seq []string
	for _, s := range seq {
		if strings.HasSuffix(s, ":w1") {
			w1Seq = append(w1Seq, s)
		}
	}
	wantW1 := "tool_call:w1 approval_required:w1 tool_result:w1"
	if strings.Join(w1Seq, " ") != wantW1 {
		t.Fatalf("w1 chunk sequence = %v, want %s", w1Seq, wantW1)
	}

	// Turn 2: per-request Approver overrides engine Approver and approves w2.
	ans, err := engine.Chat(context.Background(), agent.ChatRequest{
		UserID: "u1", SessionID: "s1", Message: "do it",
		Approver: func(_ context.Context, req agent.ApprovalRequest) (agent.ApprovalDecision, error) {
			return agent.ApprovalDecision{Approved: req.ToolCallID == "w2"}, nil
		},
	})
	if err != nil || ans != "Written now." || writes != 1 {
		t.Fatalf("turn 2: ans=%q err=%v writes=%d", ans, err, writes)
	}
}
