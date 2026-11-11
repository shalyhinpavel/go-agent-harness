package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/shalyhinpavel/go-agent-harness/llm"
	"github.com/shalyhinpavel/go-agent-harness/trace"
)

// DefaultMaxIterations bounds the tool-calling loop: model calls per turn,
// the last of them a forced answer. A research or coding task takes more
// than a handful of tool rounds.
const DefaultMaxIterations = 10

func (e *AgentEngine) needsApproval(tool *Tool, args map[string]any) bool {
	if e.approvalPolicy != nil {
		return e.approvalPolicy(tool, args)
	}
	return !tool.ReadOnly
}

func skippedToolPart(c llm.ToolCall) llm.Part {
	return llm.ToolResultPart(llm.ToolResultData{
		CallID:  c.ID,
		Name:    c.Name,
		IsError: true,
		Content: map[string]any{"error": "skipped: request cancelled"},
	})
}

// executeToolsParallel runs tool calls concurrently (bounded by
// MaxParallelTools) and returns one tool_result part per call, in input
// order, unless a hook or approver aborts the turn with a non-nil error. It
// never produces nil parts: a cancelled or failed execution yields an error
// result the model can see.
//
// Results are truncated for the model when they exceed
// MaxToolResultBytes; the client stream still carries the full payload —
// the guard protects the conversation context, not the logs.
//
// The returned usage is what the tools themselves spent on model calls
// (ToolResult.Usage), summed across the batch even when a hook aborts it.
func (e *AgentEngine) executeToolsParallel(
	ctx context.Context,
	tc *ToolContext,
	hc HookContext,
	approver Approver,
	calls []llm.ToolCall,
	out chan<- StreamChunk,
) ([]llm.Part, llm.Usage, error) {

	batchCtx, batchCancel := context.WithCancel(ctx)
	defer batchCancel()

	var wg sync.WaitGroup
	parts := make([]llm.Part, len(calls))
	var usageMu sync.Mutex
	var spent llm.Usage

	var abortOnce sync.Once
	var fatalErr error
	abort := func(err error) {
		abortOnce.Do(func() {
			fatalErr = err
			batchCancel()
		})
	}

	limit := e.maxParallelTools
	if limit <= 0 || limit > len(calls) {
		limit = len(calls)
	}
	sem := make(chan struct{}, limit)

	run := func(idx int, c llm.ToolCall) {
		defer wg.Done()
		defer func() { <-sem }()

		tool, exists := e.registry.Get(c.Name)
		var result *ToolResult
		if !exists {
			result = &ToolResult{Error: fmt.Sprintf("tool %q not found in registry", c.Name)}
		} else {
			// Announce the invocation to stream subscribers.
			e.record(trace.Event{
				SessionID: tc.SessionID, TurnID: hc.TurnID, Type: trace.EventToolCall,
				ToolCallID: c.ID, ToolName: c.Name, ToolArgs: c.Args,
			})
			if out != nil {
				select {
				case out <- StreamChunk{Type: ChunkTypeToolCall, ToolCallID: c.ID, ToolName: c.Name, ToolArgs: c.Args}:
				case <-batchCtx.Done():
				}
			}

			denied := false
			if e.hooks.BeforeTool != nil {
				decision, err := e.hooks.BeforeTool(batchCtx, hc, c)
				if err != nil {
					parts[idx] = skippedToolPart(c)
					abort(fmt.Errorf("before tool %q: %w", c.Name, err))
					return
				}
				if decision.Deny {
					denied = true
					reason := strings.TrimSpace(decision.Reason)
					if reason == "" {
						reason = "denied by policy"
					}
					result = &ToolResult{Error: "tool call denied: " + reason}
				}
			}

			if !denied && approver != nil && e.needsApproval(tool, c.Args) {
				if out != nil {
					select {
					case out <- StreamChunk{Type: ChunkTypeApprovalRequired, ToolCallID: c.ID, ToolName: c.Name, ToolArgs: c.Args}:
					case <-batchCtx.Done():
						parts[idx] = skippedToolPart(c)
						abort(batchCtx.Err())
						return
					}
				}
				decision, err := approver(batchCtx, ApprovalRequest{
					UserID:     tc.UserID,
					SessionID:  tc.SessionID,
					TurnID:     hc.TurnID,
					ToolCallID: c.ID,
					ToolName:   c.Name,
					ToolArgs:   c.Args,
				})
				if err != nil {
					parts[idx] = skippedToolPart(c)
					abort(fmt.Errorf("approval for %q: %w", c.Name, err))
					return
				}
				if !decision.Approved {
					denied = true
					msg := "tool call denied by user"
					if r := strings.TrimSpace(decision.Reason); r != "" {
						msg += ": " + r
					}
					result = &ToolResult{Error: msg}
				}
			}

			if !denied {
				start := time.Now()
				res, err := e.runTool(batchCtx, tool, tc, c.Args)
				elapsed := time.Since(start)
				switch {
				case err != nil:
					result = &ToolResult{Error: err.Error()}
					if res != nil {
						result.Usage = res.Usage // a failed tool may still have spent tokens
					}
				case res == nil:
					result = &ToolResult{}
				default:
					result = res
				}
				origUsage := result.Usage
				if origUsage != nil {
					usageMu.Lock()
					spent.Add(*origUsage)
					usageMu.Unlock()
				}
				if e.hooks.AfterTool != nil {
					modified, hookErr := e.hooks.AfterTool(batchCtx, hc, c, result, elapsed)
					if hookErr != nil {
						parts[idx] = skippedToolPart(c)
						abort(fmt.Errorf("after tool %q: %w", c.Name, hookErr))
						return
					}
					if modified != nil {
						if modified.Usage == nil {
							modified.Usage = origUsage
						}
						result = modified
					}
				}
			}
		}

		data := llm.ToolResultData{CallID: c.ID, Name: c.Name}
		switch {
		case result.Error != "":
			data.IsError = true
			data.Content = map[string]any{"error": result.Error}
		case result.Data != nil:
			// The model-facing copy is bounded; the trace keeps everything.
			data.Content = e.truncateContent(result.Data)
		default:
			data.Content = map[string]any{"status": "success"}
		}
		parts[idx] = llm.ToolResultPart(data)

		// Report the outcome to stream subscribers and the trace.
		e.record(trace.Event{
			SessionID: tc.SessionID, TurnID: hc.TurnID, Type: trace.EventToolResult,
			ToolCallID: c.ID, ToolName: c.Name, ToolResult: result,
		})
		if out != nil {
			select {
			case out <- StreamChunk{Type: ChunkTypeToolResult, ToolCallID: c.ID, ToolName: c.Name, ToolResult: result}:
			case <-batchCtx.Done():
			}
		}
	}

	for i, call := range calls {
		// Bounded fan-out: wait for a slot unless the context is dead, in
		// which case the remaining calls become error results immediately.
		select {
		case sem <- struct{}{}:
		case <-batchCtx.Done():
			parts[i] = skippedToolPart(call)
			continue
		}
		wg.Add(1)
		go run(i, call)
	}

	wg.Wait()
	return parts, spent, fatalErr
}

// truncateContent bounds a tool payload by its JSON size. Over-budget
// results keep their head bytes plus a marker telling the model it asked
// for more than it got, so it can narrow the next request.
func (e *AgentEngine) truncateContent(content any) any {
	limit := e.maxToolResultBytes
	if limit <= 0 {
		return content
	}
	raw, err := json.Marshal(content)
	if err != nil || len(raw) <= limit {
		return content
	}

	markerRoom := 64
	kept := limit - markerRoom
	if kept < 0 {
		kept = 0
	}
	if kept > len(raw) {
		kept = len(raw)
	}
	head := strings.ToValidUTF8(string(raw[:kept]), "")
	return fmt.Sprintf("%s\n[truncated: first %d of %d bytes]", head, len(head), len(raw))
}

// runTool executes a tool and turns a panic in it into an error result: one
// misbehaving tool (a parser fed a hostile file, a nil in a plugin) must fail
// its call, not take the host process down.
func (e *AgentEngine) runTool(ctx context.Context, tool *Tool, tc *ToolContext, args map[string]any) (res *ToolResult, err error) {
	defer func() {
		if r := recover(); r != nil {
			e.logger().Error("tool panicked", "tool", tool.Name, "panic", r)
			res, err = nil, fmt.Errorf("tool %q panicked: %v", tool.Name, r)
		}
	}()
	return tool.Execute(ctx, tc, args)
}
