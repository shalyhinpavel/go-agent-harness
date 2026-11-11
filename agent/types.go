package agent

import (
	"context"
	"time"

	"github.com/shalyhinpavel/go-agent-harness/llm"
	"github.com/shalyhinpavel/go-agent-harness/trace"
)

// ToolContext is injected into every tool invocation, giving the handler
// identity and session scope. Extend it with service clients your tools need.
type ToolContext struct {
	UserID    string
	SessionID string
	// CustomData is the request-scoped data of ChatRequest.CustomData: a
	// tenant, a database handle, an auth claim. Tools of one batch run in
	// parallel and share it, so treat it as read-only.
	CustomData map[string]any
	// Trace, when set, routes tool-emitted trajectory events (e.g.
	// sub-agent steps) into the session's JSONL trace.
	Trace trace.Sink
}

// ToolResult is what a tool returns to the harness.
type ToolResult struct {
	Data  any    `json:"data,omitempty"`
	Error string `json:"error,omitempty"`
	// Usage is what the tool itself spent on model calls (e.g. a tool
	// that runs a sub-agent). The engine adds it to the turn and session
	// totals, so a turn's usage is its full cost, not just the agent's own.
	Usage *llm.Usage `json:"usage,omitempty"`
}

// ToolExecuteFunc is the signature of a tool handler.
type ToolExecuteFunc func(ctx context.Context, tc *ToolContext, args map[string]any) (*ToolResult, error)

// Tool is a capability the model can invoke. Parameters is canonical JSON
// Schema — every provider maps it onto its native format.
type Tool struct {
	Name        string
	Description string
	Parameters  llm.Schema
	Execute     ToolExecuteFunc
	// ReadOnly declares that the tool has no side effects on anything the
	// harness or the user owns: it looks things up and changes nothing.
	// Approval policies and sandboxes use it to tell safe tools from the rest.
	ReadOnly bool
}

// StreamChunkType identifies the kind of a client-facing stream chunk.
type StreamChunkType string

const (
	ChunkTypeText             StreamChunkType = "text"
	ChunkTypeThinking         StreamChunkType = "thinking"
	ChunkTypeToolCall         StreamChunkType = "tool_call"
	ChunkTypeApprovalRequired StreamChunkType = "approval_required"
	ChunkTypeToolResult       StreamChunkType = "tool_result"
	ChunkTypeDone             StreamChunkType = "done"
	ChunkTypeError            StreamChunkType = "error"
)

// StreamChunk is one block of a streamed agent response (SSE-ready).
type StreamChunk struct {
	Type    StreamChunkType `json:"type"`
	Content string          `json:"content,omitempty"`
	// ToolCallID pairs a tool_call chunk with its tool_result: parallel
	// calls to one tool are otherwise indistinguishable.
	ToolCallID string         `json:"tool_call_id,omitempty"`
	ToolName   string         `json:"tool_name,omitempty"`
	ToolArgs   map[string]any `json:"tool_args,omitempty"`
	ToolResult *ToolResult    `json:"tool_result,omitempty"`
	// Usage carries the turn's token totals on the done chunk: the agent's
	// own calls plus whatever tools spent (ToolResult.Usage). An error chunk
	// carries what the turn had spent before it failed — a failed turn is
	// not a free one.
	Usage *llm.Usage `json:"usage,omitempty"`
	// StopReason (done chunk) is why the final model call ended;
	// "max_tokens" means the answer text is cut off.
	StopReason llm.StopReason `json:"stop_reason,omitempty"`
	Error      string         `json:"error,omitempty"`
}

// Attachment is a user-supplied file riding on a chat turn. Images
// (PNG/JPEG/WebP/GIF) become vision input for multimodal models. Other
// types are the caller's to store in the session workspace (the HTTP
// server does, when the file tools are configured); the engine tells the
// model their names so it can open them with the read_file tool.
type Attachment struct {
	Name      string `json:"name"`       // file name, e.g. "chart.png"
	MediaType string `json:"media_type"` // e.g. "image/png"
	Data      string `json:"data"`       // base64-encoded bytes
}

// ChatRequest is the input to a chat turn.
type ChatRequest struct {
	UserID      string       `json:"user_id"`
	SessionID   string       `json:"session_id"`
	Message     string       `json:"message"`
	Attachments []Attachment `json:"attachments,omitempty"`
	// Proceed, when set, is consulted before every model call of the turn.
	// A non-nil error ends the turn there — between iterations, never
	// inside a tool batch — with an error chunk carrying that error's text.
	// An embedding program uses it to stop a long tool loop cleanly: the
	// task is finished on its side, or a spend budget or call limit ran
	// out. Cancelling the context instead could abort a tool call the
	// other side already executed.
	Proceed func() error `json:"-"`
	// CustomData travels to every tool of the turn as ToolContext.CustomData
	// (shallow-copied when the turn starts). It never comes from the wire:
	// the HTTP layer leaves it empty, an embedding program fills it from
	// its own authenticated context.
	CustomData map[string]any `json:"-"`
	// Approver overrides AgentConfig.Approver for this turn when non-nil
	// (used by the HTTP server to bind a turn's SSE approval broker).
	Approver Approver `json:"-"`
}

// HookContext carries turn and iteration metadata into Hooks. CustomData is
// the turn's read-only map (ChatRequest.CustomData).
type HookContext struct {
	UserID     string
	SessionID  string
	TurnID     string
	Iteration  int
	CustomData map[string]any
}

// ToolDecision is BeforeTool's verdict on a tool call. Deny refuses the call
// without ending the turn: the tool is not executed and Reason is returned to
// the model as an error tool result.
type ToolDecision struct {
	Deny   bool
	Reason string
}

// Hooks observes or intercepts model and tool calls inside the engine loop.
// Every field is optional. Returning a non-nil error from any hook aborts the
// turn with ChunkTypeError; return ToolDecision{Deny: true} from BeforeTool
// when the model should see a refusal and continue instead.
type Hooks struct {
	BeforeModel func(ctx context.Context, hc HookContext, req *llm.ChatRequest) error
	BeforeTool  func(ctx context.Context, hc HookContext, call llm.ToolCall) (ToolDecision, error)
	AfterTool   func(ctx context.Context, hc HookContext, call llm.ToolCall, res *ToolResult, elapsed time.Duration) (*ToolResult, error)
}

// ApprovalRequest describes a tool call awaiting human confirmation.
type ApprovalRequest struct {
	UserID     string
	SessionID  string
	TurnID     string
	ToolCallID string
	ToolName   string
	ToolArgs   map[string]any
}

// ApprovalDecision is the human's answer to an ApprovalRequest.
type ApprovalDecision struct {
	Approved bool
	Reason   string
}

// Approver asks a human (over a terminal prompt, an SSE round-trip, a chat
// bot) whether a tool call may run. Returning Approved: false denies the call
// and reports Reason to the model as a tool error so the turn can continue; a
// non-nil error aborts the turn.
type Approver func(ctx context.Context, req ApprovalRequest) (ApprovalDecision, error)

// ApprovalPolicy decides whether a tool call requires approval when an
// Approver is configured. When nil, every registered tool with ReadOnly == false
// requires approval and read-only tools run unattended.
type ApprovalPolicy func(tool *Tool, args map[string]any) bool
