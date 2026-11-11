package llm

import (
	"context"
	"errors"
)

// ChatRequest is a single completion request.
type ChatRequest struct {
	// System is the system prompt ("" = none). Providers place it natively.
	System string
	// Messages is the conversation so far, oldest first.
	Messages []Message
	// Tools are the tools the model may call.
	Tools []ToolDef
	// Model overrides the provider's default model ("" = default).
	Model string
	// MaxTokens caps the response size (0 = provider default).
	MaxTokens int
	// PreserveReasoning asks the provider to replay assistant thinking
	// parts back to reasoning-capable models (e.g. reasoning_details on
	// OpenAI-compatible endpoints; signed blocks on Anthropic). Providers
	// that cannot replay thinking ignore it.
	PreserveReasoning bool
	// ToolChoice constrains tool use (zero value = the model decides).
	ToolChoice ToolChoice
	// ExtraBody overrides provider extra-body fields for this request only,
	// merged last over the configured ones. A caller can use it to
	// cap one call's thinking budget without changing the sampling of every
	// other call in the run.
	ExtraBody map[string]any
}

// ToolChoiceMode constrains how the model may use the request's tools.
type ToolChoiceMode string

const (
	// ToolChoiceAuto lets the model decide (the default).
	ToolChoiceAuto ToolChoiceMode = ""
	// ToolChoiceNone demands a text answer. The intent is neutral; the wire
	// form is the adapter's call — Anthropic rejects a history holding tool
	// calls when no tools are declared, so it keeps the tools and sends
	// tool_choice none, while OpenAI-compatible backends honor omitted tools
	// more reliably than tool_choice none.
	ToolChoiceNone ToolChoiceMode = "none"
	// ToolChoiceRequired forces a call to some tool.
	ToolChoiceRequired ToolChoiceMode = "required"
	// ToolChoiceTool forces a call to the tool named in ToolChoice.Name.
	ToolChoiceTool ToolChoiceMode = "tool"
)

// ToolChoice is the tool-use constraint of a request.
type ToolChoice struct {
	Mode ToolChoiceMode
	Name string // ToolChoiceTool only
}

// StopReason is why a completion ended, normalized across backends.
type StopReason string

const (
	// StopUnknown: the backend did not say.
	StopUnknown StopReason = ""
	// StopEnd is a natural end of turn (end_turn, stop, stop_sequence).
	StopEnd StopReason = "end"
	// StopToolUse: the model ended its turn to call tools.
	StopToolUse StopReason = "tool_use"
	// StopMaxTokens: the output hit the token cap — text is incomplete and
	// tool-call arguments may be cut. Never treat it as a finished answer
	// without saying so.
	StopMaxTokens StopReason = "max_tokens"
	// StopOther covers content filters, refusals and anything unmapped.
	StopOther StopReason = "other"
)

// StreamEventKind discriminates provider stream events.
type StreamEventKind string

const (
	// EventText is a visible text delta.
	EventText StreamEventKind = "text"
	// EventThinking is a reasoning delta, streamed for observability. A
	// provider that can replay reasoning delivers the blocks themselves in
	// EventReasoning.
	EventThinking StreamEventKind = "thinking"
	// EventReasoning delivers the turn's reasoning blocks exactly once, in
	// the order the model wrote them, each with what its provider needs to
	// replay it. A provider that cannot replay reasoning sends none.
	EventReasoning StreamEventKind = "reasoning"
	// EventToolCalls delivers the turn's complete tool calls exactly once.
	// Providers aggregate incremental argument deltas internally.
	EventToolCalls StreamEventKind = "tool_calls"
	// EventError aborts the stream.
	EventError StreamEventKind = "error"
	// EventDone terminates the stream normally.
	EventDone StreamEventKind = "done"
)

// Usage is the token accounting of one completion. Cache counters are
// provider prompt-caching fields; they are 0 when the backend does not
// report them. Estimated marks a count derived from streamed text because
// the backend never sent a usage block (a cancelled or cut-off stream): it
// is an approximation, and anything comparing costs must be able to tell
// it apart from a server-reported figure.
type Usage struct {
	InputTokens      int  `json:"input_tokens"`
	OutputTokens     int  `json:"output_tokens"`
	CacheReadTokens  int  `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens int  `json:"cache_write_tokens,omitempty"`
	Estimated        bool `json:"estimated,omitempty"`
}

// Add accumulates another usage report into u. A total that contains any
// estimate stays marked as estimated.
func (u *Usage) Add(o Usage) {
	u.InputTokens += o.InputTokens
	u.OutputTokens += o.OutputTokens
	u.CacheReadTokens += o.CacheReadTokens
	u.CacheWriteTokens += o.CacheWriteTokens
	u.Estimated = u.Estimated || o.Estimated
}

// StreamEvent is one event of a streamed completion.
type StreamEvent struct {
	Kind StreamEventKind
	Text string
	// ToolCalls carries the turn's complete tool calls (EventToolCalls).
	ToolCalls []ToolCall
	// Reasoning carries the turn's reasoning blocks (EventReasoning).
	Reasoning []Part
	// Usage is set on EventDone when the backend reports token accounting.
	Usage *Usage
	// StopReason is set on EventDone.
	StopReason StopReason
	// Err aborts the stream (EventError).
	Err error
}

// LLMProvider is the port every backend implements. It is intentionally
// narrow: streaming chat, plain generation and schema-constrained
// generation cover agents and routers alike.
//
// Every method reports token usage: a call that spends tokens and hides
// it breaks cost accounting for everything built on top.
type LLMProvider interface {
	// Name returns the provider identifier (e.g. "anthropic", "openai", "mock").
	Name() string
	// StreamChat streams a completion for the request. The channel closes
	// after EventDone or EventError; consumers may stop reading once ctx
	// is cancelled — providers must not block on a dead consumer.
	StreamChat(ctx context.Context, req *ChatRequest) (<-chan StreamEvent, error)
	// Generate returns a complete text response and its usage.
	Generate(ctx context.Context, req *ChatRequest) (string, Usage, error)
	// GenerateStructured generates JSON constrained to schema, decodes it
	// into out (a pointer) and returns the usage of every attempt made
	// (fallback attempts included — they cost tokens too).
	GenerateStructured(ctx context.Context, req *ChatRequest, schema Schema, out any) (Usage, error)
}

// ErrStreamInterrupted marks a stream that broke after the backend had
// accepted the request: the connection dropped or the body was cut. It is
// not an API error (those are the SDKs' to retry, and a rejected request
// would be rejected again), so a caller that has seen no output yet may
// safely repeat the call.
var ErrStreamInterrupted = errors.New("llm: stream interrupted")

// ErrNoModel is returned when a request needs a model but none is configured.
var ErrNoModel = errors.New("llm: no model configured")
