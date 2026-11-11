package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/google/uuid"

	"github.com/shalyhinpavel/go-agent-harness/llm"
	"github.com/shalyhinpavel/go-agent-harness/trace"
)

// DefaultStreamRetries is how often a stream that broke before any output
// is repeated.
const DefaultStreamRetries = 2

// DefaultHistoryWindow is the compacted history size sent to the model.
const DefaultHistoryWindow = 40

// DefaultHistoryTrimTo is the fraction of the history caps a window trim
// cuts down to (see WindowPolicy): the prefix then stays stable, and
// cacheable, until the window grows back to the cap.
const DefaultHistoryTrimTo = 0.75

// DefaultMaxToolResultBytes bounds one tool result before it is truncated
// for the model (the client stream keeps the full payload).
const DefaultMaxToolResultBytes = 32 * 1024

// DefaultMaxParallelTools caps concurrent tool executions per turn.
const DefaultMaxParallelTools = 8

// maxPreservedThinkingBytes bounds one persisted reasoning block — giant
// chains of thought must not eat the context window on every later turn.
const maxPreservedThinkingBytes = 64 * 1024

// contextReserve is the output budget assumed by the context-limit guard
// when the engine does not know the provider's max_tokens (0).
const contextReserve = 4096

// finalAnswerNote rides on the last iteration's request (never on the
// session history): a system-side constraint alone did not stop models
// from emitting pseudo tool-call markup as text (observed live with
// qwen3.8).
const finalAnswerNote = "[Tool budget exhausted. Give your final text answer now — do not attempt tool calls.]"

// silentTurnNudge is the one reminder after a call that produced reasoning
// and no visible answer.
const silentTurnNudge = "[Your last reply contained reasoning but no visible answer. Write your final answer now.]"

// PromptAugmenter rewrites the system prompt per turn — the injection
// port for a memory engine: layer a user dossier and retrieved facts onto
// the base prompt without touching the engine. An error ends the turn
// before anything is stored; handle a tolerable failure (a slow store you
// would rather skip) inside the function and return the base prompt.
type PromptAugmenter func(ctx context.Context, userID, sessionID, basePrompt string) (string, error)

// ContextCompactor condenses a slice of conversation history (the part that
// would otherwise be dropped at the context limit) into a summary that stands
// in for it. It reports what it spent even when it fails. A failure ends the
// turn with the context-limit error, and the history stays as it was.
type ContextCompactor func(ctx context.Context, dropped []llm.Message) (summary string, spent llm.Usage, err error)

// AgentEngine orchestrates the tool-calling loop on top of an LLMProvider.
// It is provider-agnostic: swap the provider to switch models or move from
// a cloud API to a self-hosted endpoint without touching this code.
//
// Memory is intentionally outside the engine: it plugs in through the
// PromptAugmenter (system prompt injection) and through ordinary tools
// (memory_search) — see package memory.
type AgentEngine struct {
	provider           llm.LLMProvider
	modelName          string
	systemPrompt       string
	promptAugmenter    PromptAugmenter
	compactor          ContextCompactor
	hooks              Hooks
	approver           Approver
	approvalPolicy     ApprovalPolicy
	sessions           *SessionManager
	registry           *ToolRegistry
	trace              trace.Sink
	maxIterations      int
	log                *slog.Logger
	name               string
	streamRetries      int
	retryDelay         time.Duration // first backoff; doubles per retry
	window             WindowPolicy
	maxTokens          int
	maxParallelTools   int
	maxToolResultBytes int
	contextLimit       int
	preserveReasoning  bool
	traceRequests      bool
}

// AgentConfig configures an engine.
type AgentConfig struct {
	Provider  llm.LLMProvider
	ModelName string // "" = provider default
	// Name labels this agent in the trace (trace.Event.Agent); set it when
	// several engines share one sink, e.g. a sub-agent run inside a tool.
	Name            string
	SystemPrompt    string
	PromptAugmenter PromptAugmenter // optional per-turn system prompt injection
	// Compactor condenses dropped history when the context limit is hit; nil
	// disables compaction (the engine then fails with a context-limit error).
	Compactor ContextCompactor
	// AutoCompact, when true and Compactor is nil, installs a summarizer
	// built on Provider + ModelName.
	AutoCompact bool
	// Hooks observes or intercepts model calls and tool executions.
	Hooks Hooks
	// Approver gates non-read-only tool calls (or those selected by
	// ApprovalPolicy) behind human confirmation. Overridden per turn by
	// ChatRequest.Approver when set.
	Approver Approver
	// ApprovalPolicy overrides which tools consult Approver; nil means
	// every registered tool with ReadOnly == false.
	ApprovalPolicy ApprovalPolicy
	MaxIterations  int // <=0 = DefaultMaxIterations
	HistoryWindow  int // <=0 = DefaultHistoryWindow
	// HistoryChars caps the history window by approximate payload size
	// (text, tool arguments and results); 0 = no size cap. Long tool loops
	// need it: tool results vary too much for a message count to bound
	// the request.
	HistoryChars int
	// HistoryTrimTo: <=0 = DefaultHistoryTrimTo; >=1 slides the window by
	// the minimum on every overflow.
	HistoryTrimTo float64
	// PinTask keeps the session's first user message (the task) in front
	// of the history window when older messages are dropped.
	PinTask            bool
	MaxTokens          int        // response cap; 0 = provider default
	MaxParallelTools   int        // <=0 = DefaultMaxParallelTools
	MaxToolResultBytes int        // <=0 = DefaultMaxToolResultBytes; model-facing truncation
	ContextLimit       int        // model context window; 0 = no guard
	TraceSink          trace.Sink // optional trajectory recorder (nil = off)
	// PreserveReasoning keeps the model's thinking in the conversation so
	// reasoning-capable models continue from where they left off (replayed
	// by providers that support it). Default off here; the config layer
	// turns it on by default.
	PreserveReasoning bool
	// TraceRequests puts the exact request behind the final answer on the
	// llm_call trace event (messages, system, tools). Heavy — the history
	// is repeated per turn — and off by default; turn it on to replay or
	// inspect exactly what the model saw.
	TraceRequests bool
	// Logger receives engine diagnostics (nil = slog.Default()).
	Logger *slog.Logger
	// StreamRetries is how often a model call is repeated when its stream
	// broke before producing any output (a dropped connection): 0 =
	// DefaultStreamRetries, negative = never. Errors the API answered with
	// are not repeated here — the provider SDKs already retry rate limits
	// and server errors.
	StreamRetries int
}

// NewAgentEngine builds an engine.
//
// A nil registry or sessions manager is replaced by an empty one. A missing
// Provider is reported by StreamChat, before any turn starts.
func NewAgentEngine(cfg AgentConfig, registry *ToolRegistry, sessions *SessionManager) *AgentEngine {
	if registry == nil {
		registry = NewToolRegistry()
	}
	if sessions == nil {
		sessions = NewSessionManager()
	}
	maxIter := cfg.MaxIterations
	if maxIter <= 0 {
		maxIter = DefaultMaxIterations
	}
	streamRetries := cfg.StreamRetries
	switch {
	case streamRetries == 0:
		streamRetries = DefaultStreamRetries
	case streamRetries < 0:
		streamRetries = 0
	}
	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}
	compactor := cfg.Compactor
	if compactor == nil && cfg.AutoCompact && cfg.Provider != nil {
		compactor = llmContextCompactor(cfg.Provider, cfg.ModelName)
	}
	window := cfg.HistoryWindow
	if window <= 0 {
		window = DefaultHistoryWindow
	}
	trimTo := cfg.HistoryTrimTo
	if trimTo <= 0 {
		trimTo = DefaultHistoryTrimTo
	}
	parallel := cfg.MaxParallelTools
	if parallel <= 0 {
		parallel = DefaultMaxParallelTools
	}
	toolBytes := cfg.MaxToolResultBytes
	if toolBytes <= 0 {
		toolBytes = DefaultMaxToolResultBytes
	}

	return &AgentEngine{
		provider:        cfg.Provider,
		modelName:       cfg.ModelName,
		systemPrompt:    cfg.SystemPrompt,
		promptAugmenter: cfg.PromptAugmenter,
		compactor:       compactor,
		hooks:           cfg.Hooks,
		approver:        cfg.Approver,
		approvalPolicy:  cfg.ApprovalPolicy,
		sessions:        sessions,
		registry:        registry,
		trace:           cfg.TraceSink,
		maxIterations:   maxIter,
		log:             logger,
		name:            cfg.Name,
		streamRetries:   streamRetries,
		retryDelay:      500 * time.Millisecond,
		window: WindowPolicy{
			MaxItems: window,
			MaxChars: cfg.HistoryChars,
			TrimTo:   trimTo,
			PinTask:  cfg.PinTask,
		},
		maxTokens:          cfg.MaxTokens,
		maxParallelTools:   parallel,
		maxToolResultBytes: toolBytes,
		contextLimit:       cfg.ContextLimit,
		preserveReasoning:  cfg.PreserveReasoning,
		traceRequests:      cfg.TraceRequests,
	}
}

// record appends a trajectory event when a sink is attached.
func (e *AgentEngine) record(ev trace.Event) {
	if e.trace != nil {
		if ev.Agent == "" {
			ev.Agent = e.name
		}
		e.trace.Record(ev)
	}
}

// userMessage builds the user turn: the text plus image attachments as
// vision parts. Other attachments are named in a note — the caller stored
// them in the session workspace (see Attachment), the read_file tool opens
// them.
func (e *AgentEngine) userMessage(req *ChatRequest) *llm.Message {
	if len(req.Attachments) == 0 {
		return llm.UserText(req.Message)
	}

	parts := []llm.Part{llm.TextPart(req.Message)}
	for _, a := range req.Attachments {
		if strings.HasPrefix(a.MediaType, "image/") {
			parts = append(parts, llm.ImagePart(a.MediaType, a.Data))
			continue
		}
		parts = append(parts, llm.TextPart(fmt.Sprintf(
			"\n\n[Attached file %q (%s) is in the session workspace — open it with the read_file tool.]", a.Name, a.MediaType)))
	}
	return llm.UserMessage(parts...)
}

// hasText reports whether an assistant message carries visible text.
func hasText(msg *llm.Message) bool {
	for _, p := range msg.Parts {
		if p.Kind == llm.PartText && strings.TrimSpace(p.Text) != "" {
			return true
		}
	}
	return false
}

// withoutToolCalls returns the message minus its tool-call parts.
func withoutToolCalls(msg *llm.Message) *llm.Message {
	out := &llm.Message{Role: msg.Role}
	for _, p := range msg.Parts {
		if p.Kind != llm.PartToolCall {
			out.Parts = append(out.Parts, p)
		}
	}
	return out
}

// insertPart places part at index i (or appends when i is out of range).
func insertPart(parts []llm.Part, i int, part llm.Part) []llm.Part {
	if i < 0 || i >= len(parts) {
		return append(parts, part)
	}
	parts = append(parts, llm.Part{})
	copy(parts[i+1:], parts[i:])
	parts[i] = part
	return parts
}

// truncatedCallResults answers tool calls whose arguments were cut by the
// output token cap: running a tool on half its arguments is worse than
// telling the model to retry with a shorter call.
func truncatedCallResults(calls []llm.ToolCall) []llm.Part {
	parts := make([]llm.Part, 0, len(calls))
	for _, c := range calls {
		parts = append(parts, llm.ToolResultPart(llm.ToolResultData{
			CallID: c.ID, Name: c.Name, IsError: true,
			Content: map[string]any{"error": "not executed: the response hit the output token cap (max_tokens) while writing this call, so its arguments are incomplete — retry with a shorter call"},
		}))
	}
	return parts
}

// attachmentSummary traces attachments by name and size without the bytes.
func attachmentSummary(atts []Attachment) map[string]any {
	if len(atts) == 0 {
		return nil
	}
	out := map[string]any{"count": len(atts)}
	names := make([]string, 0, len(atts))
	for _, a := range atts {
		names = append(names, fmt.Sprintf("%s (%d b64 chars)", a.Name, len(a.Data)))
	}
	out["attachments"] = names
	return out
}

// ErrSessionOwner means the session was opened for another user. A session id
// is not a credential, so no turn runs on it for anyone else.
var ErrSessionOwner = errors.New("agent: session belongs to another user")

// ErrSessionBusy means a turn is already in flight on the session.
var ErrSessionBusy = errors.New("agent: session already has a turn in progress")

// CheckSession reports whether userID may start a turn on sessionID now. It
// is the admission StreamChat makes, offered ahead of work the caller does for
// the session, such as storing uploads, so a refused request leaves nothing
// behind. It is advisory: a turn that starts in the meantime is still refused
// by StreamChat. A session that does not exist yet passes; its first turn
// creates it.
func (e *AgentEngine) CheckSession(ctx context.Context, userID, sessionID string) error {
	owner, found, err := e.sessions.Owner(ctx, sessionID)
	if err != nil {
		return fmt.Errorf("agent: check session %q: %w", sessionID, err)
	}
	if found && owner != userID {
		return ErrSessionOwner
	}
	if e.sessions.Busy(sessionID) {
		return ErrSessionBusy
	}
	return nil
}

// StreamChat runs the agent loop and streams the outcome as chunks.
// The returned channel is closed when the turn completes, fails or is
// cancelled; stopping reads abandons the run and the goroutine exits.
// A turn ends with a done or an error chunk; a channel that closes
// without either was cut off (the context ended with the buffer full) and
// did not complete.
func (e *AgentEngine) StreamChat(ctx context.Context, req ChatRequest) (<-chan StreamChunk, error) {
	if e.provider == nil {
		return nil, errors.New("agent: no provider configured")
	}
	out := make(chan StreamChunk, 100)

	go func() {
		defer close(out)

		turnID := uuid.New().String()

		sendChunk := func(chunk StreamChunk) bool {
			// Room in the buffer wins over a dead context: the error that
			// ends a cancelled turn still reaches a consumer that keeps
			// reading. Only a full buffer and a gone consumer drop it.
			select {
			case out <- chunk:
				return true
			default:
			}
			select {
			case out <- chunk:
				return true
			case <-ctx.Done():
				return false
			}
		}

		session, err := e.sessions.GetOrCreateSession(ctx, req.SessionID, req.UserID)
		if err != nil {
			errMsg := "session: " + err.Error()
			e.record(trace.Event{SessionID: req.SessionID, TurnID: turnID, Type: trace.EventError, Error: errMsg})
			sendChunk(StreamChunk{Type: ChunkTypeError, Error: errMsg})
			return
		}
		if session.UserID != req.UserID {
			// A session id is not a credential: whoever knows it must still
			// be the user it was opened for.
			errMsg := fmt.Sprintf("session %q belongs to another user", session.ID)
			e.record(trace.Event{SessionID: session.ID, TurnID: turnID, Type: trace.EventError, Error: errMsg})
			sendChunk(StreamChunk{Type: ChunkTypeError, Error: errMsg})
			return
		}
		release, ok := e.sessions.BeginTurn(session.ID)
		if !ok {
			errMsg := fmt.Sprintf("session %q already has a turn in progress — wait for it to finish", session.ID)
			e.record(trace.Event{SessionID: session.ID, TurnID: turnID, Type: trace.EventError, Error: errMsg})
			sendChunk(StreamChunk{Type: ChunkTypeError, Error: errMsg})
			return
		}
		defer release()
		// Turn accounting: turnUsage (lead calls + tool spend) rides the
		// done chunk; the context guard projects from the session's last
		// reported call.
		var turnUsage llm.Usage
		// persist ends the turn when the session store refused a write: a
		// conversation that is not saved must not look as if it were.
		persist := func(err error) bool {
			if err == nil {
				return true
			}
			e.logger().Error("session store write failed", "session", session.ID, "err", err)
			e.record(trace.Event{SessionID: session.ID, TurnID: turnID, Type: trace.EventError, Error: err.Error()})
			sendChunk(StreamChunk{Type: ChunkTypeError, Usage: &turnUsage, Error: err.Error()})
			return false
		}
		tc := &ToolContext{UserID: req.UserID, SessionID: session.ID, Trace: e.trace, CustomData: maps.Clone(req.CustomData)}

		// Before the user message is stored: a failing augmenter must not
		// leave an unanswered message in the history.
		system := e.systemPrompt
		if e.promptAugmenter != nil {
			var err error
			if system, err = e.promptAugmenter(ctx, req.UserID, session.ID, system); err != nil {
				errMsg := "prompt augmenter: " + err.Error()
				e.record(trace.Event{SessionID: session.ID, TurnID: turnID, Type: trace.EventError, Error: errMsg})
				sendChunk(StreamChunk{Type: ChunkTypeError, Error: errMsg})
				return
			}
		}

		// Reasoning blocks are valid only under the system prompt and tools
		// that produced them, and the augmenter may return a different prompt
		// on each turn: a change strips the stored reasoning before this
		// turn's request is built.
		tools := e.registry.ToolDefs()
		currentBinding := requestBinding(system, tools)
		if err := e.sessions.BindReasoning(ctx, session.ID, currentBinding); err != nil {
			errMsg := err.Error()
			e.record(trace.Event{SessionID: session.ID, TurnID: turnID, Type: trace.EventError, Error: errMsg})
			sendChunk(StreamChunk{Type: ChunkTypeError, Error: errMsg})
			return
		}

		turnApprover := req.Approver
		if turnApprover == nil {
			turnApprover = e.approver
		}

		chatReq := &llm.ChatRequest{
			System:            system,
			Model:             e.modelName,
			Tools:             tools,
			MaxTokens:         e.maxTokens,
			PreserveReasoning: e.preserveReasoning,
		}

		var lastStop llm.StopReason

		// silentNudges counts the reminders sent after a reasoning-only reply.
		var forceNudge string
		var silentNudges int

		for iteration := 0; iteration < e.maxIterations; iteration++ {
			select {
			case <-ctx.Done():
				sendChunk(StreamChunk{Type: ChunkTypeError, Usage: &turnUsage, Error: "request cancelled: " + ctx.Err().Error()})
				return
			default:
			}
			if req.Proceed != nil {
				if err := req.Proceed(); err != nil {
					errMsg := "turn halted: " + err.Error()
					e.record(trace.Event{SessionID: session.ID, TurnID: turnID, Type: trace.EventError, Iteration: iteration, Error: errMsg})
					sendChunk(StreamChunk{Type: ChunkTypeError, Usage: &turnUsage, Error: errMsg})
					return
				}
			}

			// Context guard: project the next request size from the last
			// reported usage and stop before the provider would reject it.
			if lastInput, lastOutput := e.sessions.LastUsage(session.ID); e.contextLimit > 0 && lastInput > 0 {
				reserve := e.maxTokens
				if reserve <= 0 {
					reserve = contextReserve
				}
				projected := lastInput + lastOutput
				if projected+reserve > e.contextLimit {
					var compactErr error
					if e.compactor != nil {
						var spent llm.Usage
						spent, compactErr = e.compactSession(ctx, session, turnID)
						turnUsage.Add(spent)
						e.sessions.AddToolUsage(session.ID, spent)
					}
					if e.compactor == nil || compactErr != nil {
						errMsg := fmt.Sprintf(
							"context limit exceeded: next request projected at ~%d tokens (+%d output reserve), limit is %d — lower history_window or start a new session",
							projected, reserve, e.contextLimit)
						if compactErr != nil {
							errMsg += "; compaction failed: " + compactErr.Error()
						}
						e.record(trace.Event{SessionID: session.ID, TurnID: turnID, Type: trace.EventError, Error: errMsg})
						sendChunk(StreamChunk{Type: ChunkTypeError, Usage: &turnUsage, Error: errMsg})
						return
					}
					// The projection above described the old history; the window below is
					// built from the compacted one.
					e.logger().Info("history compacted at context limit", "session", session.ID)
				}
			}

			// The user message is stored once the first call is admitted: a turn
			// the checks above refuse leaves the history as it was.
			if iteration == 0 {
				if !persist(e.sessions.AppendMessage(ctx, session.ID, e.userMessage(&req))) {
					return
				}
				e.record(trace.Event{
					SessionID: session.ID, TurnID: turnID, Type: trace.EventTurnStart, Text: req.Message,
					ToolArgs: attachmentSummary(req.Attachments),
				})
			}
			chatReq.Messages = e.sessions.Window(session.ID, e.window)

			// The final iteration demands a text answer: a model still
			// calling tools at the budget's edge must answer, not exhaust
			// the loop in silence (observed live with reasoning models that
			// tool-call until the very end). The constraint is ToolChoice
			// none, not stripped tools — Anthropic rejects a history holding
			// tool calls when no tools are declared; each adapter picks the
			// wire form its backends honor.
			finalIteration := iteration == e.maxIterations-1
			chatReq.Model = e.modelName
			chatReq.MaxTokens = e.maxTokens
			chatReq.PreserveReasoning = e.preserveReasoning
			chatReq.Tools = e.registry.ToolDefs()
			chatReq.System = system
			chatReq.ToolChoice = llm.ToolChoice{}
			if finalIteration {
				chatReq.ToolChoice = llm.ToolChoice{Mode: llm.ToolChoiceNone}
				chatReq.Messages = append(chatReq.Messages, *llm.UserText(finalAnswerNote))
			}
			if forceNudge != "" {
				chatReq.Messages = append(chatReq.Messages, *llm.UserText(forceNudge))
			}

			hc := HookContext{
				UserID:     req.UserID,
				SessionID:  session.ID,
				TurnID:     turnID,
				Iteration:  iteration,
				CustomData: tc.CustomData,
			}
			if e.hooks.BeforeModel != nil {
				if err := e.hooks.BeforeModel(ctx, hc, chatReq); err != nil {
					errMsg := "before model: " + err.Error()
					e.record(trace.Event{SessionID: session.ID, TurnID: turnID, Type: trace.EventError, Iteration: iteration, Error: errMsg})
					sendChunk(StreamChunk{Type: ChunkTypeError, Usage: &turnUsage, Error: errMsg})
					return
				}
				if b := requestBinding(chatReq.System, chatReq.Tools); b != currentBinding {
					if err := e.sessions.BindReasoning(ctx, session.ID, b); err != nil {
						errMsg := err.Error()
						e.record(trace.Event{SessionID: session.ID, TurnID: turnID, Type: trace.EventError, Iteration: iteration, Error: errMsg})
						sendChunk(StreamChunk{Type: ChunkTypeError, Usage: &turnUsage, Error: errMsg})
						return
					}
					currentBinding = b
					chatReq.Messages = stripThinking(chatReq.Messages)
				}
			}

			e.logger().Debug("requesting completion stream", "iteration", iteration)
			callStart := time.Now()

			assistant := &llm.Message{Role: llm.RoleAssistant}
			var toolCalls []llm.ToolCall
			var callUsage *llm.Usage
			var sawDone bool
			var stop llm.StopReason
			var thinking strings.Builder
			// The reasoning blocks of this reply, whole and in order, when the
			// provider can replay them.
			var reasoning []llm.Part
			// One text part per iteration, not one per delta: a part per
			// token, replayed newline-joined by the OpenAI adapter, handed
			// the model its own answer with a newline between every token,
			// and it imitated the corruption on the next turn.
			var text strings.Builder
			textAt := -1

			// A stream that breaks before it produced anything is repeated:
			// nothing reached the consumer, so nothing is duplicated.
			for retries := 0; ; retries++ {
				events, err := e.provider.StreamChat(ctx, chatReq)
				if err != nil {
					e.logger().Error("stream call failed", "err", err)
					e.record(trace.Event{SessionID: session.ID, TurnID: turnID, Type: trace.EventError, Iteration: iteration, Error: err.Error()})
					sendChunk(StreamChunk{Type: ChunkTypeError, Usage: &turnUsage, Error: err.Error()})
					return
				}
				retry := false
				for ev := range events {
					switch ev.Kind {
					case llm.EventText:
						if textAt < 0 {
							textAt = len(assistant.Parts)
						}
						text.WriteString(ev.Text)
						e.record(trace.Event{SessionID: session.ID, TurnID: turnID, Type: trace.EventText, Iteration: iteration, Text: ev.Text})
						if !sendChunk(StreamChunk{Type: ChunkTypeText, Content: ev.Text}) {
							return
						}
					case llm.EventThinking:
						// Streamed for observability; persisted only when
						// reasoning preservation is on.
						thinking.WriteString(ev.Text)
						e.record(trace.Event{SessionID: session.ID, TurnID: turnID, Type: trace.EventThinking, Iteration: iteration, Text: ev.Text})
						if !sendChunk(StreamChunk{Type: ChunkTypeThinking, Content: ev.Text}) {
							return
						}
					case llm.EventReasoning:
						reasoning = append(reasoning, ev.Reasoning...)
					case llm.EventToolCalls:
						toolCalls = append(toolCalls, ev.ToolCalls...)
						for _, c := range ev.ToolCalls {
							assistant.Parts = append(assistant.Parts, llm.ToolCallPart(c))
						}
					case llm.EventError:
						// A cut-off stream still spent tokens: the provider books
						// an estimate on the error event, and hiding it would make
						// timed-out arms look free.
						if ev.Usage != nil {
							turnUsage.Add(*ev.Usage)
							e.sessions.AddUsage(session.ID, *ev.Usage)
						}
						if retries < e.streamRetries && ctx.Err() == nil && errors.Is(ev.Err, llm.ErrStreamInterrupted) &&
							text.Len() == 0 && thinking.Len() == 0 && len(reasoning) == 0 && len(toolCalls) == 0 {
							e.logger().Warn("model stream broke before any output; retrying", "err", ev.Err, "attempt", retries+1)
							e.record(trace.Event{SessionID: session.ID, TurnID: turnID, Type: trace.EventWarning, Iteration: iteration,
								Error: "stream interrupted before any output, retrying: " + ev.Err.Error(), Usage: ev.Usage})
							retry = true
							continue // the provider closes the channel after an error
						}
						e.logger().Error("error in provider stream", "err", ev.Err)
						e.record(trace.Event{SessionID: session.ID, TurnID: turnID, Type: trace.EventError, Iteration: iteration, Error: ev.Err.Error(), Usage: ev.Usage})
						sendChunk(StreamChunk{Type: ChunkTypeError, Usage: &turnUsage, Error: ev.Err.Error()})
						return
					case llm.EventDone:
						sawDone = true
						callUsage = ev.Usage
						stop = ev.StopReason
					}
				}

				if !retry {
					break
				}
				select {
				case <-time.After(e.retryDelay << retries):
				case <-ctx.Done():
				}
			}

			// A stream that closes without a done event did not finish: the
			// context ended or the connection dropped. Passing it on as an
			// empty answer would report a cancelled turn as a success.
			if !sawDone {
				errMsg := "the model stream ended without completing the call"
				if err := ctx.Err(); err != nil {
					errMsg = "turn interrupted: " + err.Error()
				}
				e.record(trace.Event{SessionID: session.ID, TurnID: turnID, Type: trace.EventError, Iteration: iteration, Error: errMsg})
				sendChunk(StreamChunk{Type: ChunkTypeError, Usage: &turnUsage, Error: errMsg})
				return
			}

			if text.Len() > 0 {
				assistant.Parts = insertPart(assistant.Parts, textAt, llm.TextPart(text.String()))
			}

			lastStop = stop
			// Only the call that answered the turn keeps its request: earlier
			// iterations of the same turn would store the history over and
			// over, and nothing offline needs a tool-decision prompt byte for
			// byte.
			var answeredRequest any
			if e.traceRequests && len(toolCalls) == 0 {
				// A copy: the next iteration reassigns the request's
				// fields, and a sink may keep the event past this call.
				snapshot := *chatReq
				answeredRequest = &snapshot
			}
			e.record(trace.Event{
				SessionID: session.ID, TurnID: turnID, Type: trace.EventLLMCall,
				Iteration: iteration, Usage: callUsage, StopReason: string(stop),
				LatencyMS: time.Since(callStart).Milliseconds(), Request: answeredRequest,
			})
			if callUsage != nil {
				turnUsage.Add(*callUsage)
				e.sessions.AddUsage(session.ID, *callUsage)
			}

			// Reasoning without an answer or a call: the model spent the
			// turn thinking and stopped (typically on a tool-call token the
			// request cannot honour). One reminder; the empty reply is not
			// kept. A max_tokens stop is a cut, not a silence: no retry.
			if len(toolCalls) == 0 && !hasText(assistant) && (thinking.Len() > 0 || len(reasoning) > 0) &&
				stop != llm.StopMaxTokens && !finalIteration {
				if silentNudges < 1 {
					silentNudges++
					forceNudge = silentTurnNudge
				} else {
					forceNudge = ""
				}
				if forceNudge != "" {
					e.record(trace.Event{SessionID: session.ID, TurnID: turnID, Type: trace.EventWarning, Iteration: iteration,
						Error: "the model ended its reply with reasoning and no answer; nudging once"})
					continue
				}
			}

			// Persist the turn's reasoning so reasoning-capable models can
			// continue their chain of thought on later turns. Blocks the
			// provider delivered whole go in unchanged: a signature covers the
			// full text, so a cut would make the block unreplayable. Text-only
			// reasoning (OpenAI-compatible endpoints) is bounded instead. A
			// reply that is reasoning alone is not an answer and is not kept.
			if e.preserveReasoning && (len(toolCalls) > 0 || hasText(assistant)) {
				if len(reasoning) > 0 {
					assistant.Parts = append(append([]llm.Part(nil), reasoning...), assistant.Parts...)
				} else if thinking.Len() > 0 {
					thought := thinking.String()
					if len(thought) > maxPreservedThinkingBytes {
						thought = cutUTF8(thought, maxPreservedThinkingBytes) + "\n[truncated]"
					}
					assistant.Parts = append([]llm.Part{llm.ThinkingPart(thought)}, assistant.Parts...)
				}
			}

			if len(assistant.Parts) == 0 {
				break // provider returned nothing — end the turn
			}

			if len(toolCalls) > 0 && finalIteration {
				// The backend ignored ToolChoice none. Executing the calls
				// would end the loop with no answer at all: keep the text if
				// the model gave one, otherwise fail the turn honestly.
				if !hasText(assistant) {
					errMsg := fmt.Sprintf("tool budget exhausted: the model requested %d tool call(s) on its final iteration instead of answering", len(toolCalls))
					e.record(trace.Event{SessionID: session.ID, TurnID: turnID, Type: trace.EventError, Iteration: iteration, Error: errMsg})
					sendChunk(StreamChunk{Type: ChunkTypeError, Usage: &turnUsage, Error: errMsg})
					return
				}
				e.record(trace.Event{SessionID: session.ID, TurnID: turnID, Type: trace.EventWarning, Iteration: iteration,
					Error: fmt.Sprintf("final iteration: dropped %d tool call(s) the backend sent despite tool_choice none; the text is the answer", len(toolCalls))})
				assistant = withoutToolCalls(assistant)
				toolCalls = nil
			}

			if len(toolCalls) == 0 {
				// Final answer: persist and finish. A max_tokens stop means
				// the text is cut off — the done chunk says so.
				if stop == llm.StopMaxTokens {
					e.record(trace.Event{SessionID: session.ID, TurnID: turnID, Type: trace.EventWarning, Iteration: iteration,
						Error: "answer truncated: the response hit the output token cap (max_tokens)"})
				}
				if !persist(e.sessions.AppendMessage(ctx, session.ID, assistant)) {
					return
				}
				e.record(trace.Event{SessionID: session.ID, TurnID: turnID, Type: trace.EventTurnEnd, Usage: &turnUsage, StopReason: string(stop)})
				sendChunk(StreamChunk{Type: ChunkTypeDone, Usage: &turnUsage, StopReason: stop})
				return
			}

			var resultParts []llm.Part
			if stop == llm.StopMaxTokens {
				e.record(trace.Event{SessionID: session.ID, TurnID: turnID, Type: trace.EventWarning, Iteration: iteration,
					Error: fmt.Sprintf("%d tool call(s) cut by max_tokens — not executed", len(toolCalls))})
				resultParts = truncatedCallResults(toolCalls)
			} else {
				e.logger().Debug("executing tool calls", "count", len(toolCalls))
				var toolUsage llm.Usage
				var toolErr error
				resultParts, toolUsage, toolErr = e.executeToolsParallel(ctx, tc, hc, turnApprover, toolCalls, out)
				if toolUsage != (llm.Usage{}) {
					turnUsage.Add(toolUsage)
					e.sessions.AddToolUsage(session.ID, toolUsage)
				}
				if toolErr != nil {
					errMsg := toolErr.Error()
					if err := ctx.Err(); err != nil {
						errMsg = "request cancelled: " + err.Error()
					}
					e.record(trace.Event{SessionID: session.ID, TurnID: turnID, Type: trace.EventError, Iteration: iteration, Error: errMsg})
					sendChunk(StreamChunk{Type: ChunkTypeError, Usage: &turnUsage, Error: errMsg})
					return
				}
				forceNudge = "" // a call arrived; the reminder is spent
			}

			toolMsg := &llm.Message{Role: llm.RoleTool, Parts: resultParts}
			if !persist(e.sessions.AppendMessages(ctx, session.ID, assistant, toolMsg)) {
				return
			}
		}

		e.record(trace.Event{SessionID: session.ID, TurnID: turnID, Type: trace.EventTurnEnd, Usage: &turnUsage, StopReason: string(lastStop)})
		sendChunk(StreamChunk{Type: ChunkTypeDone, Usage: &turnUsage, StopReason: lastStop})
	}()

	return out, nil
}

// Chat is a synchronous wrapper around StreamChat returning the full text.
func (e *AgentEngine) Chat(ctx context.Context, req ChatRequest) (string, error) {
	stream, err := e.StreamChat(ctx, req)
	if err != nil {
		return "", err
	}

	var sb strings.Builder
	finished := false
	for chunk := range stream {
		switch chunk.Type {
		case ChunkTypeError:
			return "", fmt.Errorf("agent error: %s", chunk.Error)
		case ChunkTypeText:
			sb.WriteString(chunk.Content)
		case ChunkTypeDone:
			finished = true
		}
	}
	if !finished {
		// The stream closed with neither done nor error: the context ended
		// with the buffer full. What was collected is a fragment, not an answer.
		if err := ctx.Err(); err != nil {
			return "", fmt.Errorf("agent: turn interrupted: %w", err)
		}
		return "", errors.New("agent: the stream ended without completing the turn")
	}
	return sb.String(), nil
}

// llmContextCompactor summarizes dropped history with the same provider that
// serves the turn.
func llmContextCompactor(provider llm.LLMProvider, model string) ContextCompactor {
	return func(ctx context.Context, dropped []llm.Message) (string, llm.Usage, error) {
		var b strings.Builder
		for _, m := range dropped {
			for _, p := range m.Parts {
				switch p.Kind {
				case llm.PartText:
					b.WriteString(string(m.Role))
					b.WriteString(": ")
					b.WriteString(p.Text)
					b.WriteString("\n")
				case llm.PartToolCall:
					b.WriteString(string(m.Role))
					b.WriteString(": [tool call ")
					b.WriteString(p.ToolCall.Name)
					b.WriteString("]\n")
				case llm.PartToolResult:
					b.WriteString("tool: ")
					if raw, err := json.Marshal(p.ToolResult.Content); err == nil {
						b.Write(raw)
					}
					b.WriteString("\n")
				}
			}
		}
		// The newest part of the transcript is the part worth keeping, and an
		// oversized transcript must not overflow the summarizer's own window.
		transcript := tailUTF8(b.String(), compactInputBytes)
		req := &llm.ChatRequest{
			Model:     model,
			MaxTokens: 512,
			Messages:  []llm.Message{*llm.UserText("Summarize the following earlier conversation in at most 8 sentences. Keep facts, decisions and constraints; omit chatter.\n\n" + transcript)},
		}
		text, usage, err := provider.Generate(ctx, req)
		return strings.TrimSpace(text), usage, err
	}
}

// compactKeepMessages is how many of the newest messages survive a compaction
// verbatim; the current request is among them.
const compactKeepMessages = 6

// compactInputBytes bounds the transcript the summarizer reads.
const compactInputBytes = 32 * 1024

// compactSession condenses the oldest part of a session's history into one
// summary and keeps the newest messages verbatim. The store is rewritten
// first, so a failed write leaves the old history in force. The turn holds the
// session, so nothing is appended between the snapshot and the rewrite. The
// returned usage is what the summarizer spent, a failed call included.
func (e *AgentEngine) compactSession(ctx context.Context, session *Session, turnID string) (llm.Usage, error) {
	sm := e.sessions

	sm.mu.RLock()
	sess, ok := sm.sessions[session.ID]
	if !ok {
		sm.mu.RUnlock()
		return llm.Usage{}, fmt.Errorf("agent: unknown session %q", session.ID)
	}
	history := append([]llm.Message(nil), sess.History...)
	sm.mu.RUnlock()

	// The cut moves back off a tool result onto the call that produced it:
	// providers reject a result that has no call in front of it.
	split := len(history) - compactKeepMessages
	for split > 0 && isToolResult(&history[split]) {
		split--
	}
	if split < 1 {
		return llm.Usage{}, errors.New("agent: nothing to compact")
	}

	summary, spent, err := e.compactor(ctx, history[:split])
	if err == nil && strings.TrimSpace(summary) == "" {
		err = errors.New("the summarizer returned an empty summary")
	}
	if err != nil {
		return spent, fmt.Errorf("agent: compaction failed: %w", err)
	}

	// The summary joins the first kept message when that is a user turn, so
	// roles keep alternating; otherwise it stands as a user message of its own.
	note := "Summary of earlier conversation: " + summary
	kept := history[split:]
	var rewritten []llm.Message
	if kept[0].Role == llm.RoleUser {
		first := llm.Message{Role: llm.RoleUser, Parts: append([]llm.Part{llm.TextPart(note + "\n\n")}, kept[0].Parts...)}
		rewritten = append([]llm.Message{first}, kept[1:]...)
	} else {
		rewritten = append([]llm.Message{*llm.UserText(note)}, kept...)
	}

	// The summary changes the prefix every kept block was produced under, so
	// no reasoning in the rewritten history can be replayed.
	rewritten = stripThinking(rewritten)

	if sm.store != nil {
		if err := sm.store.Compact(ctx, session.ID, rewritten); err != nil {
			return spent, fmt.Errorf("agent: compact session store: %w", err)
		}
	}

	sm.mu.Lock()
	defer sm.mu.Unlock()
	if sess, ok := sm.sessions[session.ID]; ok {
		sess.History = rewritten
		sess.windowStart = 0
		sess.LastInput, sess.LastOutput = 0, 0
		sess.UpdatedAt = time.Now()
	}
	e.record(trace.Event{SessionID: session.ID, TurnID: turnID, Type: trace.EventWarning,
		Error: "history compacted at the context limit", Text: summary})
	return spent, nil
}

// requestBinding fingerprints what reasoning blocks are bound to: the system
// prompt and the tool set as they are sent. ToolDef marshals deterministically
// (map keys come out sorted), so equal requests give equal fingerprints.
func requestBinding(system string, tools []llm.ToolDef) string {
	raw, _ := json.Marshal(tools)
	sum := sha256.Sum256(append([]byte(system+"\x00"), raw...))
	return hex.EncodeToString(sum[:])
}

// tailUTF8 returns the last n bytes of s, moved forward to a character boundary.
func tailUTF8(s string, n int) string {
	if len(s) <= n {
		return s
	}
	i := len(s) - n
	for i < len(s) && !utf8.RuneStart(s[i]) {
		i++
	}
	return s[i:]
}

// cutUTF8 shortens s to at most n bytes without splitting a character.
func cutUTF8(s string, n int) string {
	if len(s) <= n {
		return s
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
}

// logger returns the configured logger; an engine built as a bare struct
// (tests) logs to the default one.
func (e *AgentEngine) logger() *slog.Logger {
	if e.log != nil {
		return e.log
	}
	return slog.Default()
}
