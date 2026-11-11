// Package openai implements the llm.LLMProvider port for OpenAI and any
// OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, self-hosted
// runtimes) via a configurable base URL.
package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"github.com/openai/openai-go/v2/packages/param"
	"github.com/openai/openai-go/v2/packages/respjson"
	"github.com/openai/openai-go/v2/shared"

	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// defaultMaxTokens caps responses when the request leaves it unset.
const defaultMaxTokens = 8192

// Provider talks to OpenAI-compatible chat completion APIs.
type Provider struct {
	client              openai.Client
	model               string // default model for requests without an override
	maxCompletionTokens bool
	sampling            Sampling
}

// Sampling holds the knobs sent with every request: agent turns, sub-agents
// and structured calls all run one model under one setting. Unset fields
// leave the backend's default.
type Sampling struct {
	Temperature *float64
	TopP        *float64
	// TopK is not an OpenAI field; vLLM, SGLang and OpenRouter accept it.
	TopK *int
	// ReasoningEffort goes out as reasoning_effort; the values a model
	// accepts are its own (e.g. low / medium / high / xhigh).
	ReasoningEffort string
	// ExtraBody is merged into the request body last, so it wins over the
	// fields above: the place for backend-specific fields such as vLLM's
	// chat_template_kwargs or OpenRouter's reasoning and provider routing.
	ExtraBody map[string]any
}

// Config configures a provider.
type Config struct {
	// APIKey is the bearer token. Required for OpenAI; local runtimes
	// often accept any non-empty value.
	APIKey string
	// BaseURL points at an OpenAI-compatible server ("" = api.openai.com).
	BaseURL string
	// Model is the default model id (e.g. "gpt-5", "qwen3:32b").
	Model string
	// MaxCompletionTokens sends the response cap as max_completion_tokens
	// instead of max_tokens. api.openai.com needs it (reasoning models
	// reject max_tokens); OpenAI-compatible servers vary, so it is opt-in
	// and set by the bootstrap for the openai kind.
	MaxCompletionTokens bool
	// Sampling applies to every request.
	Sampling Sampling
}

// maxRetries bounds the SDK's automatic retries of rate-limited and 5xx requests.
const maxRetries = 8

// New creates an OpenAI-compatible provider.
func New(cfg Config) (*Provider, error) {
	// The SDK retries 429/5xx with a short backoff (2 tries by default); a
	// per-minute rate limit outlasts that and killed hour-long runs, so allow
	// more tries. The SDK honours Retry-After.
	opts := []option.RequestOption{option.WithMaxRetries(maxRetries)}
	if cfg.APIKey != "" {
		opts = append(opts, option.WithAPIKey(cfg.APIKey))
	}
	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}

	return &Provider{
		client:              openai.NewClient(opts...),
		model:               cfg.Model,
		maxCompletionTokens: cfg.MaxCompletionTokens,
		sampling:            cfg.Sampling,
	}, nil
}

// Name implements llm.LLMProvider.
func (p *Provider) Name() string { return "openai" }

// modelID resolves the request model or the provider default.
func (p *Provider) modelID(req *llm.ChatRequest) (string, error) {
	if req.Model != "" {
		return req.Model, nil
	}
	if p.model != "" {
		return p.model, nil
	}
	return "", llm.ErrNoModel
}

// convertTools maps JSON-Schema tool defs onto OpenAI function declarations.
func convertTools(defs []llm.ToolDef) []openai.ChatCompletionToolUnionParam {
	if len(defs) == 0 {
		return nil
	}
	tools := make([]openai.ChatCompletionToolUnionParam, 0, len(defs))
	for _, d := range defs {
		tools = append(tools, openai.ChatCompletionToolUnionParam{
			OfFunction: &openai.ChatCompletionFunctionToolParam{
				Function: shared.FunctionDefinitionParam{
					Name:        d.Name,
					Description: param.NewOpt(d.Description),
					Parameters:  shared.FunctionParameters(d.Parameters),
				},
			},
		})
	}
	return tools
}

// convertMessages maps the neutral conversation onto OpenAI messages.
// Consecutive tool calls merge into one assistant turn; each tool result
// becomes a "tool"-role message keyed by tool_call_id. With
// PreserveReasoning, assistant thinking parts ride back on the assistant
// message as reasoning_details (the OpenRouter/OpenAI reasoning round-trip).
func convertMessages(req *llm.ChatRequest) []openai.ChatCompletionMessageParamUnion {
	out := make([]openai.ChatCompletionMessageParamUnion, 0, len(req.Messages)+1)
	if req.System != "" {
		out = append(out, openai.SystemMessage(req.System))
	}

	var pendingCalls []openai.ChatCompletionMessageToolCallUnionParam
	var pendingThinking []string

	// replayThinking attaches the collected reasoning to an assistant
	// message param as extra fields (the SDK has no typed field for it).
	replayThinking := func(p *openai.ChatCompletionAssistantMessageParam) {
		if len(pendingThinking) == 0 || !req.PreserveReasoning {
			return
		}
		details := make([]map[string]any, 0, len(pendingThinking))
		for _, text := range pendingThinking {
			details = append(details, map[string]any{"type": "reasoning.text", "text": text})
		}
		p.SetExtraFields(map[string]any{"reasoning_details": details})
		pendingThinking = nil
	}

	flushAssistant := func() {
		if len(pendingCalls) == 0 {
			return
		}
		assistant := &openai.ChatCompletionAssistantMessageParam{
			ToolCalls: pendingCalls,
		}
		replayThinking(assistant)
		out = append(out, openai.ChatCompletionMessageParamUnion{OfAssistant: assistant})
		pendingCalls = nil
	}

	for _, msg := range req.Messages {
		var textParts []string
		var imageParts []openai.ChatCompletionContentPartUnionParam
		for _, part := range msg.Parts {
			switch part.Kind {
			case llm.PartText:
				textParts = append(textParts, part.Text)
			case llm.PartImage:
				// Vision input rides on user messages as content parts.
				if msg.Role == llm.RoleUser {
					imageParts = append(imageParts, openai.ChatCompletionContentPartUnionParam{
						OfImageURL: &openai.ChatCompletionContentPartImageParam{
							ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
								URL: imageURL(part.Image),
							},
						},
					})
				}
			case llm.PartToolCall:
				args, _ := marshalArgs(part.ToolCall.Args)
				pendingCalls = append(pendingCalls, openai.ChatCompletionMessageToolCallUnionParam{
					OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
						ID: part.ToolCall.ID,
						Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
							Name:      part.ToolCall.Name,
							Arguments: args,
						},
					},
				})
			case llm.PartToolResult:
				// A tool result terminates the preceding assistant turn.
				flushAssistant()
				content, _ := marshalArgs(part.ToolResult.Content)
				out = append(out, openai.ToolMessage(content, part.ToolResult.CallID))
			case llm.PartThinking:
				// Collected here, replayed when the assistant turn is
				// emitted (see replayThinking). Empty text has nothing to
				// replay: a blank detail would only be noise on the wire.
				if part.Text != "" {
					pendingThinking = append(pendingThinking, part.Text)
				}
			}
		}

		// A user message with images becomes a multipart content array
		// (text part first, then the images).
		if msg.Role == llm.RoleUser && len(imageParts) > 0 {
			var parts []openai.ChatCompletionContentPartUnionParam
			if len(textParts) > 0 {
				parts = append(parts, openai.ChatCompletionContentPartUnionParam{
					OfText: &openai.ChatCompletionContentPartTextParam{
						Text: strings.Join(textParts, ""),
					},
				})
			}
			parts = append(parts, imageParts...)
			out = append(out, openai.ChatCompletionMessageParamUnion{
				OfUser: &openai.ChatCompletionUserMessageParam{
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfArrayOfContentParts: parts,
					},
				},
			})
			continue
		}

		if len(textParts) > 0 {
			content := strings.Join(textParts, "")
			switch msg.Role {
			case llm.RoleUser, llm.RoleTool:
				out = append(out, openai.UserMessage(content))
			case llm.RoleAssistant:
				// Text and tool calls of one turn are one message: the API
				// wants a tool_calls message followed directly by its
				// tool results.
				assistant := &openai.ChatCompletionAssistantMessageParam{ToolCalls: pendingCalls}
				pendingCalls = nil
				assistant.Content = openai.ChatCompletionAssistantMessageParamContentUnion{}
				assistant.Content.OfString = param.NewOpt(content)
				replayThinking(assistant)
				out = append(out, openai.ChatCompletionMessageParamUnion{OfAssistant: assistant})
			}
		} else if msg.Role == llm.RoleAssistant {
			// A thinking-only assistant turn still needs its reasoning.
			flushAssistant()
			if len(pendingThinking) > 0 {
				assistant := &openai.ChatCompletionAssistantMessageParam{}
				replayThinking(assistant)
				out = append(out, openai.ChatCompletionMessageParamUnion{OfAssistant: assistant})
			}
		}
	}
	flushAssistant()

	return out
}

// imageURL renders an image as the OpenAI image_url form: a data URL for
// inline bytes, the reference URL otherwise.
func imageURL(img *llm.ImageData) string {
	if img.Data != "" {
		return "data:" + img.MediaType + ";base64," + img.Data
	}
	return img.URL
}

// marshalArgs serializes a tool argument/result payload to a JSON string.
func marshalArgs(v any) (string, error) {
	if s, ok := v.(string); ok {
		return s, nil
	}
	b, err := json.Marshal(v)
	if err != nil {
		return "{}", err
	}
	return string(b), nil
}

// baseParams builds the request shared by every port method: model,
// messages, the response cap in the field this backend expects, the
// configured sampling, and the tool set shaped by the request's ToolChoice.
func (p *Provider) baseParams(model string, req *llm.ChatRequest) openai.ChatCompletionNewParams {
	maxTokens := int64(req.MaxTokens)
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
	}
	params := openai.ChatCompletionNewParams{
		Model:    model,
		Messages: convertMessages(req),
	}
	if p.maxCompletionTokens {
		params.MaxCompletionTokens = param.NewOpt(maxTokens)
	} else {
		params.MaxTokens = param.NewOpt(maxTokens)
	}
	p.applySampling(&params, req.ExtraBody)

	// ToolChoiceNone omits the tools: OpenAI-compatible backends accept a
	// history with tool calls and no declared tools, and omission is
	// honored everywhere, while tool_choice "none" is not (verified live
	// on OpenRouter). tool_choice is only sent next to declared tools —
	// it is a 400 on its own.
	if req.ToolChoice.Mode == llm.ToolChoiceNone || len(req.Tools) == 0 {
		return params
	}
	params.Tools = convertTools(req.Tools)
	switch req.ToolChoice.Mode {
	case llm.ToolChoiceRequired:
		params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: param.NewOpt("required")}
	case llm.ToolChoiceTool:
		params.ToolChoice = openai.ToolChoiceOptionFunctionToolChoice(
			openai.ChatCompletionNamedToolChoiceFunctionParam{Name: req.ToolChoice.Name})
	}
	return params
}

// applySampling sets the configured sampling knobs on a request. Request
// extras (req.ExtraBody) are merged last, so a single call can override a
// configured field — a caller can cap one call's thinking
// budget this way — without changing the run's sampling.
func (p *Provider) applySampling(params *openai.ChatCompletionNewParams, overrides map[string]any) {
	s := p.sampling
	if s.Temperature != nil {
		params.Temperature = param.NewOpt(*s.Temperature)
	}
	if s.TopP != nil {
		params.TopP = param.NewOpt(*s.TopP)
	}
	if s.ReasoningEffort != "" {
		params.ReasoningEffort = shared.ReasoningEffort(s.ReasoningEffort)
	}
	extra := map[string]any{}
	if s.TopK != nil {
		extra["top_k"] = *s.TopK
	}
	for k, v := range s.ExtraBody {
		extra[k] = v
	}
	for k, v := range overrides {
		extra[k] = v // a single call overriding the run's configuration
	}
	if len(extra) > 0 {
		params.SetExtraFields(extra)
	}
}

// stopReason maps an OpenAI finish_reason onto the neutral enum.
func stopReason(finish string) llm.StopReason {
	switch finish {
	case "":
		return llm.StopUnknown
	case "stop":
		return llm.StopEnd
	case "tool_calls", "function_call":
		return llm.StopToolUse
	case "length":
		return llm.StopMaxTokens
	default:
		return llm.StopOther
	}
}

// streamState accumulates one streamed completion. The SDK accumulator is
// not used: it drops tool calls that arrive whole in one chunk or next to
// an empty content field (both seen on OpenAI-compatible servers), stops
// accumulating when a server varies chunk ids, and sums usage across
// chunks, which over-counts servers that report cumulative usage.
type streamState struct {
	calls    []llm.ToolCall
	args     []strings.Builder
	slotByIx map[int64]int // wire index -> slot of the call it currently feeds
	finish   string
	usage    openai.CompletionUsage
	// outChars counts the bytes the stream produced (text, reasoning, tool
	// arguments) — the basis of the usage estimate when no usage block ever
	// arrives (cancelled stream, backend that ignores include_usage).
	outChars int
}

// estimate books output tokens for a stream that ended without a usage
// block. Four bytes per token is the usual rough ratio; the count is marked
// Estimated so it is never mistaken for a server-reported figure. A stream
// that produced nothing books nothing.
func (st *streamState) estimate() llm.Usage {
	if st.outChars == 0 {
		return llm.Usage{}
	}
	n := st.outChars / 4
	if n < 1 {
		n = 1
	}
	return llm.Usage{OutputTokens: n, Estimated: true}
}

// mergeUsage prefers the backend's own count whenever a usage block arrived;
// the stream estimate is only for streams that ended before reporting. The
// estimate is chars/4, which sits ten to thirty percent off the real count on
// prose — taking it over a reported figure silently inflates what a call
// spent, and budgets and cost reports are built on exactly this number.
func (st *streamState) mergeUsage() *llm.Usage {
	if u := usageOf(st.usage); u != nil {
		return u
	}
	est := st.estimate()
	if est.OutputTokens == 0 {
		return nil
	}
	return &est
}

// addToolDelta merges one tool-call delta. Calls are keyed by their wire
// index; a new id on an index that already holds a different call opens a
// new slot — servers that omit the index send whole calls with distinct
// ids, and must not be glued into one.
func (st *streamState) addToolDelta(d openai.ChatCompletionChunkChoiceDeltaToolCall) {
	if st.slotByIx == nil {
		st.slotByIx = map[int64]int{}
	}
	slot, seen := st.slotByIx[d.Index]
	if !seen || (d.ID != "" && st.calls[slot].ID != "" && d.ID != st.calls[slot].ID) {
		st.calls = append(st.calls, llm.ToolCall{})
		st.args = append(st.args, strings.Builder{})
		slot = len(st.calls) - 1
		st.slotByIx[d.Index] = slot
	}
	if d.ID != "" {
		st.calls[slot].ID = d.ID
	}
	st.calls[slot].Name += d.Function.Name
	st.args[slot].WriteString(d.Function.Arguments)
	st.outChars += len(d.Function.Name) + len(d.Function.Arguments)
}

// toolCalls returns the completed calls with decoded arguments.
func (st *streamState) toolCalls() []llm.ToolCall {
	out := make([]llm.ToolCall, 0, len(st.calls))
	for i, c := range st.calls {
		if c.Name == "" {
			continue // an index that only ever carried noise
		}
		c.Args = parseArgs(st.args[i].String())
		out = append(out, c)
	}
	return out
}

// StreamChat implements llm.LLMProvider. Text and thinking deltas stream as
// they arrive; tool calls are emitted once, aggregated, when the stream
// ends — the engine acts on them only after the turn anyway.
func (p *Provider) StreamChat(ctx context.Context, req *llm.ChatRequest) (<-chan llm.StreamEvent, error) {
	model, err := p.modelID(req)
	if err != nil {
		return nil, err
	}

	params := p.baseParams(model, req)
	// Without include_usage api.openai.com and vLLM stream no token counts
	// at all — cost accounting and the context guard would silently see 0.
	params.StreamOptions = openai.ChatCompletionStreamOptionsParam{IncludeUsage: param.NewOpt(true)}

	stream := p.client.Chat.Completions.NewStreaming(ctx, params)

	out := make(chan llm.StreamEvent, 32)
	go func() {
		defer close(out)
		defer stream.Close()

		// A consumer that stopped reading (cancelled turn) must not leave
		// this goroutine blocked on a full channel.
		emit := func(ev llm.StreamEvent) bool {
			select {
			case out <- ev:
				return true
			case <-ctx.Done():
				return false
			}
		}
		// emitFinal delivers the last event even when the context is done:
		// the usage estimate of a cancelled stream rides on it, and dropping
		// it books a timed-out call as free.
		emitFinal := func(ev llm.StreamEvent) {
			select {
			case out <- ev:
				return
			default:
			}
			select {
			case out <- ev:
			case <-time.After(250 * time.Millisecond):
			}
		}

		var st streamState
		for stream.Next() {
			chunk := stream.Current()
			// Usage arrives once (final chunk) or cumulatively on every
			// chunk depending on the server: the latest report wins.
			if chunk.Usage.PromptTokens > 0 || chunk.Usage.CompletionTokens > 0 {
				st.usage = chunk.Usage
			}
			for _, choice := range chunk.Choices {
				if choice.Index != 0 {
					continue // n>1 is never requested
				}
				if choice.FinishReason != "" {
					st.finish = choice.FinishReason
				}
				for _, d := range choice.Delta.ToolCalls {
					st.addToolDelta(d)
				}
				if choice.Delta.Content != "" {
					st.outChars += len(choice.Delta.Content)
					if !emit(llm.StreamEvent{Kind: llm.EventText, Text: choice.Delta.Content}) {
						return
					}
				}
				if choice.Delta.Refusal != "" {
					st.outChars += len(choice.Delta.Refusal)
					if !emit(llm.StreamEvent{Kind: llm.EventText, Text: choice.Delta.Refusal}) {
						return
					}
				}
				// Reasoning deltas ("reasoning_content" / "reasoning" on
				// OpenAI-compatible servers like OpenRouter) are not typed
				// by the SDK — they ride in the raw extra fields.
				if s := reasoningDelta(choice.Delta.JSON.ExtraFields); s != "" {
					st.outChars += len(s)
					if !emit(llm.StreamEvent{Kind: llm.EventThinking, Text: s}) {
						return
					}
				}
			}
		}
		if err := stream.Err(); err != nil {
			// No usage chunk is coming: book what the stream produced, so a
			// cancelled or dropped call is not accounted as zero tokens.
			emitFinal(llm.StreamEvent{
				Kind:  llm.EventError,
				Err:   streamError(err),
				Usage: st.mergeUsage(),
			})
			return
		}

		if calls := st.toolCalls(); len(calls) > 0 {
			if !emit(llm.StreamEvent{Kind: llm.EventToolCalls, ToolCalls: calls}) {
				return
			}
		}
		emit(llm.StreamEvent{Kind: llm.EventDone, Usage: st.mergeUsage(), StopReason: stopReason(st.finish)})
	}()

	return out, nil
}

// streamError wraps a stream failure, marking a broken connection (as
// opposed to an error answer from the API) with llm.ErrStreamInterrupted.
func streamError(err error) error {
	var apiErr *openai.Error
	if errors.As(err, &apiErr) || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return fmt.Errorf("openai: stream: %w", err)
	}
	return fmt.Errorf("openai: stream: %w: %w", llm.ErrStreamInterrupted, err)
}

// usageOf converts a wire usage block (nil when the backend reported none).
func usageOf(u openai.CompletionUsage) *llm.Usage {
	if u.PromptTokens == 0 && u.CompletionTokens == 0 {
		return nil
	}
	return &llm.Usage{
		InputTokens:     int(u.PromptTokens),
		OutputTokens:    int(u.CompletionTokens),
		CacheReadTokens: int(u.PromptTokensDetails.CachedTokens),
	}
}

// addUsage folds a wire usage block into an accumulator.
func addUsage(acc *llm.Usage, u openai.CompletionUsage) {
	if v := usageOf(u); v != nil {
		acc.Add(*v)
	}
}

// reasoningDelta extracts a reasoning delta the SDK does not type
// (OpenRouter and compatible servers stream it as reasoning_content or
// reasoning inside the delta). Null or empty fields are skipped, so a
// "reasoning_content": null next to a populated "reasoning" still reads.
func reasoningDelta(extra map[string]respjson.Field) string {
	for _, key := range []string{"reasoning_content", "reasoning"} {
		f, ok := extra[key]
		if !ok {
			continue
		}
		raw := f.Raw()
		if raw == "" || raw == "null" {
			continue
		}
		var s string
		if err := json.Unmarshal([]byte(raw), &s); err == nil {
			if s != "" {
				return s
			}
			continue
		}
		return raw
	}
	return ""
}

// parseArgs decodes a tool-call argument JSON string, tolerating empty or
// malformed payloads (the model gets an error result, not a crash).
func parseArgs(raw string) map[string]any {
	args := map[string]any{}
	if raw == "" {
		return args
	}
	if err := json.Unmarshal([]byte(raw), &args); err != nil {
		return map[string]any{"_raw": raw}
	}
	return args
}

// Generate implements llm.LLMProvider (non-streaming completion).
func (p *Provider) Generate(ctx context.Context, req *llm.ChatRequest) (string, llm.Usage, error) {
	model, err := p.modelID(req)
	if err != nil {
		return "", llm.Usage{}, err
	}

	params := p.baseParams(model, req)
	params.Tools, params.ToolChoice = nil, openai.ChatCompletionToolChoiceOptionUnionParam{}
	completion, err := p.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return "", llm.Usage{}, fmt.Errorf("openai: generate: %w", err)
	}
	var usage llm.Usage
	addUsage(&usage, completion.Usage)
	if len(completion.Choices) == 0 {
		return "", usage, nil
	}
	return completion.Choices[0].Message.Content, usage, nil
}

// maxJSONWhitespaceRun bounds the whitespace a constrained reply may emit
// in a row. Grammar-constrained decoding on some backends falls into an
// endless run of spaces, tabs and newlines inside the JSON object — legal
// under the grammar, so it only ends at max_tokens, minutes later (seen
// live on qwen/qwen3-32b via OpenRouter: 7 of 10 requests). Raw tabs and
// newlines cannot occur inside JSON strings and no reply needs this much
// indentation, so a longer run marks the reply as degenerate.
const maxJSONWhitespaceRun = 256

// errDegenerateJSON marks a native attempt cut off by the whitespace bound.
var errDegenerateJSON = errors.New("constrained reply degenerated into a whitespace run")

// GenerateStructured implements llm.LLMProvider with a two-step format
// chain, so weak or self-hosted backends still produce JSON:
//  1. native structured outputs (response_format json_schema), streamed so
//     a degenerate whitespace loop is cut off as it starts;
//  2. if the backend rejects the format, degenerates, or returns
//     undecodable text, one retry without response_format — the schema
//     rides in the prompt and the reply goes through a lenient parser
//     (strips markdown fences and surrounding prose).
//
// Cancellation and deadline errors are never retried: the caller gave up.
// The returned usage covers both attempts.
func (p *Provider) GenerateStructured(ctx context.Context, req *llm.ChatRequest, schema llm.Schema, out any) (llm.Usage, error) {
	model, err := p.modelID(req)
	if err != nil {
		return llm.Usage{}, err
	}

	params := p.baseParams(model, req)
	params.Tools, params.ToolChoice = nil, openai.ChatCompletionToolChoiceOptionUnionParam{}
	params.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
		OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
			JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:   "response",
				Schema: schema,
			},
		},
	}

	content, usage, err := p.nativeStructured(ctx, params)
	if err != nil {
		retryable := errors.Is(err, errDegenerateJSON) || isRetryableFormatError(err)
		if ctx.Err() != nil || !retryable {
			return usage, fmt.Errorf("openai: structured generation failed: %w", err)
		}
		content = "" // rejected or degenerate: the prompt-guided attempt below
	}

	// Native mode produced no decodable JSON (format ignored, prose, a
	// truncated object): prompt-guided attempts. Backends that ignore
	// response_format answer in prose on some calls and in JSON on others,
	// so one prompt-guided try is not enough.
	if raw := llm.ExtractJSON(content); raw != "" && json.Unmarshal([]byte(raw), out) == nil {
		return usage, nil
	}
	var lastErr error
	for range structuredPromptAttempts {
		if ctx.Err() != nil {
			break
		}
		completion, err := p.structuredByPrompt(ctx, model, req, schema)
		if err != nil {
			return usage, err
		}
		addUsage(&usage, completion.Usage)
		if len(completion.Choices) == 0 {
			lastErr = errors.New("openai: empty structured response")
			continue
		}
		content = completion.Choices[0].Message.Content
		raw := llm.ExtractJSON(content)
		if raw == "" {
			lastErr = fmt.Errorf("openai: decode structured response %q: no JSON found", content)
			continue
		}
		if err := json.Unmarshal([]byte(raw), out); err != nil {
			lastErr = fmt.Errorf("openai: decode structured response %q: %w", raw, err)
			continue
		}
		return usage, nil
	}
	if lastErr == nil {
		lastErr = ctx.Err()
	}
	return usage, lastErr
}

// structuredPromptAttempts bounds the prompt-guided tries after a native
// attempt that yielded no decodable JSON.
const structuredPromptAttempts = 3

// nativeStructured streams the response_format attempt and returns its
// text. A whitespace run past maxJSONWhitespaceRun aborts the request with
// errDegenerateJSON.
func (p *Provider) nativeStructured(ctx context.Context, params openai.ChatCompletionNewParams) (string, llm.Usage, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	params.StreamOptions = openai.ChatCompletionStreamOptionsParam{IncludeUsage: param.NewOpt(true)}
	stream := p.client.Chat.Completions.NewStreaming(ctx, params)
	defer stream.Close()

	var (
		text   strings.Builder
		wire   openai.CompletionUsage
		run    int // trailing whitespace characters in text
		deltas int // content and reasoning deltas seen
	)
	for stream.Next() {
		chunk := stream.Current()
		if chunk.Usage.PromptTokens > 0 || chunk.Usage.CompletionTokens > 0 {
			wire = chunk.Usage
		}
		for _, choice := range chunk.Choices {
			if choice.Index != 0 {
				continue
			}
			if choice.Delta.Content != "" || reasoningDelta(choice.Delta.JSON.ExtraFields) != "" {
				deltas++
			}
			for _, r := range choice.Delta.Content {
				if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
					run++
				} else {
					run = 0
				}
			}
			text.WriteString(choice.Delta.Content)
			if run > maxJSONWhitespaceRun {
				// The server reports usage only when a stream ends. The
				// delta count (about a token each) stands in for the
				// output, so the aborted attempt is not booked as free;
				// its input tokens stay unknown.
				return text.String(), llm.Usage{OutputTokens: deltas}, errDegenerateJSON
			}
		}
	}
	var usage llm.Usage
	addUsage(&usage, wire)
	if err := stream.Err(); err != nil {
		return "", usage, err
	}
	return text.String(), usage, nil
}

// structuredByPrompt asks for JSON in the prompt (no response_format) —
// the compatibility path for backends without structured-output support.
func (p *Provider) structuredByPrompt(ctx context.Context, model string, req *llm.ChatRequest, schema llm.Schema) (*openai.ChatCompletion, error) {
	schemaJSON, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal schema: %w", err)
	}
	params := p.baseParams(model, llm.WithSchemaNote(req, string(schemaJSON)))
	params.Tools, params.ToolChoice = nil, openai.ChatCompletionToolChoiceOptionUnionParam{}

	completion, err := p.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("openai: structured generation (prompt-guided) failed: %w", err)
	}
	return completion, nil
}

// isRetryableFormatError reports whether a structured-output failure is
// worth a format downgrade rather than a hard fail: backends without
// json_schema support reject the parameter outright.
func isRetryableFormatError(err error) bool {
	msg := err.Error()
	return strings.Contains(msg, "response_format") ||
		strings.Contains(msg, "json_schema") ||
		strings.Contains(msg, "not supported") ||
		strings.Contains(msg, "unsupported") ||
		strings.Contains(msg, "400")
}
