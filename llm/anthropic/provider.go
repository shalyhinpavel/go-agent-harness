// Package anthropic implements the llm.LLMProvider port for the Anthropic
// Messages API (Claude models).
package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// defaultMaxTokens caps responses when the request leaves it unset.
const defaultMaxTokens = 16000

// Provider talks to the Anthropic Messages API.
type Provider struct {
	client anthropicsdk.Client
	model  string // default model for requests without an override
	cfg    Config
}

// Config configures a provider.
type Config struct {
	// APIKey is the Anthropic API key ("" = ANTHROPIC_API_KEY env).
	APIKey string
	// Model is the default model id (e.g. "claude-sonnet-5-5").
	Model string
	// BaseURL points at an Anthropic-compatible endpoint: a gateway, a
	// proxy, a test server ("" = api.anthropic.com).
	BaseURL string
	// Temperature and TopP apply to every request when set. Some models
	// reject one of them, or both together: leave nil what the model
	// does not take.
	Temperature *float64
	TopP        *float64
	// MaxRetries bounds the SDK's automatic retries of rate-limited and 5xx
	// requests: 0 = DefaultMaxRetries, negative = no retries.
	MaxRetries int
}

// DefaultMaxRetries is the retry bound when Config.MaxRetries is 0. A
// per-minute rate limit outlasts the SDK's own default of two tries.
const DefaultMaxRetries = 8

// New creates an Anthropic provider.
func New(cfg Config) (*Provider, error) {
	retries := cfg.MaxRetries
	switch {
	case retries == 0:
		retries = DefaultMaxRetries
	case retries < 0:
		retries = 0
	}
	opts := []option.RequestOption{option.WithMaxRetries(retries)}
	if cfg.APIKey != "" {
		opts = append(opts, option.WithAPIKey(cfg.APIKey))
	}
	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}
	return &Provider{
		client: anthropicsdk.NewClient(opts...),
		model:  cfg.Model,
		cfg:    cfg,
	}, nil
}

// Name implements llm.LLMProvider.
func (p *Provider) Name() string { return "anthropic" }

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

// convertTools maps JSON-Schema tool defs onto Anthropic tool params.
func convertTools(defs []llm.ToolDef) []anthropicsdk.ToolUnionParam {
	if len(defs) == 0 {
		return nil
	}
	tools := make([]anthropicsdk.ToolUnionParam, 0, len(defs))
	for _, d := range defs {
		schema := anthropicsdk.ToolInputSchemaParam{
			Properties: d.Parameters["properties"],
			Required:   requiredList(d.Parameters["required"]),
		}
		// Everything beyond type/properties/required (additionalProperties,
		// $defs, ...) travels as-is instead of being silently dropped.
		for k, v := range d.Parameters {
			switch k {
			case "type", "properties", "required":
				continue
			}
			if schema.ExtraFields == nil {
				schema.ExtraFields = map[string]any{}
			}
			schema.ExtraFields[k] = v
		}
		tools = append(tools, anthropicsdk.ToolUnionParam{
			OfTool: &anthropicsdk.ToolParam{
				Name:        d.Name,
				Description: anthropicsdk.String(d.Description),
				InputSchema: schema,
			},
		})
	}
	return tools
}

// requiredList accepts the required list as written in Go ([]string) or
// decoded from JSON/YAML ([]any).
func requiredList(v any) []string {
	switch list := v.(type) {
	case []string:
		return list
	case []any:
		out := make([]string, 0, len(list))
		for _, item := range list {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}

// convertMessages maps the neutral conversation onto Anthropic messages.
// The system prompt travels separately; tool results ride in user turns as
// tool_result blocks; thinking parts go back only when the request asks for
// reasoning to be preserved.
func convertMessages(req *llm.ChatRequest) []anthropicsdk.MessageParam {
	out := make([]anthropicsdk.MessageParam, 0, len(req.Messages))

	appendBlocks := func(role llm.Role, blocks []anthropicsdk.ContentBlockParamUnion) {
		if len(blocks) == 0 {
			return
		}
		if role == llm.RoleAssistant {
			out = append(out, anthropicsdk.MessageParam{
				Role:    anthropicsdk.MessageParamRoleAssistant,
				Content: blocks,
			})
		} else {
			out = append(out, anthropicsdk.NewUserMessage(blocks...))
		}
	}

	for _, msg := range req.Messages {
		var blocks []anthropicsdk.ContentBlockParamUnion
		for _, part := range msg.Parts {
			switch part.Kind {
			case llm.PartText:
				blocks = append(blocks, anthropicsdk.NewTextBlock(part.Text))
			case llm.PartImage:
				// Vision input: base64 inline bytes or a reference URL.
				if part.Image.Data != "" {
					blocks = append(blocks, anthropicsdk.NewImageBlockBase64(part.Image.MediaType, part.Image.Data))
				} else if part.Image.URL != "" {
					blocks = append(blocks, anthropicsdk.ContentBlockParamUnion{
						OfImage: &anthropicsdk.ImageBlockParam{
							Source: anthropicsdk.ImageBlockParamSourceUnion{
								OfURL: &anthropicsdk.URLImageSourceParam{URL: part.Image.URL},
							},
						},
					})
				}
			case llm.PartToolCall:
				// Text and every tool_use of a turn stay in one assistant
				// message, as the Messages API lays tool use out.
				raw, _ := json.Marshal(part.ToolCall.Args)
				blocks = append(blocks, anthropicsdk.NewToolUseBlock(part.ToolCall.ID, part.ToolCall.Name, string(raw)))
			case llm.PartToolResult:
				// All results of a batch share one user message.
				content, err := json.Marshal(part.ToolResult.Content)
				if err != nil {
					content = []byte(`"result serialization error"`)
				}
				blocks = append(blocks, anthropicsdk.NewToolResultBlock(
					part.ToolResult.CallID, string(content), part.ToolResult.IsError))
			case llm.PartThinking:
				// A block goes back exactly as the API wrote it: the API
				// rejects a modified one, and an unsigned one (from an old
				// session, or reasoning kept from another provider) has
				// nothing it can verify, so it is left out.
				if !req.PreserveReasoning {
					break
				}
				switch {
				case part.Redacted != "":
					blocks = append(blocks, anthropicsdk.NewRedactedThinkingBlock(part.Redacted))
				case part.Signature != "":
					blocks = append(blocks, anthropicsdk.NewThinkingBlock(part.Signature, part.Text))
				}
			}
		}
		appendBlocks(msg.Role, blocks)
	}

	return out
}

// params builds base request parameters shared by all three port methods.
func (p *Provider) params(model string, req *llm.ChatRequest, maxTokens int64) anthropicsdk.MessageNewParams {
	params := anthropicsdk.MessageNewParams{
		Model:     anthropicsdk.Model(model),
		MaxTokens: maxTokens,
		Messages:  convertMessages(req),
		Tools:     convertTools(req.Tools),
	}
	if req.System != "" {
		params.System = []anthropicsdk.TextBlockParam{{Text: req.System}}
	}
	if p.cfg.Temperature != nil {
		params.Temperature = anthropicsdk.Float(*p.cfg.Temperature)
	}
	if p.cfg.TopP != nil {
		params.TopP = anthropicsdk.Float(*p.cfg.TopP)
	}
	markCacheBreakpoints(&params)
	// The API rejects a history holding tool_use/tool_result blocks when no
	// tools are declared, so "no tools now" keeps the declarations and
	// says tool_choice none instead.
	if len(params.Tools) > 0 {
		switch req.ToolChoice.Mode {
		case llm.ToolChoiceNone:
			params.ToolChoice = anthropicsdk.ToolChoiceUnionParam{OfNone: &anthropicsdk.ToolChoiceNoneParam{}}
		case llm.ToolChoiceRequired:
			params.ToolChoice = anthropicsdk.ToolChoiceUnionParam{OfAny: &anthropicsdk.ToolChoiceAnyParam{}}
		case llm.ToolChoiceTool:
			params.ToolChoice = anthropicsdk.ToolChoiceUnionParam{OfTool: &anthropicsdk.ToolChoiceToolParam{Name: req.ToolChoice.Name}}
		}
	}
	return params
}

// markCacheBreakpoints asks Anthropic to cache the request prefix: one
// breakpoint after the system prompt (tools + system, stable for the whole
// session) and one on the last block of the conversation, so the next
// iteration of a tool loop reads everything up to here from the cache.
// Prompts below the model's minimum cacheable length are simply not
// cached; the API does not reject them.
func markCacheBreakpoints(params *anthropicsdk.MessageNewParams) {
	if n := len(params.System); n > 0 {
		params.System[n-1].CacheControl = anthropicsdk.NewCacheControlEphemeralParam()
	}
	if n := len(params.Messages); n > 0 {
		blocks := params.Messages[n-1].Content
		if m := len(blocks); m > 0 {
			if cc := blocks[m-1].GetCacheControl(); cc != nil {
				*cc = anthropicsdk.NewCacheControlEphemeralParam()
			}
		}
	}
}

// maxTokensOf resolves the response cap.
func maxTokensOf(req *llm.ChatRequest) int64 {
	if req.MaxTokens > 0 {
		return int64(req.MaxTokens)
	}
	return defaultMaxTokens
}

// stopReason maps an Anthropic stop_reason onto the neutral enum.
func stopReason(r anthropicsdk.StopReason) llm.StopReason {
	switch r {
	case "":
		return llm.StopUnknown
	case anthropicsdk.StopReasonEndTurn, anthropicsdk.StopReasonStopSequence:
		return llm.StopEnd
	case anthropicsdk.StopReasonToolUse:
		return llm.StopToolUse
	case anthropicsdk.StopReasonMaxTokens, anthropicsdk.StopReasonModelContextWindowExceeded:
		return llm.StopMaxTokens
	default:
		return llm.StopOther
	}
}

// usageOf converts the API usage block.
func usageOf(u anthropicsdk.Usage) llm.Usage {
	return llm.Usage{
		InputTokens:      int(u.InputTokens),
		OutputTokens:     int(u.OutputTokens),
		CacheReadTokens:  int(u.CacheReadInputTokens),
		CacheWriteTokens: int(u.CacheCreationInputTokens),
	}
}

// StreamChat implements llm.LLMProvider. Text and thinking deltas stream as
// they arrive; tool calls are emitted (aggregated) once per turn.
func (p *Provider) StreamChat(ctx context.Context, req *llm.ChatRequest) (<-chan llm.StreamEvent, error) {
	model, err := p.modelID(req)
	if err != nil {
		return nil, err
	}

	stream := p.client.Messages.NewStreaming(ctx, p.params(model, req, maxTokensOf(req)))

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
		// the usage of a cancelled stream rides on it, and dropping it
		// books a timed-out call as free.
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

		message := anthropicsdk.Message{}
		for stream.Next() {
			event := stream.Current()
			if err := message.Accumulate(event); err != nil {
				emit(llm.StreamEvent{Kind: llm.EventError, Err: fmt.Errorf("anthropic: accumulate: %w", err)})
				return
			}

			switch variant := event.AsAny().(type) {
			case anthropicsdk.ContentBlockDeltaEvent:
				switch delta := variant.Delta.AsAny().(type) {
				case anthropicsdk.TextDelta:
					if !emit(llm.StreamEvent{Kind: llm.EventText, Text: delta.Text}) {
						return
					}
				case anthropicsdk.ThinkingDelta:
					if !emit(llm.StreamEvent{Kind: llm.EventThinking, Text: delta.Thinking}) {
						return
					}
				}
			}
		}
		if err := stream.Err(); err != nil {
			// The final usage block never came: book what message_start and
			// the deltas reported so far, so a dropped or cancelled call is
			// not accounted as zero tokens.
			var usage *llm.Usage
			if message.Usage.InputTokens > 0 || message.Usage.OutputTokens > 0 {
				u := usageOf(message.Usage)
				u.Estimated = true
				usage = &u
			}
			emitFinal(llm.StreamEvent{Kind: llm.EventError, Err: streamError(err), Usage: usage})
			return
		}

		// Emit the reasoning blocks and aggregated tool calls from the
		// accumulated message. A reasoning block is kept whole: its signature
		// covers the full text, so nothing may be cut from it.
		var calls []llm.ToolCall
		var reasoning []llm.Part
		for _, block := range message.Content {
			switch variant := block.AsAny().(type) {
			case anthropicsdk.ThinkingBlock:
				reasoning = append(reasoning, llm.Part{Kind: llm.PartThinking, Text: variant.Thinking, Signature: variant.Signature})
			case anthropicsdk.RedactedThinkingBlock:
				reasoning = append(reasoning, llm.Part{Kind: llm.PartThinking, Redacted: variant.Data})
			case anthropicsdk.ToolUseBlock:
				args := map[string]any{}
				if err := json.Unmarshal([]byte(variant.JSON.Input.Raw()), &args); err != nil {
					args = map[string]any{"_raw": variant.JSON.Input.Raw()}
				}
				calls = append(calls, llm.ToolCall{
					ID:   variant.ID,
					Name: variant.Name,
					Args: args,
				})
			}
		}
		if len(reasoning) > 0 && !emit(llm.StreamEvent{Kind: llm.EventReasoning, Reasoning: reasoning}) {
			return
		}
		if len(calls) > 0 && !emit(llm.StreamEvent{Kind: llm.EventToolCalls, ToolCalls: calls}) {
			return
		}

		var usage *llm.Usage
		if message.Usage.InputTokens > 0 || message.Usage.OutputTokens > 0 {
			u := usageOf(message.Usage)
			usage = &u
		}
		emit(llm.StreamEvent{Kind: llm.EventDone, Usage: usage, StopReason: stopReason(message.StopReason)})
	}()

	return out, nil
}

// streamError wraps a stream failure, marking a broken connection (as
// opposed to an error answer from the API) with llm.ErrStreamInterrupted.
func streamError(err error) error {
	var apiErr *anthropicsdk.Error
	if errors.As(err, &apiErr) || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return fmt.Errorf("anthropic: stream: %w", err)
	}
	return fmt.Errorf("anthropic: stream: %w: %w", llm.ErrStreamInterrupted, err)
}

// textOf concatenates the text blocks of a response.
func textOf(resp *anthropicsdk.Message) string {
	var sb strings.Builder
	for _, block := range resp.Content {
		if text, ok := block.AsAny().(anthropicsdk.TextBlock); ok {
			sb.WriteString(text.Text)
		}
	}
	return sb.String()
}

// Generate implements llm.LLMProvider (non-streaming completion).
func (p *Provider) Generate(ctx context.Context, req *llm.ChatRequest) (string, llm.Usage, error) {
	model, err := p.modelID(req)
	if err != nil {
		return "", llm.Usage{}, err
	}

	params := p.params(model, req, maxTokensOf(req))
	params.Tools, params.ToolChoice = nil, anthropicsdk.ToolChoiceUnionParam{}
	resp, err := p.client.Messages.New(ctx, params)
	if err != nil {
		return "", llm.Usage{}, fmt.Errorf("anthropic: generate: %w", err)
	}
	return textOf(resp), usageOf(resp.Usage), nil
}

// GenerateStructured implements llm.LLMProvider with the same two-step
// format chain as the OpenAI adapter: native structured outputs
// (output_config.format) first; if the model rejects the format or the
// reply does not decode, one prompt-guided attempt with lenient parsing.
// Cancellation is never retried. The returned usage covers both attempts.
func (p *Provider) GenerateStructured(ctx context.Context, req *llm.ChatRequest, schema llm.Schema, out any) (llm.Usage, error) {
	var usage llm.Usage
	model, err := p.modelID(req)
	if err != nil {
		return usage, err
	}

	params := p.params(model, req, maxTokensOf(req))
	params.Tools, params.ToolChoice = nil, anthropicsdk.ToolChoiceUnionParam{}
	params.OutputConfig = anthropicsdk.OutputConfigParam{
		Format: anthropicsdk.JSONOutputFormatParam{
			Schema: schema,
		},
	}

	resp, err := p.client.Messages.New(ctx, params)
	if err == nil {
		usage.Add(usageOf(resp.Usage))
		if raw := llm.ExtractJSON(textOf(resp)); raw != "" {
			if err := json.Unmarshal([]byte(raw), out); err == nil {
				return usage, nil
			}
		}
	}
	if ctx.Err() != nil {
		return usage, fmt.Errorf("anthropic: structured generation failed: %w", ctx.Err())
	}
	if err != nil && !isRetryableFormatError(err) {
		return usage, fmt.Errorf("anthropic: structured generation failed: %w", err)
	}

	// Prompt-guided attempt: the schema rides in the prompt.
	schemaJSON, merr := json.Marshal(schema)
	if merr != nil {
		return usage, fmt.Errorf("anthropic: marshal schema: %w", merr)
	}
	params = p.params(model, llm.WithSchemaNote(req, string(schemaJSON)), maxTokensOf(req))
	params.Tools, params.ToolChoice = nil, anthropicsdk.ToolChoiceUnionParam{}
	resp, err = p.client.Messages.New(ctx, params)
	if err != nil {
		return usage, fmt.Errorf("anthropic: structured generation (prompt-guided) failed: %w", err)
	}
	usage.Add(usageOf(resp.Usage))
	text := textOf(resp)
	raw := llm.ExtractJSON(text)
	if raw == "" {
		if text == "" {
			return usage, errors.New("anthropic: empty structured response")
		}
		return usage, fmt.Errorf("anthropic: decode structured response %q: no JSON found", text)
	}
	if err := json.Unmarshal([]byte(raw), out); err != nil {
		return usage, fmt.Errorf("anthropic: decode structured response %q: %w", raw, err)
	}
	return usage, nil
}

// isRetryableFormatError reports whether a structured-output failure is
// worth a format downgrade: models without structured-output support
// reject output_config with a 400.
func isRetryableFormatError(err error) bool {
	msg := err.Error()
	return strings.Contains(msg, "output_config") ||
		strings.Contains(msg, "output_format") ||
		strings.Contains(msg, "not supported") ||
		strings.Contains(msg, "unsupported") ||
		strings.Contains(msg, "400")
}
