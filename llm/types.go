// Package llm defines the provider-agnostic contract between the agent
// engine and any LLM backend: Anthropic, OpenAI-compatible endpoints
// (OpenAI, Ollama, vLLM, LM Studio, self-hosted runtimes) and test mocks.
package llm

// Role identifies the author of a message.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	// RoleTool carries tool results back to the model. Providers map it onto
	// their native representation (a user message holding tool_result blocks,
	// or a dedicated "tool" role).
	RoleTool Role = "tool"
)

// Message is a single turn in a conversation.
type Message struct {
	Role  Role   `json:"role"`
	Parts []Part `json:"parts"`
}

// PartKind discriminates the payload of a Part.
type PartKind string

const (
	// PartText is regular visible text.
	PartText PartKind = "text"
	// PartThinking is model reasoning output. Streamed for observability;
	// persisted into session history only with reasoning preservation on.
	PartThinking PartKind = "thinking"
	// PartToolCall is a tool invocation requested by the model.
	PartToolCall PartKind = "tool_call"
	// PartToolResult carries the outcome of a tool invocation.
	PartToolResult PartKind = "tool_result"
	// PartImage is a user-supplied image (vision input). Requires a
	// multimodal model; providers map it onto their native image blocks.
	PartImage PartKind = "image"
)

// ImageData is an image block: base64-encoded bytes with their media
// type, or a remote URL (provider support for URLs varies — base64 is
// the portable form).
type ImageData struct {
	MediaType string `json:"media_type,omitempty"` // image/png, image/jpeg, image/webp, image/gif
	Data      string `json:"data,omitempty"`       // base64-encoded bytes
	URL       string `json:"url,omitempty"`
}

// Part is one block of message content. Exactly one payload field is set,
// matching Kind. Signature and Redacted belong to thinking parts only.
type Part struct {
	Kind       PartKind        `json:"kind"`
	Text       string          `json:"text,omitempty"`
	Image      *ImageData      `json:"image,omitempty"`
	ToolCall   *ToolCall       `json:"tool_call,omitempty"`
	ToolResult *ToolResultData `json:"tool_result,omitempty"`
	// Signature is the provider's integrity token for a thinking part. It
	// binds the text to the conversation that produced it, so it must come
	// back with the text unchanged; a part without one cannot be replayed.
	Signature string `json:"signature,omitempty"`
	// Redacted is the opaque payload of a reasoning block the provider
	// withheld. Such a part carries no Text.
	Redacted string `json:"redacted,omitempty"`
}

// TextPart builds a text part.
func TextPart(text string) Part {
	return Part{Kind: PartText, Text: text}
}

// ImagePart builds an image part from base64 data.
func ImagePart(mediaType, base64Data string) Part {
	return Part{Kind: PartImage, Image: &ImageData{MediaType: mediaType, Data: base64Data}}
}

// ThinkingPart builds a reasoning part (persisted only when reasoning
// preservation is enabled).
func ThinkingPart(text string) Part {
	return Part{Kind: PartThinking, Text: text}
}

// ImageURLPart builds an image part referencing a remote URL.
func ImageURLPart(url string) Part {
	return Part{Kind: PartImage, Image: &ImageData{URL: url}}
}

// ToolCallPart builds a part holding a tool invocation.
func ToolCallPart(call ToolCall) Part {
	return Part{Kind: PartToolCall, ToolCall: &call}
}

// ToolResultPart builds a part holding a tool outcome.
func ToolResultPart(res ToolResultData) Part {
	return Part{Kind: PartToolResult, ToolResult: &res}
}

// UserText is a shorthand for a plain user message.
func UserText(text string) *Message {
	return &Message{Role: RoleUser, Parts: []Part{TextPart(text)}}
}

// UserMessage builds a user message from explicit parts (text, images...).
func UserMessage(parts ...Part) *Message {
	return &Message{Role: RoleUser, Parts: parts}
}

// AssistantText is a shorthand for a plain assistant message.
func AssistantText(text string) *Message {
	return &Message{Role: RoleAssistant, Parts: []Part{TextPart(text)}}
}

// ToolCall is a tool invocation requested by the model.
type ToolCall struct {
	ID   string         `json:"id"`
	Name string         `json:"name"`
	Args map[string]any `json:"args,omitempty"`
}

// ToolResultData is the outcome of a tool invocation. Content must be
// JSON-serializable; IsError marks a failed execution for the model.
type ToolResultData struct {
	CallID  string `json:"call_id"` // references ToolCall.ID
	Name    string `json:"name"`
	Content any    `json:"content,omitempty"`
	IsError bool   `json:"is_error,omitempty"`
}

// Schema is a canonical JSON Schema describing tool parameters or
// structured-output shape.
type Schema = map[string]any

// ToolDef describes a tool to the model. Execution lives in the agent's
// tool registry, not here.
type ToolDef struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  Schema `json:"parameters,omitempty"`
}
