package anthropic

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// reasoningStream is a turn whose reasoning comes as a signed block, a
// redacted block and then the answer.
const reasoningStream = `event: message_start
data: {"type":"message_start","message":{"id":"m3","type":"message","role":"assistant","content":[],"model":"m","usage":{"input_tokens":1,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"weigh "}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"it"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"SIG=="}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"ENC=="}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: content_block_start
data: {"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"4"}}

event: content_block_stop
data: {"type":"content_block_stop","index":2}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":3}}

event: message_stop
data: {"type":"message_stop"}

`

// A signed block is delivered whole, with its signature; a redacted block
// keeps its opaque payload. Both come in one EventReasoning before the done.
func TestStreamChat_ReasoningBlocksComeBackWhole(t *testing.T) {
	s := newSSEServer(t, reasoningStream)

	events, err := s.provider().StreamChat(context.Background(), &llm.ChatRequest{
		Messages: []llm.Message{*llm.UserText("2+2?")},
	})
	if err != nil {
		t.Fatal(err)
	}

	got := collect(t, events)
	var reasoning *llm.StreamEvent
	for i := range got {
		if got[i].Kind == llm.EventReasoning {
			reasoning = &got[i]
		}
	}
	if reasoning == nil {
		t.Fatalf("no reasoning event in %+v", got)
	}
	want := []llm.Part{
		{Kind: llm.PartThinking, Text: "weigh it", Signature: "SIG=="},
		{Kind: llm.PartThinking, Redacted: "ENC=="},
	}
	if len(reasoning.Reasoning) != len(want) {
		t.Fatalf("reasoning = %+v, want %+v", reasoning.Reasoning, want)
	}
	for i := range want {
		if reasoning.Reasoning[i] != want[i] {
			t.Fatalf("block %d = %+v, want %+v", i, reasoning.Reasoning[i], want[i])
		}
	}
}

// Replay sends signed and redacted blocks back exactly as they came, and
// leaves out a block with no signature: the API could not verify it.
func TestConvertMessages_ReplaysSignedReasoningOnly(t *testing.T) {
	history := []llm.Message{
		*llm.UserText("q"),
		{Role: llm.RoleAssistant, Parts: []llm.Part{
			{Kind: llm.PartThinking, Text: "why", Signature: "SIG"},
			{Kind: llm.PartThinking, Redacted: "ENC"},
			{Kind: llm.PartThinking, Text: "unsigned, from another provider"},
			llm.TextPart("a"),
		}},
	}

	type block struct {
		Type      string `json:"type"`
		Text      string `json:"text"`
		Thinking  string `json:"thinking"`
		Signature string `json:"signature"`
		Data      string `json:"data"`
	}
	type message struct {
		Role    string  `json:"role"`
		Content []block `json:"content"`
	}
	replayed := func(preserve bool) []block {
		raw, err := json.Marshal(convertMessages(&llm.ChatRequest{Messages: history, PreserveReasoning: preserve}))
		if err != nil {
			t.Fatal(err)
		}
		var msgs []message
		if err := json.Unmarshal(raw, &msgs); err != nil {
			t.Fatal(err)
		}
		if len(msgs) != 2 {
			t.Fatalf("messages = %d, want 2", len(msgs))
		}
		return msgs[1].Content
	}

	got := replayed(true)
	if len(got) != 3 {
		t.Fatalf("assistant blocks = %+v, want thinking, redacted_thinking, text", got)
	}
	if got[0].Type != "thinking" || got[0].Thinking != "why" || got[0].Signature != "SIG" {
		t.Fatalf("signed block = %+v", got[0])
	}
	if got[1].Type != "redacted_thinking" || got[1].Data != "ENC" {
		t.Fatalf("redacted block = %+v", got[1])
	}
	if got[2].Type != "text" || got[2].Text != "a" {
		t.Fatalf("text block = %+v", got[2])
	}

	// Without preserved reasoning the request carries the answer alone.
	if got := replayed(false); len(got) != 1 || got[0].Type != "text" {
		t.Fatalf("reasoning replayed although the request did not ask: %+v", got)
	}
}
