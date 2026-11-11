package anthropic

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// sseServer serves an SSE body for every POST and records the last request.
type sseServer struct {
	t       *testing.T
	lastReq map[string]any
	srv     *httptest.Server
}

func newSSEServer(t *testing.T, body string) *sseServer {
	s := &sseServer{t: t}
	s.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		s.lastReq = req
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(s.srv.Close)
	return s
}

func (s *sseServer) provider() *Provider {
	p, err := New(Config{APIKey: "test-key", Model: "test-model"})
	if err != nil {
		s.t.Fatalf("new provider: %v", err)
	}
	// Route the client at the test server.
	p.client = anthropicsdk.NewClient(
		option.WithAPIKey("test-key"),
		option.WithBaseURL(s.srv.URL),
	)
	return p
}

const textStream = `event: message_start
data: {"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","content":[],"model":"m","usage":{"input_tokens":1,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}

`

const toolUseStream = `event: message_start
data: {"type":"message_start","message":{"id":"m2","type":"message","role":"assistant","content":[],"model":"m","usage":{"input_tokens":1,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tool_1","name":"calculator","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"a\":2"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":",\"b\":3}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":4}}

event: message_stop
data: {"type":"message_stop"}

`

func collect(t *testing.T, events <-chan llm.StreamEvent) []llm.StreamEvent {
	t.Helper()
	var got []llm.StreamEvent
	for ev := range events {
		if ev.Kind == llm.EventError {
			t.Fatalf("stream error: %v", ev.Err)
		}
		got = append(got, ev)
	}
	return got
}

func TestStreamChat_Text(t *testing.T) {
	s := newSSEServer(t, textStream)

	events, err := s.provider().StreamChat(context.Background(), &llm.ChatRequest{
		System:   "be brief",
		Messages: []llm.Message{*llm.UserText("hi")},
	})
	if err != nil {
		t.Fatal(err)
	}

	got := collect(t, events)
	if len(got) != 3 ||
		got[0].Kind != llm.EventText || got[0].Text != "Hel" ||
		got[1].Kind != llm.EventText || got[1].Text != "lo" ||
		got[2].Kind != llm.EventDone {
		t.Fatalf("unexpected events: %+v", got)
	}

	// Request shape: system prompt, model, user message.
	if s.lastReq["model"] != "test-model" {
		t.Fatalf("model = %v", s.lastReq["model"])
	}
	if sys, ok := s.lastReq["system"].([]any); !ok || len(sys) != 1 {
		t.Fatalf("system = %+v", s.lastReq["system"])
	}
}

func TestStreamChat_ThinkingDelta(t *testing.T) {
	body := strings.Replace(textStream, `"delta":{"type":"text_delta","text":"Hel"}`,
		`"delta":{"type":"thinking_delta","thinking":"pondering"}`, 1)
	s := newSSEServer(t, body)

	events, err := s.provider().StreamChat(context.Background(), &llm.ChatRequest{
		Messages: []llm.Message{*llm.UserText("hi")},
	})
	if err != nil {
		t.Fatal(err)
	}

	got := collect(t, events)
	if got[0].Kind != llm.EventThinking || got[0].Text != "pondering" {
		t.Fatalf("expected thinking event first, got %+v", got[0])
	}
}

func TestStreamChat_ToolCall(t *testing.T) {
	s := newSSEServer(t, toolUseStream)

	events, err := s.provider().StreamChat(context.Background(), &llm.ChatRequest{
		Messages: []llm.Message{*llm.UserText("add")},
	})
	if err != nil {
		t.Fatal(err)
	}

	var calls []llm.ToolCall
	for _, ev := range collect(t, events) {
		if ev.Kind == llm.EventToolCalls {
			calls = append(calls, ev.ToolCalls...)
		}
	}
	if len(calls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(calls))
	}
	if calls[0].ID != "tool_1" || calls[0].Name != "calculator" {
		t.Fatalf("unexpected call: %+v", calls[0])
	}
	if calls[0].Args["a"] != float64(2) || calls[0].Args["b"] != float64(3) {
		t.Fatalf("args not aggregated: %+v", calls[0].Args)
	}
}

func TestGenerate(t *testing.T) {
	s := newSSEServer(t, "")
	s.srv.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"m1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"answer"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	})

	got, usage, err := s.provider().Generate(context.Background(), &llm.ChatRequest{
		Messages: []llm.Message{*llm.UserText("q")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != "answer" {
		t.Fatalf("Generate = %q", got)
	}
	if usage.InputTokens != 1 || usage.OutputTokens != 1 {
		t.Fatalf("Generate must report usage, got %+v", usage)
	}
}

func TestGenerateStructured(t *testing.T) {
	var seenOutputConfig map[string]any
	s := newSSEServer(t, "")
	s.srv.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		if oc, ok := req["output_config"].(map[string]any); ok {
			seenOutputConfig = oc
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"m1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"{\"answer\":true}"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	})

	schema := llm.Schema{
		"type": "object",
		"properties": map[string]any{
			"answer": map[string]any{"type": "boolean"},
		},
	}
	var out struct {
		Answer bool `json:"answer"`
	}
	_, err := s.provider().GenerateStructured(context.Background(),
		&llm.ChatRequest{Messages: []llm.Message{*llm.UserText("q")}}, schema, &out)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Answer {
		t.Fatal("expected answer=true")
	}
	if seenOutputConfig == nil || seenOutputConfig["format"] == nil {
		t.Fatalf("output_config missing: %+v", seenOutputConfig)
	}
}

func TestConvertMessages_ToolRoundtrip(t *testing.T) {
	s := newSSEServer(t, textStream)
	req := &llm.ChatRequest{
		System: "be brief",
		Messages: []llm.Message{
			*llm.UserText("add 2 and 3"),
			{Role: llm.RoleAssistant, Parts: []llm.Part{
				llm.ToolCallPart(llm.ToolCall{ID: "tool_1", Name: "calculator", Args: map[string]any{"a": 2.0}}),
			}},
			{Role: llm.RoleTool, Parts: []llm.Part{
				llm.ToolResultPart(llm.ToolResultData{CallID: "tool_1", Name: "calculator", Content: map[string]any{"sum": 5.0}, IsError: false}),
			}},
		},
	}

	events, err := s.provider().StreamChat(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	collect(t, events)

	msgs := s.lastReq["messages"].([]any)
	if len(msgs) != 3 { // user(text), user(tool_use), user(tool_result)
		t.Fatalf("messages = %d, want 3", len(msgs))
	}

	first := msgs[1].(map[string]any)["content"].([]any)
	if first[0].(map[string]any)["type"] != "tool_use" {
		t.Fatalf("expected tool_use block second message: %+v", first[0])
	}
	second := msgs[2].(map[string]any)["content"].([]any)
	if second[0].(map[string]any)["type"] != "tool_result" {
		t.Fatalf("expected tool_result block third message: %+v", second[0])
	}
}

func TestNoModelConfigured(t *testing.T) {
	p := &Provider{}
	_, err := p.StreamChat(context.Background(), &llm.ChatRequest{Messages: []llm.Message{*llm.UserText("x")}})
	if err != llm.ErrNoModel {
		t.Fatalf("want ErrNoModel, got %v", err)
	}
}

func TestStreamChat_UsageReported(t *testing.T) {
	// textStream already carries usage: message_start input=1, message_delta
	// output=2 — the accumulator must merge them onto the done event.
	s := newSSEServer(t, textStream)

	events, err := s.provider().StreamChat(context.Background(), &llm.ChatRequest{
		Messages: []llm.Message{*llm.UserText("hi")},
	})
	if err != nil {
		t.Fatal(err)
	}

	var done *llm.StreamEvent
	for _, ev := range collect(t, events) {
		if ev.Kind == llm.EventDone {
			done = &ev
		}
	}
	if done == nil || done.Usage == nil {
		t.Fatal("done event must carry usage")
	}
	if done.Usage.InputTokens != 1 || done.Usage.OutputTokens != 2 {
		t.Fatalf("usage = %+v, want {1 2}", done.Usage)
	}
}

func TestConvertMessages_ImageAttachment(t *testing.T) {
	s := newSSEServer(t, "data: {\"type\":\"message_stop\"}\n\n")

	msg := llm.UserMessage(
		llm.TextPart("what is this?"),
		llm.ImagePart("image/png", "AAAA"),
	)
	if _, err := s.provider().StreamChat(context.Background(), &llm.ChatRequest{
		Messages: []llm.Message{*msg},
	}); err != nil {
		t.Fatal(err)
	}

	content, ok := s.lastReq["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if !ok || len(content) != 2 {
		t.Fatalf("expected text+image blocks, got %+v", s.lastReq["messages"])
	}
	image := content[1].(map[string]any)
	if image["type"] != "image" {
		t.Fatalf("image block wrong: %+v", image)
	}
	source := image["source"].(map[string]any)
	if source["type"] != "base64" || source["media_type"] != "image/png" || source["data"] != "AAAA" {
		t.Fatalf("image source wrong: %+v", source)
	}
}
