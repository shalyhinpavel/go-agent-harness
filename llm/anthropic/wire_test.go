package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// historyWithToolUse is the shape of an engine's final iteration: the
// conversation already holds a tool call and its result.
func historyWithToolUse() []llm.Message {
	return []llm.Message{
		*llm.UserText("what is 2+2?"),
		{Role: llm.RoleAssistant, Parts: []llm.Part{llm.ToolCallPart(llm.ToolCall{ID: "t1", Name: "calc", Args: map[string]any{"a": 2}})}},
		{Role: llm.RoleTool, Parts: []llm.Part{llm.ToolResultPart(llm.ToolResultData{CallID: "t1", Name: "calc", Content: 4})}},
	}
}

func TestToolChoice_WireShape(t *testing.T) {
	tools := []llm.ToolDef{{Name: "calc", Parameters: llm.Schema{"type": "object"}}}
	cases := map[string]struct {
		choice llm.ToolChoice
		want   string
	}{
		// The API rejects tool_use history without declared tools: "none"
		// must keep the tools and say tool_choice none.
		"none keeps tools": {llm.ToolChoice{Mode: llm.ToolChoiceNone}, `{"type":"none"}`},
		"required is any":  {llm.ToolChoice{Mode: llm.ToolChoiceRequired}, `{"type":"any"}`},
		"named tool":       {llm.ToolChoice{Mode: llm.ToolChoiceTool, Name: "calc"}, `{"name":"calc","type":"tool"}`},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			s := newSSEServer(t, textStream)
			events, err := s.provider().StreamChat(context.Background(), &llm.ChatRequest{
				Messages: historyWithToolUse(), Tools: tools, ToolChoice: tc.choice,
			})
			if err != nil {
				t.Fatal(err)
			}
			for range events {
			}
			if _, ok := s.lastReq["tools"]; !ok {
				t.Fatal("tools must stay declared next to tool_use history")
			}
			got, _ := json.Marshal(s.lastReq["tool_choice"])
			if string(got) != tc.want {
				t.Fatalf("tool_choice = %s, want %s", got, tc.want)
			}
		})
	}
}

func TestStreamChat_StopReasons(t *testing.T) {
	cases := map[string]llm.StopReason{
		"end_turn":   llm.StopEnd,
		"max_tokens": llm.StopMaxTokens,
		"refusal":    llm.StopOther,
	}
	for wire, want := range cases {
		t.Run(wire, func(t *testing.T) {
			body := strings.Replace(textStream, `"stop_reason":"end_turn"`, `"stop_reason":"`+wire+`"`, 1)
			s := newSSEServer(t, body)
			events, err := s.provider().StreamChat(context.Background(), &llm.ChatRequest{Messages: []llm.Message{*llm.UserText("hi")}})
			if err != nil {
				t.Fatal(err)
			}
			var done llm.StreamEvent
			for ev := range events {
				if ev.Kind == llm.EventDone {
					done = ev
				}
			}
			if done.StopReason != want {
				t.Fatalf("stop reason = %q, want %q", done.StopReason, want)
			}
		})
	}
}

func TestConvertTools_SchemaFidelity(t *testing.T) {
	// A schema decoded from JSON/YAML carries required as []any; extra
	// keys must survive the conversion.
	var params llm.Schema
	_ = json.Unmarshal([]byte(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"],"additionalProperties":false}`), &params)
	tools := convertTools([]llm.ToolDef{{Name: "search", Parameters: params}})
	raw, _ := json.Marshal(tools[0])
	if !strings.Contains(string(raw), `"required":["q"]`) {
		t.Fatalf("required dropped: %s", raw)
	}
	if !strings.Contains(string(raw), `"additionalProperties":false`) {
		t.Fatalf("extra schema keys dropped: %s", raw)
	}
}

func TestGenerateStructured_FormatFallback(t *testing.T) {
	var sawPromptGuided bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		if _, has := body["output_config"]; has {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"type":"error","error":{"type":"invalid_request_error","message":"output_config.format is not supported for this model"}}`))
			return
		}
		sawPromptGuided = true
		_, _ = w.Write([]byte(`{"id":"m","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"Sure:\n` + "```json" + `\n{\"answer\":true}\n` + "```" + `"}],"stop_reason":"end_turn","usage":{"input_tokens":9,"output_tokens":4}}`))
	}))
	defer srv.Close()

	p, _ := New(Config{APIKey: "k", Model: "old-model"})
	p.client = anthropicsdk.NewClient(option.WithAPIKey("k"), option.WithBaseURL(srv.URL), option.WithMaxRetries(0))

	var out struct {
		Answer bool `json:"answer"`
	}
	usage, err := p.GenerateStructured(context.Background(),
		&llm.ChatRequest{Messages: []llm.Message{*llm.UserText("q")}}, llm.Schema{"type": "object"}, &out)
	if err != nil {
		t.Fatalf("format fallback must decode: %v", err)
	}
	if !out.Answer || !sawPromptGuided {
		t.Fatalf("prompt-guided attempt not used: answer=%v saw=%v", out.Answer, sawPromptGuided)
	}
	if usage.InputTokens != 9 || usage.OutputTokens != 4 {
		t.Fatalf("usage of the fallback attempt lost: %+v", usage)
	}
}

func TestCacheBreakpoints_SystemAndLastBlock(t *testing.T) {
	s := newSSEServer(t, textStream)
	events, err := s.provider().StreamChat(context.Background(), &llm.ChatRequest{
		System: "you are a test", Messages: historyWithToolUse(),
		Tools: []llm.ToolDef{{Name: "calc", Parameters: llm.Schema{"type": "object"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	for range events {
	}
	raw, _ := json.Marshal(s.lastReq)
	var req struct {
		System []struct {
			CacheControl map[string]any `json:"cache_control"`
		} `json:"system"`
		Messages []struct {
			Content []struct {
				Type         string         `json:"type"`
				CacheControl map[string]any `json:"cache_control"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatal(err)
	}
	if len(req.System) != 1 || req.System[0].CacheControl["type"] != "ephemeral" {
		t.Fatalf("system block must carry an ephemeral breakpoint: %s", raw)
	}
	marked := 0
	for i, m := range req.Messages {
		for j, b := range m.Content {
			if b.CacheControl != nil {
				marked++
				if i != len(req.Messages)-1 || j != len(m.Content)-1 || b.Type != "tool_result" {
					t.Fatalf("breakpoint on message %d block %d (%s), want only the last block", i, j, b.Type)
				}
			}
		}
	}
	if marked != 1 {
		t.Fatalf("conversation breakpoints = %d, want 1", marked)
	}
}

// A stream that dies after message_start must still report what it spent.
func TestStreamChat_BrokenStreamCarriesUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`event: message_start
data: {"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"model":"m","usage":{"input_tokens":42,"output_tokens":1}}}

`))
		w.(http.Flusher).Flush()
		panic(http.ErrAbortHandler) // drop the connection mid-stream
	}))
	defer srv.Close()

	p, _ := New(Config{APIKey: "k", Model: "m"})
	p.client = anthropicsdk.NewClient(option.WithAPIKey("k"), option.WithBaseURL(srv.URL), option.WithMaxRetries(0))
	events, err := p.StreamChat(context.Background(), &llm.ChatRequest{Messages: []llm.Message{*llm.UserText("q")}})
	if err != nil {
		t.Fatal(err)
	}
	var last llm.StreamEvent
	for ev := range events {
		last = ev
	}
	if last.Kind != llm.EventError {
		t.Fatalf("want an error event, got kind %v", last.Kind)
	}
	if last.Usage == nil || last.Usage.InputTokens != 42 || !last.Usage.Estimated {
		t.Fatalf("usage of the broken stream lost: %+v", last.Usage)
	}
	if !errors.Is(last.Err, llm.ErrStreamInterrupted) {
		t.Fatalf("a dropped connection must be marked interrupted: %v", last.Err)
	}
}

// An error answer from the API is not an interrupted stream: repeating the
// request would be rejected again.
func TestStreamChat_APIErrorIsNotMarkedInterrupted(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"invalid_request_error","message":"nope"}}`))
	}))
	defer srv.Close()
	p, _ := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL, MaxRetries: -1})
	events, err := p.StreamChat(context.Background(), &llm.ChatRequest{Messages: []llm.Message{*llm.UserText("q")}})
	if err != nil {
		t.Fatal(err)
	}
	var last llm.StreamEvent
	for ev := range events {
		last = ev
	}
	if last.Kind != llm.EventError || errors.Is(last.Err, llm.ErrStreamInterrupted) {
		t.Fatalf("a 400 must be a plain error, got %+v", last)
	}
}

func TestConfig_BaseURLSamplingAndRetries(t *testing.T) {
	var hits int
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		if hits == 1 {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"m","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer srv.Close()

	temp, topP := 0.3, 0.9
	p, err := New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL, Temperature: &temp, TopP: &topP})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := p.Generate(context.Background(), &llm.ChatRequest{Messages: []llm.Message{*llm.UserText("q")}}); err != nil {
		t.Fatalf("a 500 must be retried by default: %v", err)
	}
	if hits != 2 {
		t.Fatalf("want 2 requests (one retry), got %d", hits)
	}
	if body["temperature"] != 0.3 || body["top_p"] != 0.9 {
		t.Fatalf("sampling not sent: %v", body)
	}

	hits = 0
	p, _ = New(Config{APIKey: "k", Model: "m", BaseURL: srv.URL, MaxRetries: -1})
	if _, _, err := p.Generate(context.Background(), &llm.ChatRequest{Messages: []llm.Message{*llm.UserText("q")}}); err == nil {
		t.Fatal("with retries off the 500 must surface")
	}
}
