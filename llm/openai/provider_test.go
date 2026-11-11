package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// sseServer serves an SSE body for every POST and records the last request.
type sseServer struct {
	t       *testing.T
	body    string
	lastReq map[string]any
	srv     *httptest.Server
}

func newSSEServer(t *testing.T, body string) *sseServer {
	s := &sseServer{t: t, body: body}
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
	p, err := New(Config{APIKey: "test-key", BaseURL: s.srv.URL, Model: "test-model"})
	if err != nil {
		s.t.Fatalf("new provider: %v", err)
	}
	return p
}

func TestStreamChat_Text(t *testing.T) {
	chunk := func(delta string) string {
		b, _ := json.Marshal(map[string]any{
			"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
			"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"content": delta}}},
		})
		return "data: " + string(b) + "\n\n"
	}
	s := newSSEServer(t, chunk("Hel")+chunk("lo")+"data: [DONE]\n\n")

	events, err := s.provider().StreamChat(context.Background(), &llm.ChatRequest{
		Messages: []llm.Message{*llm.UserText("hi")},
	})
	if err != nil {
		t.Fatal(err)
	}

	var got []llm.StreamEvent
	for ev := range events {
		got = append(got, ev)
	}
	if len(got) != 3 ||
		got[0].Kind != llm.EventText || got[0].Text != "Hel" ||
		got[1].Kind != llm.EventText || got[1].Text != "lo" ||
		got[2].Kind != llm.EventDone {
		t.Fatalf("unexpected events: %+v", got)
	}
}

func TestStreamChat_ToolCall(t *testing.T) {
	tc := func(delta map[string]any, finish string) string {
		entry := map[string]any{"index": 0, "delta": delta}
		choice := map[string]any{"index": 0, "delta": delta}
		if delta["tool_calls"] != nil {
			choice["delta"] = map[string]any{"tool_calls": delta["tool_calls"]}
		}
		if finish != "" {
			choice["finish_reason"] = finish
		}
		b, _ := json.Marshal(map[string]any{
			"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
			"choices": []any{choice},
		})
		_ = entry
		return "data: " + string(b) + "\n\n"
	}

	body := tc(map[string]any{"tool_calls": []any{map[string]any{
		"index": 0, "id": "call_1",
		"function": map[string]any{"name": "calculator", "arguments": `{"a":2`},
	}}}, "") +
		tc(map[string]any{"tool_calls": []any{map[string]any{
			"index":    0,
			"function": map[string]any{"arguments": `,"b":3}`},
		}}}, "") +
		tc(map[string]any{}, "tool_calls") +
		"data: [DONE]\n\n"

	s := newSSEServer(t, body)
	events, err := s.provider().StreamChat(context.Background(), &llm.ChatRequest{
		Messages: []llm.Message{*llm.UserText("add")},
	})
	if err != nil {
		t.Fatal(err)
	}

	var toolCalls []llm.ToolCall
	for ev := range events {
		if ev.Kind == llm.EventToolCalls {
			toolCalls = append(toolCalls, ev.ToolCalls...)
		}
	}
	if len(toolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(toolCalls))
	}
	if toolCalls[0].ID != "call_1" || toolCalls[0].Name != "calculator" {
		t.Fatalf("unexpected call: %+v", toolCalls[0])
	}
	if toolCalls[0].Args["a"] != float64(2) || toolCalls[0].Args["b"] != float64(3) {
		t.Fatalf("args not aggregated: %+v", toolCalls[0].Args)
	}
}

func TestGenerate(t *testing.T) {
	s := newSSEServer(t, "")
	s.srv.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"c1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"answer"}}],"usage":{"prompt_tokens":7,"completion_tokens":2}}`))
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
	if usage.InputTokens != 7 || usage.OutputTokens != 2 {
		t.Fatalf("Generate must report usage, got %+v", usage)
	}
}

func TestGenerateStructured(t *testing.T) {
	var seen []map[string]any
	s := newSSEServer(t, "")
	s.srv.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		seen = append(seen, req)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(sseBody(
			chunkPre+`"delta":{"content":"{\n  \"answer\""}}]}`,
			chunkPre+`"delta":{"content":": true\n}"},"finish_reason":"stop"}]}`,
			`{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3}}`,
		)))
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
	usage, err := s.provider().GenerateStructured(context.Background(),
		&llm.ChatRequest{Messages: []llm.Message{*llm.UserText("q")}}, schema, &out)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Answer {
		t.Fatal("expected answer=true")
	}
	if usage.InputTokens != 5 || usage.OutputTokens != 3 {
		t.Fatalf("GenerateStructured must report usage, got %+v", usage)
	}
	if len(seen) != 1 {
		t.Fatalf("a good native reply needs one request, got %d", len(seen))
	}
	rf, _ := seen[0]["response_format"].(map[string]any)
	if rf["type"] != "json_schema" || seen[0]["stream"] != true {
		t.Fatalf("native attempt must stream with json_schema: %+v", seen[0])
	}
	if so, _ := seen[0]["stream_options"].(map[string]any); so["include_usage"] != true {
		t.Fatalf("native attempt must ask for usage: %+v", seen[0])
	}
}

// TestGenerateStructured_DegenerateWhitespace: a backend whose constrained
// decoding loops on whitespace (it streams until max_tokens) must be cut
// off early and answered through the prompt-guided attempt.
func TestGenerateStructured_DegenerateWhitespace(t *testing.T) {
	var fallback map[string]any
	s := newSSEServer(t, "")
	s.srv.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		if _, native := req["response_format"]; !native {
			fallback = req
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"{\"answer\": true}"}}],"usage":{"prompt_tokens":9,"completion_tokens":4}}`))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		_, _ = w.Write([]byte("data: " + chunkPre + `"delta":{"content":"{"}}]}` + "\n\n"))
		// Whitespace until the client hangs up — the live failure never
		// ends on its own before max_tokens.
		for {
			_, err := w.Write([]byte("data: " + chunkPre + `"delta":{"content":"\t\t\n  "}}]}` + "\n\n"))
			flusher.Flush()
			if err != nil || r.Context().Err() != nil {
				return
			}
		}
	})

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	var out struct {
		Answer bool `json:"answer"`
	}
	usage, err := s.provider().GenerateStructured(ctx,
		&llm.ChatRequest{Messages: []llm.Message{*llm.UserText("q")}}, llm.Schema{"type": "object"}, &out)
	if err != nil {
		t.Fatalf("degenerate native reply must fall back, got: %v", err)
	}
	if !out.Answer || fallback == nil {
		t.Fatalf("answer must come from the prompt-guided attempt: %+v", out)
	}
	// 1 opening delta + ceil(256/5)+1 whitespace deltas until the bound
	// trips, then the fallback's own report.
	if usage.InputTokens != 9 || usage.OutputTokens != 4+1+52 {
		t.Fatalf("the aborted attempt must be booked by its deltas: %+v", usage)
	}
}

func TestConvertMessages_ToolRoundtrip(t *testing.T) {
	s := newSSEServer(t, "data: [DONE]\n\n")
	req := &llm.ChatRequest{
		System: "be brief",
		Messages: []llm.Message{
			*llm.UserText("add 2 and 3"),
			{Role: llm.RoleAssistant, Parts: []llm.Part{
				llm.ToolCallPart(llm.ToolCall{ID: "call_1", Name: "calculator", Args: map[string]any{"a": 2.0, "b": 3.0}}),
			}},
			{Role: llm.RoleTool, Parts: []llm.Part{
				llm.ToolResultPart(llm.ToolResultData{CallID: "call_1", Name: "calculator", Content: map[string]any{"sum": 5.0}}),
			}},
		},
	}

	events, err := s.provider().StreamChat(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	for range events {
	}

	msgs := s.lastReq["messages"].([]any)
	if len(msgs) != 4 { // system, user, assistant(tool_calls), tool
		t.Fatalf("messages = %d, want 4", len(msgs))
	}
	if msgs[0].(map[string]any)["role"] != "system" {
		t.Fatal("system message missing")
	}
	tool := msgs[3].(map[string]any)
	if tool["role"] != "tool" || tool["tool_call_id"] != "call_1" {
		t.Fatalf("tool message malformed: %+v", tool)
	}
	if !strings.Contains(tool["content"].(string), "sum") {
		t.Fatalf("tool content missing payload: %+v", tool["content"])
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
	b, _ := json.Marshal(map[string]any{
		"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
		"choices": []any{},
		"usage": map[string]any{
			"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15,
			"prompt_tokens_details": map[string]any{"cached_tokens": 5},
		},
	})
	s := newSSEServer(t, "data: "+string(b)+"\n\n"+"data: [DONE]\n\n")

	events, err := s.provider().StreamChat(context.Background(), &llm.ChatRequest{
		Messages: []llm.Message{*llm.UserText("hi")},
	})
	if err != nil {
		t.Fatal(err)
	}

	var done *llm.StreamEvent
	for ev := range events {
		if ev.Kind == llm.EventDone {
			done = &ev
		}
	}
	if done == nil || done.Usage == nil {
		t.Fatal("done event must carry usage")
	}
	if done.Usage.InputTokens != 12 || done.Usage.OutputTokens != 3 || done.Usage.CacheReadTokens != 5 {
		t.Fatalf("usage = %+v, want {12 3 cache:5}", done.Usage)
	}
}

func TestConvertMessages_ImageAttachment(t *testing.T) {
	s := newSSEServer(t, "data: [DONE]\n\n")

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
	if !ok {
		t.Fatalf("expected multipart content, got %T", s.lastReq["messages"].([]any)[0].(map[string]any)["content"])
	}
	if len(content) != 2 {
		t.Fatalf("content parts = %d, want 2 (text+image)", len(content))
	}
	text := content[0].(map[string]any)
	if text["type"] != "text" || text["text"] != "what is this?" {
		t.Fatalf("text part wrong: %+v", text)
	}
	image := content[1].(map[string]any)
	if image["type"] != "image_url" {
		t.Fatalf("image part wrong: %+v", image)
	}
	url := image["image_url"].(map[string]any)["url"].(string)
	if url != "data:image/png;base64,AAAA" {
		t.Fatalf("image url = %q", url)
	}
}

func TestStreamChat_ReasoningDeltaStreamed(t *testing.T) {
	b, _ := json.Marshal(map[string]any{
		"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
		"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"reasoning_content": "step 1"}}},
	})
	s := newSSEServer(t, "data: "+string(b)+"\n\n"+"data: [DONE]\n\n")

	events, err := s.provider().StreamChat(context.Background(), &llm.ChatRequest{
		Messages: []llm.Message{*llm.UserText("hi")},
	})
	if err != nil {
		t.Fatal(err)
	}
	var got []llm.StreamEvent
	for ev := range events {
		got = append(got, ev)
	}
	if len(got) != 2 || got[0].Kind != llm.EventThinking || got[0].Text != "step 1" {
		t.Fatalf("reasoning delta not streamed: %+v", got)
	}
}

func TestConvertMessages_ReasoningReplayed(t *testing.T) {
	s := newSSEServer(t, "data: [DONE]\n\n")

	history := []llm.Message{
		*llm.UserText("q"),
		{Role: llm.RoleAssistant, Parts: []llm.Part{llm.ThinkingPart("deliberation"), llm.TextPart("a")}},
		*llm.UserText("sure?"),
	}
	if _, err := s.provider().StreamChat(context.Background(), &llm.ChatRequest{
		PreserveReasoning: true,
		Messages:          history,
	}); err != nil {
		t.Fatal(err)
	}

	assistant := s.lastReq["messages"].([]any)[1].(map[string]any)
	details, ok := assistant["reasoning_details"].([]any)
	if !ok || len(details) != 1 {
		t.Fatalf("reasoning_details missing on assistant message: %+v", assistant)
	}
	d := details[0].(map[string]any)
	if d["type"] != "reasoning.text" || d["text"] != "deliberation" {
		t.Fatalf("reasoning detail wrong: %+v", d)
	}
}

// A thinking part with no text (an Anthropic block whose display was omitted)
// has nothing to replay, so it must not put an empty reasoning detail on the
// wire.
func TestConvertMessages_EmptyReasoningNotReplayed(t *testing.T) {
	s := newSSEServer(t, "data: [DONE]\n\n")

	history := []llm.Message{
		*llm.UserText("q"),
		{Role: llm.RoleAssistant, Parts: []llm.Part{llm.ThinkingPart(""), llm.TextPart("a")}},
		*llm.UserText("sure?"),
	}
	if _, err := s.provider().StreamChat(context.Background(), &llm.ChatRequest{
		PreserveReasoning: true,
		Messages:          history,
	}); err != nil {
		t.Fatal(err)
	}

	assistant := s.lastReq["messages"].([]any)[1].(map[string]any)
	if _, ok := assistant["reasoning_details"]; ok {
		t.Fatalf("an empty reasoning part must not be replayed: %+v", assistant)
	}
}

// TestSampling_OnEveryMethod: the configured knobs ride on stream, plain
// and structured requests alike, and extra_body wins over typed fields.
func TestSampling_OnEveryMethod(t *testing.T) {
	var seen []map[string]any
	s := newSSEServer(t, "")
	s.srv.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		seen = append(seen, req)
		if req["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte(sseBody(chunkPre + `"delta":{"content":"{\"ok\":true}"},"finish_reason":"stop"}]}`)))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	})

	temp, topP, topK := 0.6, 0.95, 20
	p, err := New(Config{APIKey: "k", BaseURL: s.srv.URL, Model: "m", Sampling: Sampling{
		Temperature: &temp, TopP: &topP, TopK: &topK, ReasoningEffort: "low",
		ExtraBody: map[string]any{
			"chat_template_kwargs": map[string]any{"enable_thinking": true},
			"top_p":                0.8, // extra_body wins
		},
	}})
	if err != nil {
		t.Fatal(err)
	}
	req := &llm.ChatRequest{Messages: []llm.Message{*llm.UserText("q")}}
	events, err := p.StreamChat(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	for range events {
	}
	if _, _, err := p.Generate(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	if _, err := p.GenerateStructured(context.Background(), req, llm.Schema{"type": "object"}, &out); err != nil {
		t.Fatal(err)
	}

	if len(seen) != 3 {
		t.Fatalf("requests = %d, want 3", len(seen))
	}
	for i, body := range seen {
		kw, _ := body["chat_template_kwargs"].(map[string]any)
		if body["temperature"] != 0.6 || body["top_p"] != 0.8 || body["top_k"] != 20.0 ||
			body["reasoning_effort"] != "low" || kw["enable_thinking"] != true {
			t.Fatalf("request %d lacks the sampling knobs: %+v", i, body)
		}
	}
}

// TestSampling_UnsetSendsNothing: without configuration the backend
// defaults stay in charge — no knob is sent at all.
func TestSampling_UnsetSendsNothing(t *testing.T) {
	s := newSSEServer(t, "data: [DONE]\n\n")
	events, err := s.provider().StreamChat(context.Background(), &llm.ChatRequest{Messages: []llm.Message{*llm.UserText("q")}})
	if err != nil {
		t.Fatal(err)
	}
	for range events {
	}
	for _, k := range []string{"temperature", "top_p", "top_k", "reasoning_effort"} {
		if _, has := s.lastReq[k]; has {
			t.Fatalf("%s sent without configuration: %+v", k, s.lastReq)
		}
	}
}
