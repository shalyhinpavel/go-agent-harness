package openai

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// Wire-shape fixtures: OpenAI-compatible servers disagree on how a tool
// call is chunked, whether usage is streamed, and how finish reasons are
// reported. Every shape here was either seen in the wild or is allowed by
// the spec; the adapter must read all of them the same way.

const chunkPre = `{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,`

func sseBody(chunks ...string) string {
	var sb strings.Builder
	for _, c := range chunks {
		sb.WriteString("data: " + c + "\n\n")
	}
	sb.WriteString("data: [DONE]\n\n")
	return sb.String()
}

// drain collects a stream into its tool calls and the done event.
func drain(t *testing.T, events <-chan llm.StreamEvent) (calls []llm.ToolCall, text string, done llm.StreamEvent) {
	t.Helper()
	for ev := range events {
		switch ev.Kind {
		case llm.EventToolCalls:
			calls = append(calls, ev.ToolCalls...)
		case llm.EventText:
			text += ev.Text
		case llm.EventError:
			t.Fatalf("stream error: %v", ev.Err)
		case llm.EventDone:
			done = ev
		}
	}
	return
}

func toolReq() *llm.ChatRequest {
	return &llm.ChatRequest{
		Messages: []llm.Message{*llm.UserText("hi")},
		Tools:    []llm.ToolDef{{Name: "calc", Parameters: llm.Schema{"type": "object"}}},
	}
}

func TestStreamChat_ToolCallShapes(t *testing.T) {
	cases := map[string][]string{
		// The OpenAI reference shape: id+name first, argument fragments,
		// then a bare finish chunk.
		"incremental": {
			chunkPre + `"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"calc","arguments":""}}]}}]}`,
			chunkPre + `"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"a\":"}}]}}]}`,
			chunkPre + `"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}`,
			chunkPre + `"delta":{},"finish_reason":"tool_calls"}]}`,
		},
		// Whole call and the finish reason in one chunk — the SDK
		// accumulator's JustFinishedToolCall never fires for it.
		"whole call with finish": {
			chunkPre + `"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"calc","arguments":"{\"a\":1}"}}]},"finish_reason":"tool_calls"}]}`,
		},
		// An empty content field next to the tool call: the SDK classified
		// the chunk as text and lost the call.
		"empty content alongside": {
			chunkPre + `"delta":{"role":"assistant","content":"","tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"calc","arguments":"{\"a\":1}"}}]}}]}`,
			chunkPre + `"delta":{"content":""},"finish_reason":"tool_calls"}]}`,
		},
	}
	for name, chunks := range cases {
		t.Run(name, func(t *testing.T) {
			s := newSSEServer(t, sseBody(chunks...))
			events, err := s.provider().StreamChat(context.Background(), toolReq())
			if err != nil {
				t.Fatal(err)
			}
			calls, _, done := drain(t, events)
			if len(calls) != 1 || calls[0].Name != "calc" || calls[0].ID != "c1" || calls[0].Args["a"] != float64(1) {
				t.Fatalf("tool call lost or garbled: %+v", calls)
			}
			if done.StopReason != llm.StopToolUse {
				t.Fatalf("stop reason = %q", done.StopReason)
			}
		})
	}
}

func TestStreamChat_ParallelCallsWithoutIndex(t *testing.T) {
	// Servers that omit the index send each call whole with its own id:
	// two calls must stay two calls, not one glued payload.
	s := newSSEServer(t, sseBody(
		chunkPre+`"delta":{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"calc","arguments":"{\"a\":1}"}}]}}]}`,
		chunkPre+`"delta":{"tool_calls":[{"id":"c2","type":"function","function":{"name":"calc","arguments":"{\"a\":2}"}}]},"finish_reason":"tool_calls"}]}`,
	))
	events, err := s.provider().StreamChat(context.Background(), toolReq())
	if err != nil {
		t.Fatal(err)
	}
	calls, _, _ := drain(t, events)
	if len(calls) != 2 || calls[0].ID != "c1" || calls[1].ID != "c2" || calls[1].Args["a"] != float64(2) {
		t.Fatalf("parallel calls merged: %+v", calls)
	}
}

func TestStreamChat_UsageAndStopReason(t *testing.T) {
	cases := map[string]struct {
		chunks []string
		in     int
		out    int
		stop   llm.StopReason
	}{
		// include_usage: one trailing chunk with no choices.
		"trailing usage chunk": {[]string{
			chunkPre + `"delta":{"content":"hi"}}]}`,
			chunkPre + `"delta":{},"finish_reason":"stop"}]}`,
			`{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":4}}}`,
		}, 10, 2, llm.StopEnd},
		// Cumulative usage on every chunk: the latest wins, never the sum.
		"cumulative usage": {[]string{
			`{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"a"}}],"usage":{"prompt_tokens":10,"completion_tokens":1}}`,
			`{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"b"},"finish_reason":"length"}],"usage":{"prompt_tokens":10,"completion_tokens":2}}`,
		}, 10, 2, llm.StopMaxTokens},
		// Chunk ids that change mid-stream must not stop accumulation.
		"varying chunk ids": {[]string{
			`{"id":"a","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"x"}}]}`,
			`{"id":"b","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1}}`,
		}, 3, 1, llm.StopEnd},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			s := newSSEServer(t, sseBody(tc.chunks...))
			events, err := s.provider().StreamChat(context.Background(), &llm.ChatRequest{Messages: []llm.Message{*llm.UserText("hi")}})
			if err != nil {
				t.Fatal(err)
			}
			_, _, done := drain(t, events)
			if done.Usage == nil || done.Usage.InputTokens != tc.in || done.Usage.OutputTokens != tc.out {
				t.Fatalf("usage = %+v, want %d/%d", done.Usage, tc.in, tc.out)
			}
			if done.StopReason != tc.stop {
				t.Fatalf("stop reason = %q, want %q", done.StopReason, tc.stop)
			}
		})
	}
}

func TestStreamChat_RequestShape(t *testing.T) {
	body := sseBody(chunkPre + `"delta":{"content":"ok"},"finish_reason":"stop"}]}`)

	t.Run("include_usage and max_tokens by default", func(t *testing.T) {
		s := newSSEServer(t, body)
		events, _ := s.provider().StreamChat(context.Background(), toolReq())
		drain(t, events)
		so, _ := s.lastReq["stream_options"].(map[string]any)
		if so["include_usage"] != true {
			t.Fatalf("stream_options.include_usage missing: %v", s.lastReq["stream_options"])
		}
		if _, ok := s.lastReq["max_tokens"]; !ok {
			t.Fatal("max_tokens expected by default")
		}
		if _, ok := s.lastReq["tool_choice"]; ok {
			t.Fatal("auto mode must not send tool_choice")
		}
	})

	t.Run("max_completion_tokens when configured", func(t *testing.T) {
		s := newSSEServer(t, body)
		p, _ := New(Config{APIKey: "k", BaseURL: s.srv.URL, Model: "m", MaxCompletionTokens: true})
		events, _ := p.StreamChat(context.Background(), toolReq())
		drain(t, events)
		if _, ok := s.lastReq["max_completion_tokens"]; !ok {
			t.Fatal("max_completion_tokens expected")
		}
		if _, ok := s.lastReq["max_tokens"]; ok {
			t.Fatal("max_tokens must not be sent next to max_completion_tokens")
		}
	})

	choices := map[string]struct {
		choice    llm.ToolChoice
		wantTools bool
		want      string
	}{
		"none omits tools": {llm.ToolChoice{Mode: llm.ToolChoiceNone}, false, ""},
		"required":         {llm.ToolChoice{Mode: llm.ToolChoiceRequired}, true, `"required"`},
		"named":            {llm.ToolChoice{Mode: llm.ToolChoiceTool, Name: "calc"}, true, `{"function":{"name":"calc"},"type":"function"}`},
	}
	for name, tc := range choices {
		t.Run("tool_choice "+name, func(t *testing.T) {
			s := newSSEServer(t, body)
			req := toolReq()
			req.ToolChoice = tc.choice
			events, _ := s.provider().StreamChat(context.Background(), req)
			drain(t, events)
			_, hasTools := s.lastReq["tools"]
			if hasTools != tc.wantTools {
				t.Fatalf("tools present = %v, want %v", hasTools, tc.wantTools)
			}
			got, _ := json.Marshal(s.lastReq["tool_choice"])
			if tc.want == "" && s.lastReq["tool_choice"] != nil {
				t.Fatalf("tool_choice must be absent, got %s", got)
			}
			if tc.want != "" && string(got) != tc.want {
				t.Fatalf("tool_choice = %s, want %s", got, tc.want)
			}
		})
	}
}

func TestStreamChat_ReasoningNullFieldSkipped(t *testing.T) {
	s := newSSEServer(t, sseBody(
		chunkPre+`"delta":{"reasoning_content":null,"reasoning":"thinking hard"}}]}`,
		chunkPre+`"delta":{"content":"done"},"finish_reason":"stop"}]}`,
	))
	events, _ := s.provider().StreamChat(context.Background(), &llm.ChatRequest{Messages: []llm.Message{*llm.UserText("hi")}})
	var thinking string
	for ev := range events {
		if ev.Kind == llm.EventThinking {
			thinking += ev.Text
		}
	}
	if thinking != "thinking hard" {
		t.Fatalf("reasoning behind a null sibling lost: %q", thinking)
	}
}
