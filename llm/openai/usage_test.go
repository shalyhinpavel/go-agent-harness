package openai

import (
	"context"
	"strconv"
	"strings"
	"testing"

	"github.com/shalyhinpavel/go-agent-harness/llm"
)

func drainEvents(t *testing.T, events <-chan llm.StreamEvent) []llm.StreamEvent {
	t.Helper()
	var got []llm.StreamEvent
	for ev := range events {
		got = append(got, ev)
	}
	return got
}

func textChunk(text string) string {
	return chunkPre + `"delta":{"content":"` + text + `"}}]}`
}

func usageChunk(prompt, completion int) string {
	return `{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[],` +
		`"usage":{"prompt_tokens":` + strconv.Itoa(prompt) + `,"completion_tokens":` + strconv.Itoa(completion) + `,"total_tokens":` + strconv.Itoa(prompt+completion) + `}}`
}

// TestStreamChat_EstimateWhenNoUsageChunk: a backend that ignores
// include_usage (or a connection that ends cleanly without [DONE]) still
// produces tokens; booking them as zero made timed-out arms look free.
func TestStreamChat_EstimateWhenNoUsageChunk(t *testing.T) {
	s := newSSEServer(t, "data: "+textChunk(strings.Repeat("x", 400))+"\n\n")
	events, err := s.provider().StreamChat(context.Background(), &llm.ChatRequest{
		Messages: []llm.Message{*llm.UserText("hi")},
	})
	if err != nil {
		t.Fatal(err)
	}
	got := drainEvents(t, events)
	last := got[len(got)-1]
	if last.Kind != llm.EventDone || last.Usage == nil {
		t.Fatalf("done must carry the estimate, got %+v", last)
	}
	if !last.Usage.Estimated {
		t.Fatalf("the estimate must be marked Estimated: %+v", *last.Usage)
	}
	if last.Usage.OutputTokens != 100 { // 400 bytes / 4
		t.Fatalf("estimate = %d tokens, want 100", last.Usage.OutputTokens)
	}
}

// TestStreamChat_EstimateOnBrokenStream: the stream dies mid-way (bad chunk,
// dropped connection). The error event carries what the call spent.
func TestStreamChat_EstimateOnBrokenStream(t *testing.T) {
	s := newSSEServer(t, "data: "+textChunk(strings.Repeat("y", 40))+"\n\ndata: {not json}\n\n")
	events, err := s.provider().StreamChat(context.Background(), &llm.ChatRequest{
		Messages: []llm.Message{*llm.UserText("hi")},
	})
	if err != nil {
		t.Fatal(err)
	}
	got := drainEvents(t, events)
	last := got[len(got)-1]
	if last.Kind != llm.EventError {
		t.Fatalf("last event = %+v, want an error", last)
	}
	if last.Usage == nil || !last.Usage.Estimated || last.Usage.OutputTokens != 10 {
		t.Fatalf("the error must carry the estimate (10 tokens), got %+v", last.Usage)
	}
}

// TestStreamChat_ServerUsageWins: a reported usage at or above the estimate
// is the real figure, not an estimate.
func TestStreamChat_ServerUsageWins(t *testing.T) {
	s := newSSEServer(t, sseBody(textChunk("hello"), usageChunk(7, 123)))
	events, err := s.provider().StreamChat(context.Background(), &llm.ChatRequest{
		Messages: []llm.Message{*llm.UserText("hi")},
	})
	if err != nil {
		t.Fatal(err)
	}
	got := drainEvents(t, events)
	last := got[len(got)-1]
	if last.Kind != llm.EventDone || last.Usage == nil {
		t.Fatalf("done = %+v", last)
	}
	if last.Usage.OutputTokens != 123 || last.Usage.InputTokens != 7 || last.Usage.Estimated {
		t.Fatalf("server usage must win verbatim: %+v", *last.Usage)
	}
}

// TestStreamChat_ServerUsageIsTheTruth: whenever a usage block arrives it is
// the count, even when the chars/4 estimate is larger. Observed live: a
// completed call streamed 3,944 characters and the estimate (986) replaced
// vLLM's own count — cost accounting must use the backend's figure, not a
// proxy.
func TestStreamChat_ServerUsageIsTheTruth(t *testing.T) {
	s := newSSEServer(t, "data: "+textChunk(strings.Repeat("z", 400))+"\n\ndata: "+usageChunk(9, 5)+"\n\n")
	events, err := s.provider().StreamChat(context.Background(), &llm.ChatRequest{
		Messages: []llm.Message{*llm.UserText("hi")},
	})
	if err != nil {
		t.Fatal(err)
	}
	got := drainEvents(t, events)
	last := got[len(got)-1]
	if last.Usage == nil || last.Usage.OutputTokens != 5 || last.Usage.Estimated {
		t.Fatalf("a reported count must win over a larger estimate: %+v", last.Usage)
	}
	if last.Usage.InputTokens != 9 {
		t.Fatalf("the server's input count is kept: %+v", *last.Usage)
	}
}

// TestSampling_RequestExtraBodyOverrides: a single call can override the
// run's configured extra body — e.g. a helper call caps its own thinking
// budget without changing the run's sampling.
func TestSampling_RequestExtraBodyOverrides(t *testing.T) {
	s := newSSEServer(t, "data: [DONE]\n\n")
	p, err := New(Config{APIKey: "k", BaseURL: s.srv.URL, Model: "m", Sampling: Sampling{
		ExtraBody: map[string]any{"thinking_token_budget": 8192, "top_k": 20},
	}})
	if err != nil {
		t.Fatal(err)
	}
	events, err := p.StreamChat(context.Background(), &llm.ChatRequest{
		Messages:  []llm.Message{*llm.UserText("q")},
		ExtraBody: map[string]any{"thinking_token_budget": 1024},
	})
	if err != nil {
		t.Fatal(err)
	}
	for range events {
	}
	if got := s.lastReq["thinking_token_budget"]; got != 1024.0 {
		t.Fatalf("request extra must win, got %v", got)
	}
	if got := s.lastReq["top_k"]; got != 20.0 {
		t.Fatalf("configured extras must survive the override, got %v", got)
	}
}
