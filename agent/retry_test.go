package agent

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/shalyhinpavel/go-agent-harness/llm"
)

var errBroken = fmt.Errorf("conn reset: %w", llm.ErrStreamInterrupted)

func retryEngine(cfg AgentConfig, turns ...llm.MockTurn) (*AgentEngine, *llm.MockProvider) {
	mock := llm.NewMockProvider(turns...)
	cfg.Provider = mock
	e := NewAgentEngine(cfg, NewToolRegistry(), NewSessionManager())
	e.retryDelay = 0
	return e, mock
}

func collect(t *testing.T, e *AgentEngine) (text string, last StreamChunk) {
	t.Helper()
	stream, err := e.StreamChat(context.Background(), ChatRequest{UserID: "u", SessionID: "s", Message: "hi"})
	if err != nil {
		t.Fatal(err)
	}
	for c := range stream {
		if c.Type == ChunkTypeText {
			text += c.Content
		}
		last = c
	}
	return text, last
}

func TestStreamBrokenBeforeAnyOutputIsRetried(t *testing.T) {
	e, mock := retryEngine(AgentConfig{},
		llm.MockTurn{Err: errBroken, Usage: &llm.Usage{InputTokens: 7}},
		llm.MockTurn{Text: "ok", Usage: &llm.Usage{InputTokens: 10, OutputTokens: 5}},
	)
	text, last := collect(t, e)
	if text != "ok" || last.Type != ChunkTypeDone {
		t.Fatalf("the retry must deliver the answer: %q, last chunk %+v", text, last)
	}
	if n := len(mock.Requests()); n != 2 {
		t.Fatalf("want 2 model calls, got %d", n)
	}
	// The failed attempt was billed too.
	if last.Usage == nil || last.Usage.InputTokens != 17 {
		t.Fatalf("usage must include the broken attempt: %+v", last.Usage)
	}
	if h := e.sessions.GetHistory("s"); len(h) != 2 {
		t.Fatalf("history must hold one exchange, got %d messages", len(h))
	}
}

func TestStreamRetriesAreBounded(t *testing.T) {
	e, mock := retryEngine(AgentConfig{}, llm.MockTurn{Err: errBroken})
	_, last := collect(t, e)
	if last.Type != ChunkTypeError {
		t.Fatalf("a stream that keeps breaking must end in an error, got %+v", last)
	}
	if n := len(mock.Requests()); n != 1+DefaultStreamRetries {
		t.Fatalf("want %d calls, got %d", 1+DefaultStreamRetries, n)
	}
}

func TestStreamRetriesCanBeTurnedOff(t *testing.T) {
	e, mock := retryEngine(AgentConfig{StreamRetries: -1}, llm.MockTurn{Err: errBroken})
	if _, last := collect(t, e); last.Type != ChunkTypeError {
		t.Fatalf("got %+v", last)
	}
	if n := len(mock.Requests()); n != 1 {
		t.Fatalf("want 1 call, got %d", n)
	}
}

// An error the API answered with is not a broken stream: no repeat.
func TestAPIErrorsAreNotRetried(t *testing.T) {
	e, mock := retryEngine(AgentConfig{}, llm.MockTurn{Err: errors.New("400 bad request")})
	if _, last := collect(t, e); last.Type != ChunkTypeError {
		t.Fatalf("got %+v", last)
	}
	if n := len(mock.Requests()); n != 1 {
		t.Fatalf("a rejected request must not be repeated, got %d calls", n)
	}
}
