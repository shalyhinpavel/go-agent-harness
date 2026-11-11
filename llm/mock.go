package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
)

// MockTurn is one scripted completion.
type MockTurn struct {
	Text     string
	Thinking string
	// Reasoning scripts the turn's reasoning blocks, delivered as one
	// EventReasoning after the thinking deltas.
	Reasoning []Part
	ToolCalls []ToolCall
	// Usage is reported on the done event when set (nil = not reported); a
	// turn with Err reports it on the error event: what a broken call spent.
	Usage *Usage
	// StopReason rides on the done event ("" = derived: tool_use when the
	// turn calls tools, end otherwise).
	StopReason StopReason
	Err        error
}

// MockProvider is a deterministic provider for tests and CI: turns are
// consumed in order and the last one repeats. GenerateStructured is served
// by StructuredFn. Every request is recorded for assertions.
type MockProvider struct {
	mu sync.Mutex

	turns    []MockTurn
	fn       func(schema Schema, req *ChatRequest) (json.RawMessage, error)
	sUsage   Usage
	requests []ChatRequest
}

// NewMockProvider creates a scripted provider.
func NewMockProvider(turns ...MockTurn) *MockProvider {
	return &MockProvider{turns: turns}
}

// SetStructuredFn installs the structured-output handler.
func (m *MockProvider) SetStructuredFn(fn func(schema Schema, req *ChatRequest) (json.RawMessage, error)) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.fn = fn
}

// SetStructuredUsage sets the usage every GenerateStructured call reports.
func (m *MockProvider) SetStructuredUsage(u Usage) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.sUsage = u
}

// Requests returns a copy of all recorded requests, oldest first.
func (m *MockProvider) Requests() []ChatRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]ChatRequest, len(m.requests))
	copy(out, m.requests)
	return out
}

// Name implements LLMProvider.
func (m *MockProvider) Name() string { return "mock" }

func (m *MockProvider) record(req *ChatRequest) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.requests = append(m.requests, *req)
}

func (m *MockProvider) popTurn() MockTurn {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.turns) == 0 {
		return MockTurn{}
	}
	t := m.turns[0]
	if len(m.turns) > 1 {
		m.turns = m.turns[1:]
	}
	return t
}

// StreamChat implements LLMProvider.
func (m *MockProvider) StreamChat(ctx context.Context, req *ChatRequest) (<-chan StreamEvent, error) {
	m.record(req)
	turn := m.popTurn()

	stop := turn.StopReason
	if stop == StopUnknown {
		stop = StopEnd
		if len(turn.ToolCalls) > 0 {
			stop = StopToolUse
		}
	}

	out := make(chan StreamEvent, 8)
	go func() {
		defer close(out)
		emit := func(ev StreamEvent) bool {
			select {
			case out <- ev:
				return true
			case <-ctx.Done():
				return false
			}
		}
		if turn.Err != nil {
			emit(StreamEvent{Kind: EventError, Err: turn.Err, Usage: turn.Usage})
			return
		}
		if turn.Thinking != "" && !emit(StreamEvent{Kind: EventThinking, Text: turn.Thinking}) {
			return
		}
		if len(turn.Reasoning) > 0 && !emit(StreamEvent{Kind: EventReasoning, Reasoning: turn.Reasoning}) {
			return
		}
		if turn.Text != "" && !emit(StreamEvent{Kind: EventText, Text: turn.Text}) {
			return
		}
		if len(turn.ToolCalls) > 0 && !emit(StreamEvent{Kind: EventToolCalls, ToolCalls: turn.ToolCalls}) {
			return
		}
		emit(StreamEvent{Kind: EventDone, Usage: turn.Usage, StopReason: stop})
	}()
	return out, nil
}

// Generate implements LLMProvider.
func (m *MockProvider) Generate(ctx context.Context, req *ChatRequest) (string, Usage, error) {
	m.record(req)
	turn := m.popTurn()
	if turn.Err != nil {
		return "", Usage{}, turn.Err
	}
	var u Usage
	if turn.Usage != nil {
		u = *turn.Usage
	}
	return turn.Text, u, nil
}

// GenerateStructured implements LLMProvider.
func (m *MockProvider) GenerateStructured(ctx context.Context, req *ChatRequest, schema Schema, out any) (Usage, error) {
	m.record(req)

	m.mu.Lock()
	fn, u := m.fn, m.sUsage
	m.mu.Unlock()

	if fn == nil {
		return Usage{}, fmt.Errorf("mock: StructuredFn is not set")
	}
	raw, err := fn(schema, req)
	if err != nil {
		return u, err
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return u, fmt.Errorf("mock: decode structured response: %w", err)
	}
	return u, nil
}
