package openai

import (
	"encoding/json"
	"testing"

	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// roles renders a converted history as "role[:tool_calls]" for assertions on
// the order the API validates: an assistant message with tool_calls must be
// followed directly by one tool message per call.
func roles(t *testing.T, req *llm.ChatRequest) []string {
	t.Helper()
	raw, err := json.Marshal(convertMessages(req))
	if err != nil {
		t.Fatal(err)
	}
	var msgs []struct {
		Role      string            `json:"role"`
		Content   json.RawMessage   `json:"content"`
		ToolCalls []json.RawMessage `json:"tool_calls"`
	}
	if err := json.Unmarshal(raw, &msgs); err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, m := range msgs {
		s := m.Role
		if len(m.ToolCalls) > 0 {
			s += ":calls"
		}
		if len(m.Content) > 0 && string(m.Content) != "null" && m.Role == "assistant" {
			s += ":text"
		}
		out = append(out, s)
	}
	return out
}

func TestConvertMessages_TextAndToolCallsShareOneAssistantMessage(t *testing.T) {
	req := &llm.ChatRequest{Messages: []llm.Message{
		*llm.UserText("look it up"),
		{Role: llm.RoleAssistant, Parts: []llm.Part{
			llm.TextPart("Let me check."),
			llm.ToolCallPart(llm.ToolCall{ID: "c1", Name: "a"}),
			llm.ToolCallPart(llm.ToolCall{ID: "c2", Name: "b"}),
		}},
		{Role: llm.RoleTool, Parts: []llm.Part{
			llm.ToolResultPart(llm.ToolResultData{CallID: "c1", Name: "a", Content: 1}),
			llm.ToolResultPart(llm.ToolResultData{CallID: "c2", Name: "b", Content: 2}),
		}},
	}}
	got := roles(t, req)
	want := []string{"user", "assistant:calls:text", "tool", "tool"}
	if len(got) != len(want) {
		t.Fatalf("history = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("history = %v, want %v", got, want)
		}
	}
}
