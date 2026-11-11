package anthropic

import (
	"testing"

	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// TestConvertMessages_BatchStaysTogether: one assistant turn with text and
// parallel tool calls is one assistant message, and its results are one user
// message — the shape the Messages API documents for tool use.
func TestConvertMessages_BatchStaysTogether(t *testing.T) {
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
	got := convertMessages(req)
	if len(got) != 3 {
		var shape []string
		for _, m := range got {
			shape = append(shape, string(m.Role))
		}
		t.Fatalf("messages = %v, want [user assistant user]", shape)
	}
	if n := len(got[1].Content); n != 3 {
		t.Errorf("assistant blocks = %d, want text + 2 tool_use", n)
	}
	if n := len(got[2].Content); n != 2 {
		t.Errorf("user blocks = %d, want 2 tool_result", n)
	}
}
