package agent

import (
	"testing"

	"github.com/shalyhinpavel/go-agent-harness/llm"
)

func TestNewAgentEngine_Defaults(t *testing.T) {
	e := NewAgentEngine(AgentConfig{}, NewToolRegistry(), NewSessionManager())

	if e.maxIterations != DefaultMaxIterations {
		t.Errorf("maxIterations = %d, want %d", e.maxIterations, DefaultMaxIterations)
	}
	if e.window.MaxItems != DefaultHistoryWindow {
		t.Errorf("window = %d, want %d", e.window.MaxItems, DefaultHistoryWindow)
	}
}

func TestNewAgentEngine_CustomValues(t *testing.T) {
	cfg := AgentConfig{
		ModelName:     "custom-model",
		MaxIterations: 3,
		HistoryWindow: 10,
	}
	e := NewAgentEngine(cfg, NewToolRegistry(), NewSessionManager())

	if e.modelName != "custom-model" || e.maxIterations != 3 || e.window.MaxItems != 10 {
		t.Errorf("engine config not applied: %+v", e)
	}
}

func TestToolContextFields(t *testing.T) {
	tc := &ToolContext{UserID: "u1", SessionID: "s1"}
	if tc.UserID != "u1" || tc.SessionID != "s1" {
		t.Errorf("unexpected ToolContext: %+v", tc)
	}
}

func TestToolResultSerialization(t *testing.T) {
	// ToolResult must stay JSON-friendly for SSE transport.
	tr := ToolResult{Data: map[string]any{"k": "v"}}
	if tr.Error != "" || tr.Data == nil {
		t.Errorf("unexpected ToolResult: %+v", tr)
	}
}

func TestStreamChunkTypes(t *testing.T) {
	if ChunkTypeText != "text" || ChunkTypeDone != "done" {
		t.Error("stream chunk type constants changed unexpectedly")
	}
	_ = llm.PartText // ensure llm import stays for future use
}
