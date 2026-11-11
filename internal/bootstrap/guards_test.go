package bootstrap

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/internal/config"
)

func TestBuildEngine_TraceOffByDefault(t *testing.T) {
	h := buildHarness(t, nil)
	defer h.Close()
	if h.Trace != nil {
		t.Fatal("trace must be off unless trace.dir is set")
	}
}

func TestBuildEngine_TraceWritesJSONL(t *testing.T) {
	dir := t.TempDir()
	h := buildHarness(t, func(c *config.Config) { c.Trace.Dir = dir })
	defer h.Close()
	if h.Trace == nil {
		t.Fatal("trace sink must be wired when trace.dir is set")
	}

	if _, err := h.Engine.Chat(context.Background(), agent.ChatRequest{
		UserID: "u1", SessionID: "s1", Message: "hi",
	}); err != nil {
		t.Fatal(err)
	}

	if _, err := os.Stat(filepath.Join(dir, "s1.jsonl")); err != nil {
		t.Fatalf("session trace file: %v", err)
	}
}
