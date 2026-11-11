package bootstrap

import (
	"context"
	"strings"
	"testing"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/internal/config"
	"github.com/shalyhinpavel/go-agent-harness/llm"
	"github.com/shalyhinpavel/go-agent-harness/memory"
)

// buildHarness builds a mock-provider harness with the given mutations.
func buildHarness(t *testing.T, mut func(c *config.Config)) *Harness {
	t.Helper()

	cfg := config.Default()
	cfg.Provider.Kind = "mock"
	cfg.Prompts.Dir = "" // embedded defaults
	if mut != nil {
		mut(&cfg)
	}

	h, err := BuildEngine(cfg)
	if err != nil {
		t.Fatalf("BuildEngine: %v", err)
	}
	return h
}

func mockOf(t *testing.T, h *Harness) *llm.MockProvider {
	t.Helper()
	m, ok := h.Provider.(*llm.MockProvider)
	if !ok {
		t.Fatalf("expected MockProvider, got %T", h.Provider)
	}
	return m
}

func TestBuildEngine_MemoryOff(t *testing.T) {
	h := buildHarness(t, nil)

	if _, ok := h.Registry.Get("memory_search"); ok {
		t.Fatal("memory tool must not be registered when memory is off")
	}
	if h.Memory != nil {
		t.Fatal("memory store must not be wired when memory is off")
	}
}

func TestBuildEngine_MemoryToolPort(t *testing.T) {
	h := buildHarness(t, func(c *config.Config) {
		c.Memory.Enabled = true
		c.Memory.Mode = "tool"
	})

	tool, ok := h.Registry.Get("memory_search")
	if !ok {
		t.Fatal("memory_search tool must be registered in tool mode")
	}

	// Seed a fact and query through the tool.
	if err := h.Memory.InsertFact(context.Background(), &memory.UserFact{
		ID: "f1", UserID: "u1", Category: "work", Fact: "Works with the Go programming language", Confidence: 0.9,
	}); err != nil {
		t.Fatal(err)
	}

	res, err := tool.Execute(context.Background(), &agent.ToolContext{UserID: "u1"},
		map[string]any{"query": "programming language"})
	if err != nil {
		t.Fatal(err)
	}
	facts := res.Data.(map[string]any)["facts"].([]map[string]any)
	if len(facts) != 1 || facts[0]["fact"] != "Works with the Go programming language" {
		t.Fatalf("tool returned wrong facts: %+v", facts)
	}
}

func TestBuildEngine_MemoryInjectionPort(t *testing.T) {
	h := buildHarness(t, func(c *config.Config) {
		c.Memory.Enabled = true
		c.Memory.Mode = "injection"
	})
	mock := mockOf(t, h)

	// Seed a dossier, run a turn: the augmenter must layer the dossier
	// onto the system prompt the provider receives.
	if err := h.Memory.SaveDossier(context.Background(), &memory.UserDossier{
		UserID: "u1", Content: "Go developer from Berlin",
	}); err != nil {
		t.Fatal(err)
	}

	if _, err := h.Engine.Chat(context.Background(), agent.ChatRequest{
		UserID: "u1", SessionID: "s1", Message: "hi",
	}); err != nil {
		t.Fatal(err)
	}

	reqs := mock.Requests()
	if len(reqs) == 0 {
		t.Fatal("no provider call recorded")
	}
	sys := reqs[0].System
	if !strings.Contains(sys, "Go developer from Berlin") || !strings.Contains(sys, "USER DOSSIER") {
		t.Fatalf("system prompt not augmented: %q", sys)
	}
}

func TestBuildEngine_BothPorts(t *testing.T) {
	h := buildHarness(t, func(c *config.Config) {
		c.Memory.Enabled = true
		c.Memory.Mode = "both"
	})
	if _, ok := h.Registry.Get("memory_search"); !ok {
		t.Fatal("memory_search missing in both mode")
	}
}

func TestBuildEngine_UnknownToolsetFails(t *testing.T) {
	cfg := config.Default()
	cfg.Provider.Kind = "mock"
	cfg.Tools.Enabled = []string{"web_fetch", "bogus"}
	if _, err := BuildEngine(cfg); err == nil {
		t.Fatal("unknown toolset must fail the build")
	}
}

func TestBuildEngine_FileStore(t *testing.T) {
	dir := t.TempDir()
	h := buildHarness(t, func(c *config.Config) {
		c.Memory.Enabled = true
		c.Memory.Mode = "tool"
		c.Memory.Store = "files"
		c.Memory.Dir = dir
	})

	if _, ok := h.Registry.Get("memory_search"); !ok {
		t.Fatal("memory_search missing with the file store")
	}

	if err := h.Memory.InsertFact(context.Background(), &memory.UserFact{
		ID: "f1", UserID: "u1", Category: "work", Fact: "Works with the Go programming language", Confidence: 0.9,
	}); err != nil {
		t.Fatal(err)
	}

	tool, _ := h.Registry.Get("memory_search")
	res, err := tool.Execute(context.Background(), &agent.ToolContext{UserID: "u1"},
		map[string]any{"query": "programming language"})
	if err != nil {
		t.Fatal(err)
	}
	facts := res.Data.(map[string]any)["facts"].([]map[string]any)
	if len(facts) != 1 || facts[0]["fact"] != "Works with the Go programming language" {
		t.Fatalf("file store search returned wrong facts: %+v", facts)
	}
}

func TestBuildEngine_BashIsOptIn(t *testing.T) {
	cfg := config.Default()
	cfg.Tools.Enabled = []string{"bash"}
	cfg.Provider.Kind = "mock"
	if _, err := BuildEngine(cfg); err == nil || !strings.Contains(err.Error(), "opt-in") {
		t.Fatalf("bash without opt-in must fail loudly, got %v", err)
	}
	h := buildHarness(t, func(c *config.Config) {
		c.Tools.Enabled = []string{"bash"}
		c.Tools.Bash.Enabled = true
	})
	if _, ok := h.Registry.Get("bash"); !ok {
		t.Fatal("bash must register behind the opt-in flag")
	}
}

func TestSystemPromptName(t *testing.T) {
	// The default prompt promises tools; a tool-less run swaps it for the
	// tool-free default. A file the user named is never overridden.
	cases := []struct {
		name     string
		file     string
		hasTools bool
		want     string
	}{
		{"default, no tools", "system.md", false, "system_notools.md"},
		{"default, tools", "system.md", true, "system.md"},
		{"custom file, no tools", "team.md", false, "team.md"},
		{"custom file, tools", "team.md", true, "team.md"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var cfg config.Config
			cfg.Agent.SystemPromptFile = tc.file
			if got := SystemPromptName(cfg, tc.hasTools); got != tc.want {
				t.Fatalf("SystemPromptName(%q, %v) = %q, want %q", tc.file, tc.hasTools, got, tc.want)
			}
		})
	}
}

func TestBuildEngine_ToolLessRunGetsTheToolFreePrompt(t *testing.T) {
	// With nothing registered the lead used to be told it had tools; Qwen3.8
	// then opened a tool call it could not place and stopped with an empty
	// answer (P3 records: finish_reason stop, 30k-55k thinking tokens, no
	// content). The prompt the provider sees must not promise tools here.
	h := buildHarness(t, func(c *config.Config) { c.Tools.Enabled = nil })
	if n := len(h.Registry.List()); n != 0 {
		t.Fatalf("registry should be empty, has %d", n)
	}
	mock := mockOf(t, h)
	if _, err := h.Engine.Chat(context.Background(), agent.ChatRequest{
		UserID: "u1", SessionID: "s1", Message: "hi",
	}); err != nil {
		t.Fatal(err)
	}
	reqs := mock.Requests()
	if len(reqs) == 0 {
		t.Fatal("no provider call recorded")
	}
	if sys := reqs[0].System; !strings.Contains(sys, "no tools in this run") {
		t.Fatalf("tool-less run got a tool-promising prompt: %.160q", sys)
	}
}

func TestBuildEngine_WithToolsKeepsTheDefaultPrompt(t *testing.T) {
	h := buildHarness(t, nil) // the default toolset
	mock := mockOf(t, h)
	if _, err := h.Engine.Chat(context.Background(), agent.ChatRequest{
		UserID: "u1", SessionID: "s1", Message: "hi",
	}); err != nil {
		t.Fatal(err)
	}
	reqs := mock.Requests()
	if len(reqs) == 0 {
		t.Fatal("no provider call recorded")
	}
	if sys := reqs[0].System; !strings.Contains(sys, "with access to tools") {
		t.Fatalf("a run with tools lost the tool prompt: %.160q", sys)
	}
}

func TestBuildEngine_UnknownPromptIsAnErrorNotAPanic(t *testing.T) {
	cfg := config.Default()
	cfg.Provider.Kind = "mock"
	cfg.Agent.SystemPromptFile = "no-such-prompt.md"
	cfg.Trace.Dir = t.TempDir()
	if _, err := BuildEngine(cfg); err == nil || !strings.Contains(err.Error(), "system prompt") {
		t.Fatalf("a typo in the prompt file must be a build error, got %v", err)
	}
}

// With agent.session_dir a conversation survives a rebuilt harness.
func TestBuildEngine_SessionDirKeepsConversations(t *testing.T) {
	dir := t.TempDir()
	withDir := func(c *config.Config) { c.Agent.SessionDir = dir }

	h1 := buildHarness(t, withDir)
	if _, err := h1.Engine.Chat(context.Background(), agent.ChatRequest{UserID: "u", SessionID: "s", Message: "first"}); err != nil {
		t.Fatal(err)
	}
	h1.Close()

	h2 := buildHarness(t, withDir)
	if _, err := h2.Engine.Chat(context.Background(), agent.ChatRequest{UserID: "u", SessionID: "s", Message: "second"}); err != nil {
		t.Fatal(err)
	}
	reqs := mockOf(t, h2).Requests()
	if len(reqs) != 1 || len(reqs[0].Messages) != 3 {
		t.Fatalf("the rebuilt harness should see the stored exchange plus the new message, got %+v", reqs)
	}
}
