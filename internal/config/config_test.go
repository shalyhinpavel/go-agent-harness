package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoad_MissingFileAppliesDefaults(t *testing.T) {
	cfg, err := Load("/nonexistent/config.yaml")
	if err != nil {
		t.Fatalf("missing file must fall back to defaults: %v", err)
	}
	if cfg.Provider.Kind != "auto" || cfg.Agent.MaxIterations != 10 || cfg.Server.Port != 8080 {
		t.Fatalf("defaults not applied: %+v", cfg)
	}
}

func TestLoad_ParsesYAML(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	yaml := `
server: {port: 9090}
provider: {kind: openai-compat, model: "qwen3:32b", api_key_env: MY_KEY, base_url: http://localhost:11434/v1}
agent:
  system_prompt: "inline prompt"
  max_iterations: 9
  history_window: 10
  session_ttl: 1h
  max_sessions: 42
tools:
  enabled: [web_fetch]
  files: {dir: /tmp/ws}
memory: {enabled: true, mode: tool, store: in-memory}
`
	if err := os.WriteFile(path, []byte(yaml), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Server.Port != 9090 || cfg.Provider.Kind != "openai-compat" || cfg.Provider.Model != "qwen3:32b" {
		t.Fatalf("yaml not parsed: %+v", cfg)
	}
	if cfg.Agent.MaxIterations != 9 || cfg.Agent.HistoryWindow != 10 || cfg.Agent.MaxSessions != 42 {
		t.Fatalf("agent section wrong: %+v", cfg.Agent)
	}
	if cfg.Agent.SystemPrompt != "inline prompt" {
		t.Fatalf("inline prompt lost: %q", cfg.Agent.SystemPrompt)
	}
	if len(cfg.Tools.Enabled) != 1 || cfg.Tools.Enabled[0] != "web_fetch" {
		t.Fatalf("tools wrong: %v", cfg.Tools.Enabled)
	}
	if cfg.Tools.Files.Dir != "/tmp/ws" {
		t.Fatalf("tools.files.dir wrong: %q", cfg.Tools.Files.Dir)
	}
	if !cfg.Memory.Enabled || cfg.Memory.Mode != "tool" {
		t.Fatalf("memory wrong: %+v", cfg.Memory)
	}

	ttl, err := cfg.Agent.TTL()
	if err != nil || ttl.Hours() != 1 {
		t.Fatalf("ttl = %v err=%v", ttl, err)
	}
}

func TestLoad_MalformedFileFails(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte("provider: [broken"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); err == nil {
		t.Fatal("malformed yaml must error")
	}
}

func TestEnvOverrides(t *testing.T) {
	t.Setenv("AGENT_PROVIDER", "openrouter")
	t.Setenv("AGENT_MODEL", "test/model")
	t.Setenv("PORT", "7777")

	cfg, err := Load("")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Provider.Kind != "openrouter" || cfg.Provider.Model != "test/model" || cfg.Server.Port != 7777 {
		t.Fatalf("env override not applied: %+v %+v", cfg.Provider, cfg.Server)
	}
}

func TestMemoryMode(t *testing.T) {
	cases := []struct {
		memory Memory
		want   string
	}{
		{Memory{Enabled: false, Mode: "both"}, "off"},
		{Memory{Enabled: true, Mode: "injection"}, "injection"},
		{Memory{Enabled: true, Mode: "tool"}, "tool"},
		{Memory{Enabled: true, Mode: "both"}, "both"},
		{Memory{Enabled: true, Mode: "garbage"}, "both"},
	}
	for _, c := range cases {
		if got := c.memory.MemoryMode(); got != c.want {
			t.Errorf("mode %q enabled=%v = %q, want %q", c.memory.Mode, c.memory.Enabled, got, c.want)
		}
	}
}

func TestLoad_ProviderParams(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	body := `provider:
  params:
    temperature: 1.0
    top_p: 0.95
    top_k: 20
    reasoning_effort: low
    extra_body:
      chat_template_kwargs:
        enable_thinking: true
`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	p := cfg.Provider.Params
	if p.Temperature == nil || *p.Temperature != 1.0 || p.TopP == nil || *p.TopP != 0.95 ||
		p.TopK == nil || *p.TopK != 20 || p.ReasoningEffort != "low" || p.IsZero() {
		t.Fatalf("params = %+v", p)
	}
	kw, ok := p.ExtraBody["chat_template_kwargs"].(map[string]any)
	if !ok || kw["enable_thinking"] != true {
		t.Fatalf("extra_body must decode to JSON-ready maps: %#v", p.ExtraBody)
	}

	cfg, _ = Load("")
	if !cfg.Provider.Params.IsZero() {
		t.Fatalf("no params by default, got %+v", cfg.Provider.Params)
	}
}

func TestLoad_ProviderParamsOutOfRangeFail(t *testing.T) {
	cases := map[string]string{
		"temperature": "provider:\n  params:\n    temperature: -0.1\n",
		"top_p":       "provider:\n  params:\n    top_p: 1.5\n",
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "config.yaml")
			if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
				t.Fatal(err)
			}
			if _, err := Load(path); err == nil || !strings.Contains(err.Error(), name) {
				t.Fatalf("want an error naming %s, got %v", name, err)
			}
		})
	}
}

func loadYAML(t *testing.T, body string) (Config, error) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return Load(path)
}

func TestLoad_UnknownKeyIsAnError(t *testing.T) {
	_, err := loadYAML(t, "agent:\n  max_iteratons: 9\n")
	if err == nil || !strings.Contains(err.Error(), "max_iteratons") {
		t.Fatalf("a mistyped key must fail and name the key, got %v", err)
	}
}

func TestLoad_EmptyFileIsDefaults(t *testing.T) {
	cfg, err := loadYAML(t, "# nothing here\n")
	if err != nil || cfg.Server.Port != 8080 {
		t.Fatalf("an empty file means defaults: %v %+v", err, cfg.Server)
	}
}

func TestLoad_MemoryEnumsAreValidated(t *testing.T) {
	if _, err := loadYAML(t, "memory: {enabled: true, mode: injecton}\n"); err == nil {
		t.Fatal("a mistyped memory.mode must fail")
	}
	if _, err := loadYAML(t, "memory: {enabled: true, store: file}\n"); err == nil {
		t.Fatal("a mistyped memory.store must fail")
	}
}

func TestLoad_MCPServerAndWebFetchKeys(t *testing.T) {
	cfg, err := loadYAML(t, `
tools:
  web_fetch: {timeout_s: 3, block_private: false}
  mcp_servers:
    - {name: fs, command: npx, args: [server], env: {TOKEN: t}, tool_prefix: fs_}
`)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Tools.WebFetch.TimeoutS != 3 || cfg.Tools.WebFetch.BlockPrivate {
		t.Fatalf("web_fetch: %+v", cfg.Tools.WebFetch)
	}
	if s := cfg.Tools.MCPServers; len(s) != 1 || s[0].Command != "npx" || s[0].Env["TOKEN"] != "t" || s[0].ToolPrefix != "fs_" {
		t.Fatalf("mcp_servers: %+v", cfg.Tools.MCPServers)
	}
}

// The shipped example config must stay loadable under strict key checking.
func TestLoad_ShippedConfig(t *testing.T) {
	if _, err := Load("../../configs/config.yaml"); err != nil {
		t.Fatalf("configs/config.yaml: %v", err)
	}
}

func TestLoad_ServerApprovalConfig(t *testing.T) {
	cfg, err := loadYAML(t, "server: {require_approval: true, approval_timeout: 2m}\n")
	if err != nil {
		t.Fatal(err)
	}
	d, err := cfg.Server.ParsedApprovalTimeout()
	if err != nil || !cfg.Server.RequireApproval || d.Minutes() != 2 {
		t.Fatalf("server approval = %+v (%v, %v)", cfg.Server, d, err)
	}
	if _, err := loadYAML(t, "server: {approval_timeout: nope}\n"); err == nil {
		t.Fatal("invalid server.approval_timeout must fail")
	}
	if _, err := loadYAML(t, "server: {approval_timeout: -1s}\n"); err == nil {
		t.Fatal("negative server.approval_timeout must fail")
	}
}
