// Package config loads the harness configuration from YAML with
// environment overrides. Defaults make the harness runnable with no
// config file at all.
package config

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/shalyhinpavel/go-agent-harness/mcpclient"
	"github.com/shalyhinpavel/go-agent-harness/tools"
)

// Config is the whole harness configuration.
type Config struct {
	Server   Server   `yaml:"server"`
	Provider Provider `yaml:"provider"`
	Agent    Agent    `yaml:"agent"`
	Tools    Tools    `yaml:"tools"`
	Memory   Memory   `yaml:"memory"`
	Prompts  Prompts  `yaml:"prompts"`
	Trace    Trace    `yaml:"trace"`
}

// Server holds HTTP server settings. The API has no authentication, so
// the defaults keep it local: loopback bind, no CORS.
type Server struct {
	// Host is the bind address. 0.0.0.0 exposes the unauthenticated API to
	// the network — do that only behind a proxy that authenticates.
	Host string `yaml:"host"`
	Port int    `yaml:"port"`
	// CORSOrigins lists the browser origins allowed to call the API;
	// empty = no CORS headers, cross-origin browser calls are refused.
	CORSOrigins []string `yaml:"cors_origins"`
	// MaxBodyMB caps a request body; attachments ride inside it base64.
	MaxBodyMB int `yaml:"max_body_mb"`
	// RequireApproval pauses non-read-only tool calls until confirmed via
	// POST /v1/chat/approve.
	RequireApproval bool `yaml:"require_approval"`
	// ApprovalTimeout bounds how long a tool call waits for approval
	// before being denied ("" = 5m).
	ApprovalTimeout string `yaml:"approval_timeout"`
}

// Provider selects and parameterizes the LLM backend.
type Provider struct {
	// Kind: auto | anthropic | openai | openai-compat | openrouter | mock.
	Kind string `yaml:"kind"`
	// Model is the default model id ("" = provider default).
	Model string `yaml:"model"`
	// APIKeyEnv names the environment variable holding the API key.
	APIKeyEnv string `yaml:"api_key_env"`
	// BaseURLEnv names the environment variable holding a custom base URL
	// (openai-compat); the literal value is used verbatim if set here instead.
	BaseURL    string `yaml:"base_url"`
	BaseURLEnv string `yaml:"base_url_env"`
	// ContextLimit is the model's context window in tokens; the engine
	// stops with a clear error before a request would exceed it (0 = off).
	ContextLimit int `yaml:"context_limit"`
	// Params go out with every request of this provider — agent turns,
	// sub-agents and structured calls alike, so one model runs under one
	// setting in every role. OpenAI-compatible kinds only for now.
	Params ProviderParams `yaml:"params"`
}

// ProviderParams are sampling and reasoning knobs; unset = backend default.
type ProviderParams struct {
	Temperature *float64 `yaml:"temperature"`
	TopP        *float64 `yaml:"top_p"`
	TopK        *int     `yaml:"top_k"`
	// ReasoningEffort is sent as reasoning_effort; the accepted values
	// are the model's own (e.g. low / medium / high / xhigh).
	ReasoningEffort string `yaml:"reasoning_effort"`
	// ExtraBody is merged into the request body last and wins over the
	// fields above: backend-specific fields go here (vLLM's
	// chat_template_kwargs, OpenRouter's reasoning or provider routing).
	ExtraBody map[string]any `yaml:"extra_body"`
}

// IsZero reports whether no knob is set.
func (p ProviderParams) IsZero() bool {
	return p.Temperature == nil && p.TopP == nil && p.TopK == nil &&
		p.ReasoningEffort == "" && len(p.ExtraBody) == 0
}

// Agent tunes the engine loop.
type Agent struct {
	// SystemPromptFile points into the prompts dir ("system.md"); inline
	// SystemPrompt wins when non-empty.
	SystemPrompt     string `yaml:"system_prompt"`
	SystemPromptFile string `yaml:"system_prompt_file"`
	MaxIterations    int    `yaml:"max_iterations"`
	HistoryWindow    int    `yaml:"history_window"`
	// HistoryChars caps the history window by approximate payload size
	// (0 = messages only). Set it for long tool loops.
	HistoryChars int `yaml:"history_chars"`
	// HistoryTrimTo is the fraction of the caps an overflowing window is
	// cut down to (0 = 0.75, 1 = slide every request); a lower value keeps
	// the request prefix stable, and cacheable, for longer.
	HistoryTrimTo float64 `yaml:"history_trim_to"`
	// HistoryPinTask keeps the first user message of a session (the task)
	// at the front of the window.
	HistoryPinTask bool   `yaml:"history_pin_task"`
	SessionTTL     string `yaml:"session_ttl"`
	MaxSessions    int    `yaml:"max_sessions"`
	// SessionDir keeps conversations as JSONL files, one per session, so
	// they survive a restart ("" = in memory only). With it set, the TTL and
	// the session cap only drop the in-memory copy.
	SessionDir string `yaml:"session_dir"`
	// MaxTokens caps the response size (0 = provider default).
	MaxTokens int `yaml:"max_tokens"`
	// MaxParallelTools caps concurrent tool executions per turn (0 = 8).
	MaxParallelTools int `yaml:"max_parallel_tools"`
	// MaxToolResultBytes truncates oversized tool results for the model
	// (0 = 32768); the client stream keeps the full payload.
	MaxToolResultBytes int `yaml:"max_tool_result_bytes"`
	// PreserveReasoning keeps the model's thinking in the conversation so
	// reasoning-capable models continue their chain of thought across
	// turns. Default true (nil); stripped automatically when the history
	// window compacts.
	PreserveReasoning *bool `yaml:"preserve_reasoning"`
	// AutoCompact summarizes dropped history when the context guard fires.
	// nil = on whenever provider.context_limit is set; false turns it off.
	AutoCompact *bool `yaml:"auto_compact"`
}

// Tools selects the toolsets to register and tunes their options.
type Tools struct {
	// Enabled lists toolset names from the catalogue (package tools).
	Enabled []string `yaml:"enabled"`
	// Files configures the file tools (session workspaces).
	Files FilesOpts `yaml:"files"`
	// WebFetch configures the web_fetch tool.
	WebFetch tools.WebFetchConfig `yaml:"web_fetch"`
	// Bash gates the bash tool — dangerous, opt-in.
	Bash BashOpts `yaml:"bash"`
	// MCPServers are Model Context Protocol servers whose tools join the
	// registry at startup (see package mcpclient).
	MCPServers []mcpclient.ServerConfig `yaml:"mcp_servers"`
}

// FilesOpts is the file tools configuration.
type FilesOpts struct {
	// Dir is the workspace root; each session gets a subdirectory.
	Dir string `yaml:"dir"`
}

// BashOpts is the bash tool configuration.
type BashOpts struct {
	// Enabled must be true to allow registering the bash tool.
	Enabled bool `yaml:"enabled"`
	// TimeoutS bounds each command.
	TimeoutS int `yaml:"timeout_s"`
}

// Memory controls the (optional) memory access points. The memory engine
// itself is not part of the core; both modes are ports for a future engine.
type Memory struct {
	// Enabled turns memory wiring on. When false neither the augmenter
	// nor the memory tools are attached.
	Enabled bool `yaml:"enabled"`
	// Mode: injection (augment the system prompt) | tool (expose
	// memory_search to the model) | both.
	Mode string `yaml:"mode"`
	// Store: in-memory (ephemeral) | files (markdown notes, see Dir).
	Store string `yaml:"store"`
	// Dir is the file store root (store: files), one subdir per user.
	Dir string `yaml:"dir"`
}

// Prompts configures prompt loading.
type Prompts struct {
	// Dir is searched first; embedded defaults serve anything missing.
	Dir string `yaml:"dir"`
}

// Trace configures trajectory recording (JSONL sidecar per session).
type Trace struct {
	// Dir writes one <session>.jsonl per session; "" disables tracing.
	Dir string `yaml:"dir"`
	// Requests records each turn's final model request on the trace — the
	// exact messages the answer was produced from, not a reconstruction.
	// Off by default: it repeats the turn's history in the file.
	Requests bool `yaml:"requests"`
}

// Default returns the built-in configuration.
func Default() Config {
	return Config{
		Server: Server{Host: "127.0.0.1", Port: 8080, MaxBodyMB: 20, ApprovalTimeout: "5m"},
		Provider: Provider{
			Kind:      "auto",
			APIKeyEnv: "",
		},
		Agent: Agent{
			SystemPromptFile:   "system.md",
			MaxIterations:      10,
			HistoryWindow:      40,
			SessionTTL:         "24h",
			MaxSessions:        1000,
			MaxParallelTools:   8,
			MaxToolResultBytes: 32768,
		},
		Tools: Tools{
			Enabled: []string{"web_fetch", "read_file", "write_file", "list_files"},
			Files:   FilesOpts{Dir: "data/workspace"},
			WebFetch: tools.WebFetchConfig{
				TimeoutS:     15,
				MaxBytes:     2 * 1024 * 1024,
				PageChars:    12000,
				BlockPrivate: true,
			},
			Bash: BashOpts{Enabled: false, TimeoutS: 60},
		},
		Memory: Memory{
			Enabled: false,
			Mode:    "both",
			Store:   "in-memory",
			Dir:     "data/memory",
		},
		Prompts: Prompts{Dir: "prompts"},
		Trace:   Trace{Dir: ""},
	}
}

// CompactEnabled resolves agent.auto_compact: on whenever a context limit is
// set, unless the file turns it off.
func (c Config) CompactEnabled() bool {
	if c.Provider.ContextLimit <= 0 {
		return false
	}
	return c.Agent.AutoCompact == nil || *c.Agent.AutoCompact
}

// Load reads path (falling back to defaults when it does not exist), then
// applies environment overrides. A malformed file is an error, not a silent
// default: misconfiguration must fail loudly.
func Load(path string) (Config, error) {
	cfg := Default()

	if path != "" {
		data, err := os.ReadFile(path)
		if err == nil {
			// Unknown keys are errors: a typo in a key must not silently
			// leave the default in force.
			dec := yaml.NewDecoder(bytes.NewReader(data))
			dec.KnownFields(true)
			if err := dec.Decode(&cfg); err != nil && !errors.Is(err, io.EOF) {
				return cfg, fmt.Errorf("config: parse %s: %w", path, err)
			}
		} else if !os.IsNotExist(err) {
			return cfg, fmt.Errorf("config: read %s: %w", path, err)
		}
	}

	cfg.applyEnv()
	cfg.fillDefaults()
	if err := cfg.validate(); err != nil {
		return cfg, err
	}
	return cfg, nil
}

// validate rejects enum values the code would otherwise quietly read as
// a default: a typo in a mode must not silently run another setup.
func (c *Config) validate() error {
	if _, err := c.Server.ParsedApprovalTimeout(); err != nil {
		return err
	}
	if t := c.Provider.Params.Temperature; t != nil && *t < 0 {
		return fmt.Errorf("config: provider.params.temperature %v is negative", *t)
	}
	if p := c.Provider.Params.TopP; p != nil && (*p <= 0 || *p > 1) {
		return fmt.Errorf("config: provider.params.top_p %v is outside (0, 1]", *p)
	}
	switch c.Memory.Mode {
	case "injection", "tool", "both":
	default:
		return fmt.Errorf("config: memory.mode %q: want injection, tool or both", c.Memory.Mode)
	}
	switch c.Memory.Store {
	case "in-memory", "files":
	default:
		return fmt.Errorf("config: memory.store %q: want in-memory or files", c.Memory.Store)
	}
	return nil
}

// applyEnv lets environment variables win over the file — convenient for
// containers and one-off runs (AGENT_PROVIDER, AGENT_MODEL, AGENT_HOST,
// PORT, AGENT_BASE_URL; API keys come from the variables the provider
// section names).
func (c *Config) applyEnv() {
	if v := os.Getenv("AGENT_HOST"); v != "" {
		c.Server.Host = v
	}
	if v := os.Getenv("AGENT_PROVIDER"); v != "" {
		c.Provider.Kind = v
	}
	if v := os.Getenv("AGENT_MODEL"); v != "" {
		c.Provider.Model = v
	}
	if v := os.Getenv("PORT"); v != "" {
		if port, err := strconv.Atoi(v); err == nil {
			c.Server.Port = port
		}
	}
	if v := os.Getenv("AGENT_BASE_URL"); v != "" && c.Provider.BaseURLEnv == "" {
		c.Provider.BaseURL = v
	}
}

// fillDefaults repairs zero values after file parsing.
func (c *Config) fillDefaults() {
	d := Default()
	if c.Server.Port == 0 {
		c.Server.Port = d.Server.Port
	}
	if c.Server.Host == "" {
		c.Server.Host = d.Server.Host
	}
	if c.Server.MaxBodyMB <= 0 {
		c.Server.MaxBodyMB = d.Server.MaxBodyMB
	}
	if c.Server.ApprovalTimeout == "" {
		c.Server.ApprovalTimeout = d.Server.ApprovalTimeout
	}
	if c.Provider.Kind == "" {
		c.Provider.Kind = d.Provider.Kind
	}
	if c.Agent.SystemPromptFile == "" && c.Agent.SystemPrompt == "" {
		c.Agent.SystemPromptFile = d.Agent.SystemPromptFile
	}
	if c.Agent.MaxIterations <= 0 {
		c.Agent.MaxIterations = d.Agent.MaxIterations
	}
	if c.Agent.HistoryWindow <= 0 {
		c.Agent.HistoryWindow = d.Agent.HistoryWindow
	}
	if c.Agent.SessionTTL == "" {
		c.Agent.SessionTTL = d.Agent.SessionTTL
	}
	if c.Agent.MaxSessions <= 0 {
		c.Agent.MaxSessions = d.Agent.MaxSessions
	}
	if c.Agent.MaxParallelTools <= 0 {
		c.Agent.MaxParallelTools = 8
	}
	if c.Agent.MaxToolResultBytes <= 0 {
		c.Agent.MaxToolResultBytes = 32768
	}
	if c.Memory.Mode == "" {
		c.Memory.Mode = d.Memory.Mode
	}
	if c.Tools.Files.Dir == "" {
		c.Tools.Files.Dir = d.Tools.Files.Dir
	}
	if c.Tools.WebFetch.TimeoutS <= 0 {
		c.Tools.WebFetch.TimeoutS = d.Tools.WebFetch.TimeoutS
	}
	if c.Tools.WebFetch.MaxBytes <= 0 {
		c.Tools.WebFetch.MaxBytes = d.Tools.WebFetch.MaxBytes
	}
	if c.Tools.WebFetch.PageChars <= 0 {
		c.Tools.WebFetch.PageChars = d.Tools.WebFetch.PageChars
	}
	if c.Tools.Bash.TimeoutS <= 0 {
		c.Tools.Bash.TimeoutS = d.Tools.Bash.TimeoutS
	}
	if c.Memory.Store == "" {
		c.Memory.Store = d.Memory.Store
	}
	if c.Memory.Dir == "" {
		c.Memory.Dir = d.Memory.Dir
	}
	if c.Prompts.Dir == "" {
		c.Prompts.Dir = d.Prompts.Dir
	}
}

// ParsedApprovalTimeout parses Server.ApprovalTimeout; empty means 0 (server default).
func (s Server) ParsedApprovalTimeout() (time.Duration, error) {
	if s.ApprovalTimeout == "" {
		return 0, nil
	}
	d, err := time.ParseDuration(s.ApprovalTimeout)
	if err != nil || d <= 0 {
		if err == nil {
			err = errors.New("must be positive")
		}
		return 0, fmt.Errorf("config: bad server.approval_timeout %q: %w", s.ApprovalTimeout, err)
	}
	return d, nil
}

// ReasoningPreserved resolves the preserve_reasoning tri-state (nil = on).
func (a Agent) ReasoningPreserved() bool {
	if a.PreserveReasoning == nil {
		return true
	}
	return *a.PreserveReasoning
}

// TTL parses Agent.SessionTTL; zero duration means "no TTL".
func (a Agent) TTL() (time.Duration, error) {
	if a.SessionTTL == "" {
		return 0, nil
	}
	d, err := time.ParseDuration(a.SessionTTL)
	if err != nil {
		return 0, fmt.Errorf("config: bad session_ttl %q: %w", a.SessionTTL, err)
	}
	return d, nil
}

// MemoryMode returns the parsed memory mode ("both" when enabled and unset).
func (m Memory) MemoryMode() string {
	if !m.Enabled {
		return "off"
	}
	switch m.Mode {
	case "injection", "tool", "both":
		return m.Mode
	default:
		return "both"
	}
}
