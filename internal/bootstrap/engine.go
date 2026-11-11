package bootstrap

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/internal/config"
	"github.com/shalyhinpavel/go-agent-harness/internal/prompts"
	"github.com/shalyhinpavel/go-agent-harness/llm"
	"github.com/shalyhinpavel/go-agent-harness/mcpclient"
	"github.com/shalyhinpavel/go-agent-harness/memory"
	"github.com/shalyhinpavel/go-agent-harness/tools"
	"github.com/shalyhinpavel/go-agent-harness/trace"
)

// Harness is a fully wired engine plus the pieces commands need for
// logging and teardown.
type Harness struct {
	Engine   *agent.AgentEngine
	Provider llm.LLMProvider
	Registry *agent.ToolRegistry
	Sessions *agent.SessionManager
	Memory   memory.MemoryStore // nil when memory is off
	Trace    *trace.DirSink     // nil when tracing is off
	MCP      *mcpclient.Pool    // nil when no MCP server is configured
	Config   config.Config
}

// Close releases harness resources (trace files, MCP connections).
func (h *Harness) Close() {
	if h.Trace != nil {
		h.Trace.Close()
	}
	h.MCP.Close()
}

// mcpConnectTimeout bounds connecting to and listing every MCP server.
const mcpConnectTimeout = 30 * time.Second

// BuildEngine assembles the engine from cfg: provider, prompt, tools and
// the memory access ports (injection and/or memory_search tool). The
// memory engine itself is out of scope here — both ports are attach
// points for a future one.
func BuildEngine(cfg config.Config) (_ *Harness, retErr error) {
	provider, err := Provider(cfg.Provider)
	if err != nil {
		return nil, fmt.Errorf("bootstrap: provider: %w", err)
	}

	// The default system prompt promises tools ("You are a helpful
	// assistant with access to tools"), so it can only be chosen once the
	// registry is complete: a run with nothing registered would be told to
	// use tools it does not have and answers with an empty tool-call turn.
	registry := agent.NewToolRegistry()

	// Memory wiring — the two access ports.
	var store memory.MemoryStore
	var augmenter agent.PromptAugmenter

	if mode := cfg.Memory.MemoryMode(); mode != "off" {
		switch cfg.Memory.Store {
		case "in-memory", "":
			store = memory.NewInMemoryStore()
		case "files":
			store, err = memory.NewFileStore(cfg.Memory.Dir)
			if err != nil {
				return nil, fmt.Errorf("bootstrap: file store: %w", err)
			}
		default:
			return nil, fmt.Errorf("bootstrap: unknown memory store %q", cfg.Memory.Store)
		}

		if mode == "injection" || mode == "both" {
			// Injection port: layer the dossier on top of the system prompt
			// each turn. Fact retrieval joins when a real engine lands.
			storeRef := store
			augmenter = func(ctx context.Context, userID, sessionID, base string) (string, error) {
				dossier, err := storeRef.GetDossier(ctx, userID)
				if err != nil {
					return "", fmt.Errorf("read dossier: %w", err)
				}
				if dossier == nil || dossier.Content == "" {
					return base, nil
				}
				return memory.BuildAugmentedSystemPrompt(base, dossier, nil), nil
			}
		}

		if mode == "tool" || mode == "both" {
			if err := registry.Register(memory.NewSearchTool(store, nil)); err != nil {
				return nil, fmt.Errorf("bootstrap: memory tool: %w", err)
			}
		}
	}

	// The bash tool is dangerous by design: registering it requires an
	// explicit opt-in next to naming it in tools.enabled.
	for _, name := range cfg.Tools.Enabled {
		if name == tools.BashName && !cfg.Tools.Bash.Enabled {
			return nil, fmt.Errorf("bootstrap: tool %q is opt-in — set tools.bash.enabled: true", name)
		}
	}

	if err := tools.RegisterSelected(registry, cfg.Tools.Enabled, tools.Options{
		FilesDir:      cfg.Tools.Files.Dir,
		WebFetch:      cfg.Tools.WebFetch,
		ShellTimeoutS: cfg.Tools.Bash.TimeoutS,
	}); err != nil {
		return nil, err
	}

	// MCP servers: their tools join the registry like catalogue tools.
	var mcpPool *mcpclient.Pool
	if len(cfg.Tools.MCPServers) > 0 {
		ctx, cancel := context.WithTimeout(context.Background(), mcpConnectTimeout)
		mcpPool, err = mcpclient.ConnectAll(ctx, cfg.Tools.MCPServers, registry)
		cancel()
		if err != nil {
			return nil, fmt.Errorf("bootstrap: %w", err)
		}
		// A later wiring error must not leak the connections (or a stdio
		// server process).
		defer func() {
			if retErr != nil {
				mcpPool.Close()
			}
		}()
	}

	// Trajectory recording: one JSONL file per session ("" = off).
	var traceSink *trace.DirSink
	if cfg.Trace.Dir != "" {
		traceSink, err = trace.NewDirSink(cfg.Trace.Dir)
		if err != nil {
			return nil, fmt.Errorf("bootstrap: trace dir: %w", err)
		}
		// Later steps can still fail: do not leave the files open.
		defer func() {
			if retErr != nil {
				traceSink.Close()
			}
		}()
	}

	ttl, err := cfg.Agent.TTL()
	if err != nil {
		return nil, err
	}

	// System prompt: inline wins over the prompt file, and the file follows
	// the tools (see SystemPromptName).
	systemPrompt := cfg.Agent.SystemPrompt
	if systemPrompt == "" {
		name := SystemPromptName(cfg, len(registry.List()) > 0)
		systemPrompt, err = prompts.New(cfg.Prompts.Dir).Load(name)
		if err != nil {
			return nil, fmt.Errorf("bootstrap: system prompt: %w", err)
		}
	}

	sessionOpts := []agent.SessionOption{
		agent.WithSessionTTL(ttl),
		agent.WithMaxSessions(cfg.Agent.MaxSessions),
	}
	if cfg.Agent.SessionDir != "" {
		store, err := agent.NewFileSessionStore(cfg.Agent.SessionDir)
		if err != nil {
			return nil, fmt.Errorf("bootstrap: %w", err)
		}
		sessionOpts = append(sessionOpts, agent.WithSessionStore(store))
	}
	sessions := agent.NewSessionManager(sessionOpts...)

	engineCfg := agent.AgentConfig{
		Provider:           provider,
		ModelName:          cfg.Provider.Model,
		SystemPrompt:       systemPrompt,
		PromptAugmenter:    augmenter,
		MaxIterations:      cfg.Agent.MaxIterations,
		HistoryWindow:      cfg.Agent.HistoryWindow,
		HistoryChars:       cfg.Agent.HistoryChars,
		HistoryTrimTo:      cfg.Agent.HistoryTrimTo,
		PinTask:            cfg.Agent.HistoryPinTask,
		MaxTokens:          cfg.Agent.MaxTokens,
		MaxParallelTools:   cfg.Agent.MaxParallelTools,
		MaxToolResultBytes: cfg.Agent.MaxToolResultBytes,
		ContextLimit:       cfg.Provider.ContextLimit,
		PreserveReasoning:  cfg.Agent.ReasoningPreserved(),
		TraceRequests:      cfg.Trace.Requests,
		AutoCompact:        cfg.CompactEnabled(),
	}
	// Assign the sink only when tracing is on: a typed nil inside the
	// interface would make the nil check in the engine pass.
	if traceSink != nil {
		engineCfg.TraceSink = traceSink
	}
	engine := agent.NewAgentEngine(engineCfg, registry, sessions)

	slog.Info("harness assembled",
		"provider", provider.Name(),
		"model", cfg.Provider.Model,
		"memory_mode", cfg.Memory.MemoryMode(),
		"trace_dir", cfg.Trace.Dir,
		"tools", len(registry.List()))

	return &Harness{
		Engine:   engine,
		Provider: provider,
		Registry: registry,
		Sessions: sessions,
		Memory:   store,
		Trace:    traceSink,
		MCP:      mcpPool,
		Config:   cfg,
	}, nil
}

// Prompt files that carry the tool situation. The default promises tools; a
// run with no tools must not be told that, or a reasoning model ends its turn
// on a tool-call token it cannot place and delivers nothing.
const (
	defaultSystemPrompt = "system.md"
	noToolsSystemPrompt = "system_notools.md"
)

// SystemPromptName picks the prompt file for the run's tools. A file the user
// named explicitly is always kept: the fallback is only for the default.
func SystemPromptName(cfg config.Config, hasTools bool) string {
	if hasTools || cfg.Agent.SystemPromptFile != defaultSystemPrompt {
		return cfg.Agent.SystemPromptFile
	}
	return noToolsSystemPrompt
}
