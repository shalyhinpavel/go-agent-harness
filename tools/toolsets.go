package tools

import (
	"fmt"
	"sort"

	"github.com/shalyhinpavel/go-agent-harness/agent"
)

// Factory builds a tool instance. Tools carrying stateful dependencies
// (like the memory store behind memory_search) are wired in bootstrap
// instead of this static catalogue.
type Factory func() *agent.Tool

// Options parameterize the configurable catalogue tools.
type Options struct {
	// FilesDir is the workspace root (per-session dirs) for the file tools
	// and bash.
	FilesDir string
	// WebFetch configures the web_fetch tool.
	WebFetch WebFetchConfig
	// ShellTimeoutS bounds bash commands.
	ShellTimeoutS int
}

// buildCatalogue returns the catalogue of built-in tools for the given
// options. Registering a tool here makes it selectable via the config
// `tools.enabled` list.
func buildCatalogue(opts Options) map[string]Factory {
	return map[string]Factory{
		WebFetchName:  func() *agent.Tool { return NewWebFetch(opts.WebFetch) },
		ReadFileName:  func() *agent.Tool { return NewReadFile(opts.FilesDir) },
		WriteFileName: func() *agent.Tool { return NewWriteFile(opts.FilesDir) },
		ListFilesName: func() *agent.Tool { return NewListFiles(opts.FilesDir) },
		BashName: func() *agent.Tool {
			return NewShell(ShellConfig{WorkspaceRoot: opts.FilesDir, TimeoutS: opts.ShellTimeoutS})
		},
	}
}

// Available lists the registered toolset names, sorted.
func Available() []string {
	names := make([]string, 0, len(buildCatalogue(Options{})))
	for name := range buildCatalogue(Options{}) {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// RegisterSelected registers the named toolsets into the registry.
// Unknown names fail loudly — a config typo must not silently drop a
// tool. The bash tool additionally requires an explicit opt-in
// (checked by the caller that owns the config).
func RegisterSelected(registry *agent.ToolRegistry, names []string, opts Options) error {
	catalogue := buildCatalogue(opts)
	for _, name := range names {
		factory, ok := catalogue[name]
		if !ok {
			return fmt.Errorf("tools: unknown toolset %q (available: %v)", name, Available())
		}
		if err := registry.Register(factory()); err != nil {
			return fmt.Errorf("tools: register %q: %w", name, err)
		}
	}
	return nil
}
