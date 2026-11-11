package agent

import (
	"fmt"
	"sort"
	"sync"

	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// ToolRegistry holds the tools available to an engine.
type ToolRegistry struct {
	mu    sync.RWMutex
	tools map[string]*Tool
}

// NewToolRegistry creates an empty registry.
func NewToolRegistry() *ToolRegistry {
	return &ToolRegistry{tools: make(map[string]*Tool)}
}

// Register adds a tool. Names must be unique and non-empty, and the tool
// needs an Execute function: a missing one would only surface as a panic
// in the middle of a turn.
func (r *ToolRegistry) Register(t *Tool) error {
	if t == nil {
		return fmt.Errorf("agent: cannot register a nil tool")
	}
	if t.Name == "" {
		return fmt.Errorf("agent: tool name cannot be empty")
	}
	if t.Execute == nil {
		return fmt.Errorf("agent: tool %q has no Execute function", t.Name)
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.tools[t.Name]; exists {
		return fmt.Errorf("agent: tool %q is already registered", t.Name)
	}
	r.tools[t.Name] = t
	return nil
}

// Get returns a tool by name.
func (r *ToolRegistry) Get(name string) (*Tool, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	t, exists := r.tools[name]
	return t, exists
}

// List returns all registered tools in unspecified order.
func (r *ToolRegistry) List() []*Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make([]*Tool, 0, len(r.tools))
	for _, t := range r.tools {
		result = append(result, t)
	}
	return result
}

// ToolDefs returns tool declarations for the model, sorted by name so that
// prompts stay byte-stable across calls (better provider-side caching).
func (r *ToolRegistry) ToolDefs() []llm.ToolDef {
	r.mu.RLock()
	defer r.mu.RUnlock()

	defs := make([]llm.ToolDef, 0, len(r.tools))
	for _, t := range r.tools {
		defs = append(defs, llm.ToolDef{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  t.Parameters,
		})
	}
	sort.Slice(defs, func(i, j int) bool { return defs[i].Name < defs[j].Name })
	return defs
}
