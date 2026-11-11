package agent

import (
	"context"
	"testing"

	"github.com/shalyhinpavel/go-agent-harness/llm"
)

func TestRegistry_RegisterAndGet(t *testing.T) {
	r := NewToolRegistry()
	tool := &Tool{Name: "t1", Execute: func(ctx context.Context, tc *ToolContext, args map[string]any) (*ToolResult, error) {
		return nil, nil
	}}

	if err := r.Register(tool); err != nil {
		t.Fatalf("register: %v", err)
	}
	got, ok := r.Get("t1")
	if !ok || got != tool {
		t.Fatalf("Get returned %+v, ok=%v", got, ok)
	}
}

func TestRegistry_DuplicateName(t *testing.T) {
	r := NewToolRegistry()
	mk := func() *Tool {
		return &Tool{Name: "dup", Execute: func(ctx context.Context, tc *ToolContext, args map[string]any) (*ToolResult, error) {
			return nil, nil
		}}
	}
	if err := r.Register(mk()); err != nil {
		t.Fatal(err)
	}
	if err := r.Register(mk()); err == nil {
		t.Fatal("duplicate registration must fail")
	}
}

func TestRegistry_InvalidNames(t *testing.T) {
	r := NewToolRegistry()
	if err := r.Register(nil); err == nil {
		t.Fatal("nil tool must fail")
	}
	if err := r.Register(&Tool{Name: ""}); err == nil {
		t.Fatal("empty name must fail")
	}
	// Without the check this registers fine and panics on the first call.
	if err := r.Register(&Tool{Name: "no_exec"}); err == nil {
		t.Fatal("a tool without Execute must fail")
	}
}

func TestRegistry_ToolDefsSorted(t *testing.T) {
	r := NewToolRegistry()
	for _, name := range []string{"charlie", "alpha", "bravo"} {
		n := name
		if err := r.Register(&Tool{
			Name:        n,
			Description: "d-" + n,
			Execute: func(ctx context.Context, tc *ToolContext, args map[string]any) (*ToolResult, error) {
				return nil, nil
			},
		}); err != nil {
			t.Fatal(err)
		}
	}

	defs := r.ToolDefs()
	if len(defs) != 3 {
		t.Fatalf("defs = %d, want 3", len(defs))
	}
	for i, want := range []string{"alpha", "bravo", "charlie"} {
		if defs[i].Name != want {
			t.Fatalf("defs[%d] = %s, want %s", i, defs[i].Name, want)
		}
	}
	if defs[0].Description != "d-alpha" {
		t.Fatalf("description not copied: %+v", defs[0])
	}
}

func TestRegistry_ToolDefsParametersSchema(t *testing.T) {
	r := NewToolRegistry()
	schema := llm.Schema{"type": "object", "properties": map[string]any{"x": map[string]any{"type": "number"}}}
	if err := r.Register(&Tool{
		Name:       "schemed",
		Parameters: schema,
		Execute: func(ctx context.Context, tc *ToolContext, args map[string]any) (*ToolResult, error) {
			return nil, nil
		},
	}); err != nil {
		t.Fatal(err)
	}

	defs := r.ToolDefs()
	if defs[0].Parameters["type"] != "object" {
		t.Fatalf("parameters not forwarded: %+v", defs[0].Parameters)
	}
}
