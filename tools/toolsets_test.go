package tools

import (
	"testing"

	"github.com/shalyhinpavel/go-agent-harness/agent"
)

func testOptions() Options {
	return Options{FilesDir: "workspace", ShellTimeoutS: 10}
}

func TestRegisterSelected(t *testing.T) {
	registry := agent.NewToolRegistry()
	if err := RegisterSelected(registry, []string{"web_fetch", "read_file"}, testOptions()); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"web_fetch", "read_file"} {
		if _, ok := registry.Get(name); !ok {
			t.Fatalf("%s not registered", name)
		}
	}
	// Only the named tools register — nothing sneaks in.
	if len(registry.List()) != 2 {
		t.Fatalf("registered %d tools, want 2", len(registry.List()))
	}
}

func TestRegisterSelected_UnknownFails(t *testing.T) {
	registry := agent.NewToolRegistry()
	if err := RegisterSelected(registry, []string{"nonexistent_tool"}, testOptions()); err == nil {
		t.Fatal("unknown toolset must fail loudly")
	}
}

func TestAvailable(t *testing.T) {
	avail := Available()
	found := map[string]bool{}
	for _, name := range avail {
		found[name] = true
	}
	for _, want := range []string{"web_fetch", "read_file", "write_file", "list_files", "bash"} {
		if !found[want] {
			t.Fatalf("Available() missing %q: %v", want, avail)
		}
	}
}
