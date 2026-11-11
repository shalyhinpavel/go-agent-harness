package config

import "testing"

func TestGuardDefaults(t *testing.T) {
	cfg := Default()
	if cfg.Agent.MaxParallelTools != 8 {
		t.Errorf("default max_parallel_tools = %d, want 8", cfg.Agent.MaxParallelTools)
	}
	if cfg.Agent.MaxToolResultBytes != 32768 {
		t.Errorf("default max_tool_result_bytes = %d, want 32768", cfg.Agent.MaxToolResultBytes)
	}
	if cfg.Trace.Dir != "" {
		t.Errorf("tracing must be off by default, got %q", cfg.Trace.Dir)
	}

	// An empty file (all zero values) must be repaired the same way.
	empty := Config{}
	empty.fillDefaults()
	if empty.Agent.MaxParallelTools != 8 || empty.Agent.MaxToolResultBytes != 32768 {
		t.Fatalf("fillDefaults did not repair guard zeros: %+v", empty.Agent)
	}
}
