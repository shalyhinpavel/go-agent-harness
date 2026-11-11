package config

import (
	"os"
	"path/filepath"
	"testing"
)

// A run with "no tools at all" is legitimate: an explicit empty list must override the default toolset instead of being refilled.
func TestLoad_EmptyToolsEnabledStaysEmpty(t *testing.T) {
	p := filepath.Join(t.TempDir(), "c.yaml")
	if err := os.WriteFile(p, []byte("tools:\n  enabled: []\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.Tools.Enabled) != 0 {
		t.Fatalf("tools.enabled = %v, want empty", cfg.Tools.Enabled)
	}
	// absent key keeps the default
	p2 := filepath.Join(t.TempDir(), "d.yaml")
	os.WriteFile(p2, []byte("agent:\n  max_tokens: 100\n"), 0o644)
	cfg2, _ := Load(p2)
	if len(cfg2.Tools.Enabled) == 0 {
		t.Fatal("absent tools key must keep the default toolset")
	}
}
