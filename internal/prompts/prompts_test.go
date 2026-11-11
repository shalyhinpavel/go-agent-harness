package prompts

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoad_EmbeddedDefault(t *testing.T) {
	l := New("")
	text, err := l.Load("system")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(text, "helpful assistant") {
		t.Fatalf("embedded system prompt wrong: %q", text)
	}
}

func TestLoad_ExtensionNormalization(t *testing.T) {
	l := New("")
	if _, err := l.Load("system.md"); err != nil {
		t.Fatalf("name with extension must resolve: %v", err)
	}
}

func TestLoad_DiskOverrideWins(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "system.md"), []byte("CUSTOM PROMPT"), 0o644); err != nil {
		t.Fatal(err)
	}

	text, err := New(dir).Load("system")
	if err != nil {
		t.Fatal(err)
	}
	if text != "CUSTOM PROMPT" {
		t.Fatalf("disk override ignored: %q", text)
	}
}

func TestLoad_UnknownNameFails(t *testing.T) {
	if _, err := New("").Load("no-such-prompt"); err == nil {
		t.Fatal("unknown prompt must fail")
	}
}

func TestLoad_ToolFreeDefault(t *testing.T) {
	// A run with no tools uses this prompt: it must exist in the embedded
	// defaults and must not promise tools.
	text, err := New("").Load("system_notools")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(text, "no tools in this run") {
		t.Fatalf("tool-free prompt does not say it has no tools: %q", text)
	}
	if strings.Contains(text, "with access to tools") || strings.Contains(text, "After a tool returns") {
		t.Fatalf("tool-free prompt still promises tools: %q", text)
	}

	// A disk override under the same name wins, as for every prompt.
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "system_notools.md"), []byte("CUSTOM TOOL-FREE"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got, err := New(dir).Load("system_notools"); err != nil || got != "CUSTOM TOOL-FREE" {
		t.Fatalf("override = %q, err %v", got, err)
	}
}
