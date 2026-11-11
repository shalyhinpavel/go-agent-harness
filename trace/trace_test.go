package trace_test

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shalyhinpavel/go-agent-harness/llm"
	"github.com/shalyhinpavel/go-agent-harness/trace"
)

func TestDirSink_PerSessionFiles(t *testing.T) {
	dir := t.TempDir()
	sink, err := trace.NewDirSink(dir)
	if err != nil {
		t.Fatal(err)
	}

	sink.Record(trace.Event{SessionID: "s1", Type: trace.EventTurnStart, Text: "hello"})
	sink.Record(trace.Event{SessionID: "s2", Type: trace.EventTurnStart, Text: "other"})
	sink.Record(trace.Event{
		SessionID: "s1", Type: trace.EventLLMCall,
		Usage: &llm.Usage{InputTokens: 10, OutputTokens: 2}, LatencyMS: 42,
	})
	sink.Close()

	s1, err := os.ReadFile(filepath.Join(dir, "s1.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if lines := strings.Count(string(s1), "\n"); lines != 2 {
		t.Fatalf("s1 has %d lines, want 2", lines)
	}
	if !strings.Contains(string(s1), `"latency_ms":42`) || !strings.Contains(string(s1), `"input_tokens":10`) {
		t.Fatalf("llm_call event missing usage/latency: %s", s1)
	}

	if _, err := os.Stat(filepath.Join(dir, "s2.jsonl")); err != nil {
		t.Fatalf("s2 file: %v", err)
	}
}

func TestDirSink_HostileSessionIDStaysInsideDir(t *testing.T) {
	dir := t.TempDir()
	sink, err := trace.NewDirSink(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer sink.Close()

	sink.Record(trace.Event{SessionID: "../../etc/evil", Type: trace.EventTurnStart})

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected exactly one trace file, got %d", len(entries))
	}
	name := entries[0].Name()
	if strings.Contains(name, "/") || name == "evil.jsonl" {
		t.Fatalf("session id escaped sanitization: %q", name)
	}
}

func TestDirSink_SimilarSessionIDsGetSeparateFiles(t *testing.T) {
	dir := t.TempDir()
	sink, err := trace.NewDirSink(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer sink.Close()
	for _, id := range []string{"team/1", "team:1", "team.1", "team-1"} {
		sink.Record(trace.Event{SessionID: id, Type: trace.EventTurnStart})
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 4 {
		t.Fatalf("4 sessions must write 4 files, got %d", len(entries))
	}
}

func TestDirSink_OpenFilesAreBounded(t *testing.T) {
	dir := t.TempDir()
	sink, err := trace.NewDirSink(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer sink.Close()

	// Far more sessions than the pool holds open; every one must still land
	// on disk, and an evicted session appending again must not lose lines.
	const sessions = 300
	for i := 0; i < sessions; i++ {
		sink.Record(trace.Event{SessionID: fmt.Sprintf("s%d", i), Type: trace.EventTurnStart})
	}
	sink.Record(trace.Event{SessionID: "s0", Type: trace.EventTurnEnd})

	if n := sink.OpenFiles(); n > 64 {
		t.Fatalf("%d files held open, the pool must stay bounded", n)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "s0.jsonl"))
	if err != nil || strings.Count(string(raw), "\n") != 2 {
		t.Fatalf("s0 must keep both lines across an eviction: %q %v", raw, err)
	}
	if entries, _ := os.ReadDir(dir); len(entries) != sessions {
		t.Fatalf("%d files on disk, want %d", len(entries), sessions)
	}
}
