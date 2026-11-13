package memory

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func newFileStore(t *testing.T) *FileStore {
	t.Helper()
	s, err := NewFileStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestFileStore_FactRoundtripAsMarkdown(t *testing.T) {
	s := newFileStore(t)

	if err := s.InsertFact(context.Background(), &UserFact{
		ID: "f1", UserID: "u1", Category: "work", Tags: []string{"go", "backend"},
		Fact: "Works with Go in Berlin", Confidence: 0.9,
	}); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(filepath.Join(s.dir, "u1", "fact-f1.md"))
	if err != nil {
		t.Fatal(err)
	}
	note := string(raw)
	for _, want := range []string{"id: f1", "category: work", "- go", "confidence: 0.9", "Works with Go in Berlin"} {
		if !strings.Contains(note, want) {
			t.Fatalf("note missing %q:\n%s", want, note)
		}
	}

	// Search sees it through the same file.
	facts, err := s.SearchFacts(context.Background(), "u1", "programming language go", nil, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(facts) != 1 || facts[0].Fact != "Works with Go in Berlin" || facts[0].Score <= 0 {
		t.Fatalf("unexpected search results: %+v", facts)
	}
}

func TestFileStore_HumanEditsAreVisible(t *testing.T) {
	s := newFileStore(t)

	// Write a note by hand — no API involved.
	note := "---\nid: hand\ncategory: preference\nconfidence: 0.5\n---\n\nPrefers dark mode\n"
	if err := writeFile(filepath.Join(s.dir, "u2", "fact-hand.md"), note); err != nil {
		t.Fatal(err)
	}

	facts, err := s.SearchFacts(context.Background(), "u2", "dark mode", nil, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(facts) != 1 || facts[0].Fact != "Prefers dark mode" {
		t.Fatalf("hand-written note not found: %+v", facts)
	}
}

func TestFileStore_LexicalRanking(t *testing.T) {
	s := newFileStore(t)
	for _, f := range []*UserFact{
		{ID: "a", UserID: "u1", Category: "work", Fact: "Writes python scripts for data pipelines"},
		{ID: "b", UserID: "u1", Category: "work", Fact: "Cycles to work on fridays"},
		{ID: "c", UserID: "u1", Category: "work", Fact: "Prefers python over rust for prototypes"},
	} {
		if err := s.InsertFact(context.Background(), f); err != nil {
			t.Fatal(err)
		}
	}

	facts, err := s.SearchFacts(context.Background(), "u1", "python", nil, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(facts) != 2 {
		t.Fatalf("want 2 python facts, got %+v", facts)
	}
	for _, f := range facts {
		if !strings.Contains(f.Fact, "python") {
			t.Fatalf("non-python fact ranked: %+v", f)
		}
	}

	// A query nothing matches returns nothing.
	none, err := s.SearchFacts(context.Background(), "u1", "kubernetes", nil, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(none) != 0 {
		t.Fatalf("expected no matches, got %+v", none)
	}
}

func TestFileStore_VectorPath(t *testing.T) {
	s := newFileStore(t)
	if err := s.InsertFact(context.Background(), &UserFact{
		ID: "v1", UserID: "u1", Category: "work",
		Fact: "Works with Go", Embedding: []float32{1, 0, 0},
	}); err != nil {
		t.Fatal(err)
	}

	// Aligned vector scores ~1, orthogonal ~0 (and is dropped).
	facts, err := s.SearchFacts(context.Background(), "u1", "", []float32{0.9, 0.1, 0}, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(facts) != 1 || facts[0].Score < 0.8 {
		t.Fatalf("expected high cosine score, got %+v", facts)
	}

	none, err := s.SearchFacts(context.Background(), "u1", "", []float32{0, 1, 0}, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(none) != 0 {
		t.Fatalf("orthogonal vector must not match, got %+v", none)
	}
}

func TestFileStore_DeleteAndDossier(t *testing.T) {
	s := newFileStore(t)
	if err := s.InsertFact(context.Background(), &UserFact{ID: "f1", UserID: "u1", Fact: "temp"}); err != nil {
		t.Fatal(err)
	}
	if err := s.DeleteFact(context.Background(), "u1", "f1"); err != nil {
		t.Fatal(err)
	}
	if err := s.DeleteFact(context.Background(), "u1", "f1"); err == nil {
		t.Fatal("deleting a missing fact must fail")
	}

	if err := s.SaveDossier(context.Background(), &UserDossier{UserID: "u1", Content: "Go dev"}); err != nil {
		t.Fatal(err)
	}
	d, err := s.GetDossier(context.Background(), "u1")
	if err != nil || d.Content != "Go dev" {
		t.Fatalf("dossier roundtrip: %+v err=%v", d, err)
	}

	empty, err := s.GetDossier(context.Background(), "nobody")
	if err != nil || empty.Content != "" {
		t.Fatalf("missing dossier must be empty, got %+v err=%v", empty, err)
	}
}

func TestFileStore_HostileIDStaysInsideDir(t *testing.T) {
	s := newFileStore(t)
	if err := s.InsertFact(context.Background(), &UserFact{ID: "../../pwn", UserID: "../../etc", Fact: "evil"}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.GetDossier(context.Background(), "../../etc"); err != nil {
		t.Fatal(err)
	}

	entries, err := os.ReadDir(s.dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || strings.Contains(entries[0].Name(), "/") {
		t.Fatalf("hostile id escaped the store dir: %+v", entries)
	}
}

func TestTokenize_AnyScript(t *testing.T) {
	got := tokenize("Grüße, Αβγδεζ! 東京 42 a")
	for _, want := range []string{"grüße", "αβγδεζ", "東京", "42"} {
		if !got[want] {
			t.Errorf("token %q missing from %v", want, got)
		}
	}
	if got["a"] {
		t.Error("one-letter tokens must be dropped")
	}
}

func TestFileStore_CRLFNoteIsRead(t *testing.T) {
	s := newFileStore(t)
	note := "---\r\nid: win\r\ncategory: preference\r\nconfidence: 0.5\r\n---\r\n\r\nPrefers dark mode\r\n"
	if err := writeFile(filepath.Join(s.dir, "u3", "fact-win.md"), note); err != nil {
		t.Fatal(err)
	}
	facts, err := s.SearchFacts(context.Background(), "u3", "dark mode", nil, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(facts) != 1 || facts[0].Fact != "Prefers dark mode" {
		t.Fatalf("CRLF note not read: %+v", facts)
	}
}

// Concurrent writers of one note must not trip over a shared temp file, and
// no temp file may be left behind.
func TestFileStore_ConcurrentWritesOfOneNote(t *testing.T) {
	s := newFileStore(t)
	var wg sync.WaitGroup
	errs := make(chan error, 32)
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs <- s.InsertFact(context.Background(), &UserFact{ID: "same", UserID: "u", Fact: "one fact", Confidence: 1})
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("concurrent write failed: %v", err)
		}
	}
	entries, _ := os.ReadDir(filepath.Join(s.dir, "u"))
	if len(entries) != 1 {
		t.Fatalf("want one note and no leftovers, got %v", entries)
	}
}
