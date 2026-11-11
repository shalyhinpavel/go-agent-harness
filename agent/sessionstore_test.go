package agent

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/shalyhinpavel/go-agent-harness/llm"
)

func newFileStoreForTest(t *testing.T) (*FileSessionStore, string) {
	t.Helper()
	dir := t.TempDir()
	s, err := NewFileSessionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	return s, dir
}

func TestFileSessionStore_RoundTrip(t *testing.T) {
	s, _ := newFileStoreForTest(t)
	ctx := context.Background()
	if got, err := s.Load(ctx, "none"); got != nil || err != nil {
		t.Fatalf("a missing session is (nil, nil), got %v %v", got, err)
	}
	if err := s.Create(ctx, "s1", "alice", time.Now()); err != nil {
		t.Fatal(err)
	}
	msgs := []llm.Message{
		*llm.UserText("hi"),
		{Role: llm.RoleAssistant, Parts: []llm.Part{llm.ToolCallPart(llm.ToolCall{ID: "c1", Name: "t", Args: map[string]any{"a": 1.0}})}},
		{Role: llm.RoleTool, Parts: []llm.Part{llm.ToolResultPart(llm.ToolResultData{CallID: "c1", Content: "ok"})}},
	}
	if err := s.Append(ctx, "s1", msgs...); err != nil {
		t.Fatal(err)
	}
	got, err := s.Load(ctx, "s1")
	if err != nil || got == nil {
		t.Fatalf("load: %v %v", got, err)
	}
	if got.UserID != "alice" || len(got.History) != 3 || got.History[1].Parts[0].ToolCall.Name != "t" {
		t.Fatalf("round trip lost data: %+v", got)
	}
	if err := s.Delete(ctx, "s1"); err != nil {
		t.Fatal(err)
	}
	if got, _ := s.Load(ctx, "s1"); got != nil {
		t.Fatal("deleted session still loads")
	}
	if err := s.Delete(ctx, "s1"); err != nil {
		t.Fatalf("deleting a missing session is not an error: %v", err)
	}
}

func TestFileSessionStore_TornLastLineIsDropped_DamageElsewhereIsAnError(t *testing.T) {
	s, dir := newFileStoreForTest(t)
	ctx := context.Background()
	_ = s.Create(ctx, "s1", "u", time.Now())
	_ = s.Append(ctx, "s1", *llm.UserText("one"))
	path := s.path("s1")

	// A crash mid-write leaves a line without its end.
	f, _ := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
	_, _ = f.WriteString(`{"type":"message","message":{"role":"us`)
	f.Close()
	got, err := s.Load(ctx, "s1")
	if err != nil || len(got.History) != 1 {
		t.Fatalf("a torn tail must be ignored: %v %+v", err, got)
	}

	// Damage in the middle must not shorten the conversation silently.
	raw, _ := os.ReadFile(path)
	lines := strings.SplitN(string(raw), "\n", 3)
	bad := lines[0] + "\n{not json}\n" + lines[1] + "\n"
	_ = os.WriteFile(filepath.Join(dir, filepath.Base(path)), []byte(bad), 0o600)
	if _, err := s.Load(ctx, "s1"); err == nil {
		t.Fatal("a damaged file must be an error")
	}
}

func TestFileSessionStore_HostileIDsStayInsideAndStayApart(t *testing.T) {
	s, dir := newFileStoreForTest(t)
	ctx := context.Background()
	for _, id := range []string{"../../escape", "a/b", "a_b", "A", "a"} {
		if err := s.Create(ctx, id, "u:"+id, time.Now()); err != nil {
			t.Fatalf("%q: %v", id, err)
		}
	}
	for _, id := range []string{"../../escape", "a/b", "a_b", "A", "a"} {
		got, err := s.Load(ctx, id)
		if err != nil || got == nil || got.UserID != "u:"+id {
			t.Fatalf("%q: ids collided or were lost: %v %+v", id, err, got)
		}
	}
	entries, _ := os.ReadDir(dir)
	if len(entries) != 5 {
		t.Fatalf("want 5 files inside the directory, got %d", len(entries))
	}
}

func TestSessionManager_StoreSurvivesRestartAndKeepsTheOwner(t *testing.T) {
	store, _ := newFileStoreForTest(t)
	ctx := context.Background()

	sm1 := NewSessionManager(WithSessionStore(store))
	sess, err := sm1.GetOrCreateSession(ctx, "s1", "alice")
	if err != nil {
		t.Fatal(err)
	}
	_ = sm1.AppendMessage(ctx, sess.ID, llm.UserText("remember me"))
	_ = sm1.AppendMessage(ctx, sess.ID, llm.AssistantText("noted"))

	// A new process: nothing in memory.
	sm2 := NewSessionManager(WithSessionStore(store))
	back, err := sm2.GetOrCreateSession(ctx, "s1", "mallory")
	if err != nil {
		t.Fatal(err)
	}
	if back.UserID != "alice" {
		t.Fatalf("the stored owner must win over the caller's claim, got %q", back.UserID)
	}
	if h := sm2.GetHistory("s1"); len(h) != 2 || h[0].Parts[0].Text != "remember me" {
		t.Fatalf("history not restored: %+v", h)
	}
}

func TestSessionManager_EvictedSessionComesBackFromTheStore(t *testing.T) {
	store, _ := newFileStoreForTest(t)
	ctx := context.Background()
	sm := NewSessionManager(WithSessionStore(store), WithMaxSessions(1))
	_, _ = sm.GetOrCreateSession(ctx, "a", "u")
	_ = sm.AppendMessage(ctx, "a", llm.UserText("first"))
	_, _ = sm.GetOrCreateSession(ctx, "b", "u") // evicts "a" from memory
	if h := sm.GetHistory("a"); h != nil {
		t.Fatal("a must be out of memory")
	}
	back, _ := sm.GetOrCreateSession(ctx, "a", "u")
	if len(back.History) != 1 {
		t.Fatalf("the evicted session lost its history: %+v", back.History)
	}
}

func TestSessionManager_DanglingToolCallIsDroppedOnReload(t *testing.T) {
	store, _ := newFileStoreForTest(t)
	ctx := context.Background()
	_ = store.Create(ctx, "s", "u", time.Now())
	_ = store.Append(ctx, "s", *llm.UserText("go"), llm.Message{
		Role:  llm.RoleAssistant,
		Parts: []llm.Part{llm.ToolCallPart(llm.ToolCall{ID: "c1", Name: "t"})},
	}) // crash before the tool result was written
	sm := NewSessionManager(WithSessionStore(store))
	sess, _ := sm.GetOrCreateSession(ctx, "s", "u")
	if len(sess.History) != 1 {
		t.Fatalf("a tool call without a result must be dropped, got %d messages", len(sess.History))
	}
}

type failingStore struct{ SessionStore }

func (failingStore) Append(context.Context, string, ...llm.Message) error {
	return errors.New("disk full")
}

func TestEngine_StoreWriteFailureEndsTheTurnWithAnError(t *testing.T) {
	real, _ := newFileStoreForTest(t)
	sm := NewSessionManager(WithSessionStore(failingStore{real}))
	mock := llm.NewMockProvider(llm.MockTurn{Text: "never"})
	engine := NewAgentEngine(AgentConfig{Provider: mock}, NewToolRegistry(), sm)
	_, err := engine.Chat(context.Background(), ChatRequest{UserID: "u", SessionID: "s", Message: "hi"})
	if err == nil || !strings.Contains(err.Error(), "disk full") {
		t.Fatalf("an unsaved conversation must fail the turn, got %v", err)
	}
	if n := len(mock.Requests()); n != 0 {
		t.Fatalf("the model was called %d times with nothing saved", n)
	}
}

// A conversation continues across a restart through the engine itself.
func TestEngine_ConversationContinuesAfterRestart(t *testing.T) {
	store, _ := newFileStoreForTest(t)
	ctx := context.Background()

	e1 := NewAgentEngine(AgentConfig{Provider: llm.NewMockProvider(llm.MockTurn{Text: "first answer"})},
		NewToolRegistry(), NewSessionManager(WithSessionStore(store)))
	if _, err := e1.Chat(ctx, ChatRequest{UserID: "alice", SessionID: "s", Message: "question one"}); err != nil {
		t.Fatal(err)
	}

	mock2 := llm.NewMockProvider(llm.MockTurn{Text: "second answer"})
	e2 := NewAgentEngine(AgentConfig{Provider: mock2}, NewToolRegistry(), NewSessionManager(WithSessionStore(store)))
	if _, err := e2.Chat(ctx, ChatRequest{UserID: "alice", SessionID: "s", Message: "question two"}); err != nil {
		t.Fatal(err)
	}
	reqs := mock2.Requests()
	if len(reqs) != 1 || len(reqs[0].Messages) != 3 {
		t.Fatalf("the model should see the earlier exchange, got %+v", reqs)
	}
	// The owner check survives the restart too.
	if _, err := e2.Chat(ctx, ChatRequest{UserID: "mallory", SessionID: "s", Message: "mine now"}); err == nil {
		t.Fatal("another user must not enter alice's stored session")
	}
}
