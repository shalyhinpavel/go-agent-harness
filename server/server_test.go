package server

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/llm"
)

func newTestServer(t *testing.T, opts Options) (*httptest.Server, *llm.MockProvider) {
	t.Helper()
	mock := llm.NewMockProvider(llm.MockTurn{Text: "hi"})
	engine := agent.NewAgentEngine(agent.AgentConfig{Provider: mock}, agent.NewToolRegistry(), agent.NewSessionManager())
	srv := httptest.NewServer(NewHandler(engine, opts))
	t.Cleanup(srv.Close)
	return srv, mock
}

// post sends a chat request and returns the response with its SSE chunks.
func post(t *testing.T, srv *httptest.Server, body any, header http.Header) (*http.Response, []agent.StreamChunk) {
	t.Helper()
	raw, _ := json.Marshal(body)
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/v1/chat/stream", bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	for k, v := range header {
		req.Header[k] = v
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	var chunks []agent.StreamChunk
	sc := bufio.NewScanner(res.Body)
	for sc.Scan() {
		line, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok {
			continue
		}
		var c agent.StreamChunk
		if err := json.Unmarshal([]byte(line), &c); err != nil {
			t.Fatalf("bad SSE payload %q: %v", line, err)
		}
		chunks = append(chunks, c)
	}
	return res, chunks
}

func TestChatStream_FreshSessionPerAnonymousRequest(t *testing.T) {
	srv, _ := newTestServer(t, Options{})
	r1, chunks := post(t, srv, map[string]any{"message": "a"}, nil)
	r2, _ := post(t, srv, map[string]any{"message": "b"}, nil)
	s1, s2 := r1.Header.Get(HeaderSessionID), r2.Header.Get(HeaderSessionID)
	if s1 == "" || s1 == s2 || s1 == "default" {
		t.Fatalf("anonymous requests must not share a session: %q / %q", s1, s2)
	}
	if len(chunks) == 0 || chunks[len(chunks)-1].Type != agent.ChunkTypeDone {
		t.Fatalf("stream must end with done: %+v", chunks)
	}

	r3, _ := post(t, srv, map[string]any{"message": "c", "session_id": s1}, nil)
	if r3.Header.Get(HeaderSessionID) != s1 {
		t.Fatal("an explicit session id must be kept")
	}
}

func TestChatStream_EmptyMessageRefused(t *testing.T) {
	srv, _ := newTestServer(t, Options{})
	res, _ := post(t, srv, map[string]any{"message": "  "}, nil)
	if res.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d", res.StatusCode)
	}
}

func attachment(name, media string, data []byte) map[string]any {
	return map[string]any{"name": name, "media_type": media, "data": base64.StdEncoding.EncodeToString(data)}
}

func TestChatStream_FileAttachmentLandsInTheWorkspace(t *testing.T) {
	root := t.TempDir()
	srv, mock := newTestServer(t, Options{FilesDir: root})
	res, _ := post(t, srv, map[string]any{
		"message": "summarize", "session_id": "s1",
		// A traversal attempt keeps its name, loses its path.
		"attachments": []any{attachment("../../etc/report.csv", "text/csv", []byte("a,b\n1,2\n"))},
	}, nil)
	if res.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", res.StatusCode)
	}
	body, err := os.ReadFile(filepath.Join(root, "anonymous", "s1", "report.csv"))
	if err != nil || string(body) != "a,b\n1,2\n" {
		t.Fatalf("attachment not stored in the session workspace: %q (%v)", body, err)
	}

	reqs := mock.Requests()
	msg := reqs[len(reqs)-1].Messages[0]
	var text string
	for _, p := range msg.Parts {
		text += p.Text
	}
	if !strings.Contains(text, `"report.csv"`) || strings.Contains(text, "YSxi") {
		t.Fatalf("the model must get the stored name, not the bytes: %q", text)
	}
}

func TestChatStream_AttachmentRefusals(t *testing.T) {
	cases := map[string]struct {
		opts Options
		att  map[string]any
	}{
		"no files tool":   {Options{}, attachment("notes.txt", "text/plain", []byte("x"))},
		"bad base64":      {Options{FilesDir: t.TempDir()}, map[string]any{"name": "a.txt", "media_type": "text/plain", "data": "%%%"}},
		"no usable name":  {Options{FilesDir: t.TempDir()}, attachment("../", "text/plain", []byte("x"))},
		"control in name": {Options{FilesDir: t.TempDir()}, attachment("a\x00.txt", "text/plain", []byte("x"))},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			srv, _ := newTestServer(t, tc.opts)
			res, _ := post(t, srv, map[string]any{"message": "m", "attachments": []any{tc.att}}, nil)
			if res.StatusCode != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400", res.StatusCode)
			}
		})
	}

	// Images need no workspace: they ride inline as vision input.
	srv, _ := newTestServer(t, Options{})
	res, _ := post(t, srv, map[string]any{"message": "m", "attachments": []any{attachment("p.png", "image/png", []byte{0x89})}}, nil)
	if res.StatusCode != http.StatusOK {
		t.Fatalf("image without workspace: status %d", res.StatusCode)
	}
}

// identifiedServer takes the caller from the X-Test-User header, so one test
// can speak as two users.
func identifiedServer(t *testing.T, sessions *agent.SessionManager, root string) (*httptest.Server, *llm.MockProvider) {
	t.Helper()
	mock := llm.NewMockProvider(llm.MockTurn{Text: "hi"})
	engine := agent.NewAgentEngine(agent.AgentConfig{Provider: mock}, agent.NewToolRegistry(), sessions)
	srv := httptest.NewServer(NewHandler(engine, Options{
		FilesDir: root,
		Identify: func(r *http.Request) (Identity, error) {
			return Identity{UserID: r.Header.Get("X-Test-User")}, nil
		},
	}))
	t.Cleanup(srv.Close)
	return srv, mock
}

// A session belongs to the user who opened it. Another user's upload is
// refused with 403 before it is written, and the turn never reaches the model.
func TestChatStream_ForeignSessionRefusedBeforeUpload(t *testing.T) {
	root := t.TempDir()
	srv, mock := identifiedServer(t, agent.NewSessionManager(), root)
	alice := http.Header{"X-Test-User": {"alice"}}
	mallory := http.Header{"X-Test-User": {"mallory"}}

	if res, _ := post(t, srv, map[string]any{"message": "hi", "session_id": "s1"}, alice); res.StatusCode != http.StatusOK {
		t.Fatalf("alice's own session: status %d", res.StatusCode)
	}
	res, _ := post(t, srv, map[string]any{
		"message": "summarize", "session_id": "s1",
		"attachments": []any{attachment("secret.pdf", "application/pdf", []byte("%PDF-1.4"))},
	}, mallory)
	if res.StatusCode != http.StatusForbidden {
		t.Fatalf("status = %d, want 403", res.StatusCode)
	}
	if _, err := os.Stat(filepath.Join(root, "alice", "s1", "secret.pdf")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("a refused upload must not be written into alice's workspace, stat error = %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, "mallory", "s1", "secret.pdf")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("a refused upload must not be written into mallory's workspace, stat error = %v", err)
	}
	if n := len(mock.Requests()); n != 1 {
		t.Fatalf("provider calls = %d, want 1: the refused turn must not reach the model", n)
	}
}

// A session with a turn in flight answers 409 the same way: the upload waits
// for the caller to retry rather than landing beside a running turn.
func TestChatStream_BusySessionRefusedBeforeUpload(t *testing.T) {
	root := t.TempDir()
	sessions := agent.NewSessionManager()
	srv, mock := identifiedServer(t, sessions, root)
	release, ok := sessions.BeginTurn("s1")
	if !ok {
		t.Fatal("setup: the session must start free")
	}
	defer release()

	res, _ := post(t, srv, map[string]any{
		"message": "summarize", "session_id": "s1",
		"attachments": []any{attachment("notes.txt", "text/plain", []byte("x"))},
	}, http.Header{"X-Test-User": {"alice"}})
	if res.StatusCode != http.StatusConflict {
		t.Fatalf("status = %d, want 409", res.StatusCode)
	}
	if _, err := os.Stat(filepath.Join(root, "alice", "s1", "notes.txt")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("a refused upload must not be written, stat error = %v", err)
	}
	if n := len(mock.Requests()); n != 0 {
		t.Fatalf("provider calls = %d, want 0", n)
	}
}

// Even without a durable SessionStore, evicting a session from memory must not
// expose its earlier attachments to another user who reuses the same session_id.
func TestChatStream_EvictedSessionWorkspaceStaysScopedToOwner(t *testing.T) {
	root := t.TempDir()
	sessions := agent.NewSessionManager(agent.WithMaxSessions(1))
	mock := llm.NewMockProvider(
		llm.MockTurn{Text: "uploaded"},
		llm.MockTurn{Text: "evicted"},
		llm.MockTurn{Text: "bob turn"},
	)
	engine := agent.NewAgentEngine(agent.AgentConfig{Provider: mock}, agent.NewToolRegistry(), sessions)
	srv := httptest.NewServer(NewHandler(engine, Options{
		FilesDir: root,
		Identify: func(r *http.Request) (Identity, error) {
			return Identity{UserID: r.Header.Get("X-Test-User")}, nil
		},
	}))
	defer srv.Close()

	// Alice uploads a file into s1.
	res, _ := post(t, srv, map[string]any{
		"message": "upload", "session_id": "s1",
		"attachments": []any{attachment("alice.txt", "text/plain", []byte("alice-only"))},
	}, http.Header{"X-Test-User": {"alice"}})
	if res.StatusCode != http.StatusOK {
		t.Fatalf("alice s1: status %d", res.StatusCode)
	}
	// Another session evicts s1 from the in-memory cap of 1.
	if res, _ := post(t, srv, map[string]any{"message": "other", "session_id": "s2"}, http.Header{"X-Test-User": {"alice"}}); res.StatusCode != http.StatusOK {
		t.Fatalf("alice s2: status %d", res.StatusCode)
	}
	// Bob now opens s1: his workspace is under bob/s1, never alice/s1.
	if res, _ := post(t, srv, map[string]any{"message": "hi", "session_id": "s1"}, http.Header{"X-Test-User": {"bob"}}); res.StatusCode != http.StatusOK {
		t.Fatalf("bob s1: status %d", res.StatusCode)
	}
	if _, err := os.Stat(filepath.Join(root, "bob", "s1", "alice.txt")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("bob's workspace must not contain alice's file, err = %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, "alice", "s1", "alice.txt")); err != nil {
		t.Fatalf("alice's file must remain in her own workspace: %v", err)
	}
}

func TestChatStream_BodyLimit(t *testing.T) {
	srv, _ := newTestServer(t, Options{MaxBodyMB: 1})
	big := strings.Repeat("x", 2<<20)
	res, _ := post(t, srv, map[string]any{"message": big}, nil)
	if res.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want 413", res.StatusCode)
	}
}

func TestCORS(t *testing.T) {
	origin := http.Header{"Origin": {"https://evil.example"}}

	srv, _ := newTestServer(t, Options{})
	res, _ := post(t, srv, map[string]any{"message": "m"}, origin)
	if got := res.Header.Get("Access-Control-Allow-Origin"); got != "" {
		t.Fatalf("no origins configured, yet CORS allowed %q", got)
	}

	srv, _ = newTestServer(t, Options{CORSOrigins: []string{"https://app.example"}})
	res, _ = post(t, srv, map[string]any{"message": "m"}, origin)
	if got := res.Header.Get("Access-Control-Allow-Origin"); got != "" {
		t.Fatalf("unlisted origin allowed: %q", got)
	}
	res, _ = post(t, srv, map[string]any{"message": "m"}, http.Header{"Origin": {"https://app.example"}})
	if got := res.Header.Get("Access-Control-Allow-Origin"); got != "https://app.example" {
		t.Fatalf("listed origin refused: %q", got)
	}
	if !strings.Contains(res.Header.Get("Access-Control-Expose-Headers"), HeaderSessionID) {
		t.Fatal("a browser client must be able to read the session id")
	}
}

func TestHealthAndBadJSON(t *testing.T) {
	srv, _ := newTestServer(t, Options{})
	res, err := http.Get(srv.URL + "/health")
	if err != nil || res.StatusCode != http.StatusOK {
		t.Fatalf("health: %v %v", res, err)
	}
	res.Body.Close()

	res, err = http.Post(srv.URL+"/v1/chat/stream", "application/json", strings.NewReader("{not json"))
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if res.StatusCode != http.StatusBadRequest {
		t.Fatalf("malformed JSON: status %d, want 400", res.StatusCode)
	}
}

func TestCORSPreflight(t *testing.T) {
	srv, _ := newTestServer(t, Options{CORSOrigins: []string{"https://app.example"}})
	req, _ := http.NewRequest(http.MethodOptions, srv.URL+"/v1/chat/stream", nil)
	req.Header.Set("Origin", "https://app.example")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if res.StatusCode != http.StatusNoContent || res.Header.Get("Access-Control-Allow-Origin") != "https://app.example" {
		t.Fatalf("preflight: %d %v", res.StatusCode, res.Header)
	}
	if h := res.Header.Get("Access-Control-Allow-Headers"); !strings.Contains(h, "Authorization") {
		t.Fatalf("a bearer-token client needs Authorization allowed, got %q", h)
	}
}

func TestChatStream_SSEHeaders(t *testing.T) {
	srv, _ := newTestServer(t, Options{})
	res, _ := post(t, srv, map[string]any{"message": "hi"}, nil)
	if res.Header.Get("X-Accel-Buffering") != "no" {
		t.Fatalf("a proxy may buffer the stream: %v", res.Header)
	}
}

// A panic after the stream has started must not append a JSON error to it.
func TestMiddleware_PanicAfterHeadersWritesNothingMore(t *testing.T) {
	h := &handler{}
	rec := httptest.NewRecorder()
	h.middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data: {}\n\n"))
		panic("boom")
	})).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/", nil))
	if got := rec.Body.String(); got != "data: {}\n\n" {
		t.Fatalf("body polluted after the panic: %q", got)
	}

	rec = httptest.NewRecorder()
	h.middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		panic("early")
	})).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/", nil))
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("a panic before any output must be a 500, got %d", rec.Code)
	}
}

func TestMiddleware_PanicGoesToTheConfiguredLogger(t *testing.T) {
	var buf strings.Builder
	h := &handler{log: slog.New(slog.NewTextHandler(&buf, nil))}
	h.middleware(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { panic("kaboom") })).
		ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, "/", nil))
	if !strings.Contains(buf.String(), "kaboom") {
		t.Fatalf("panic not logged: %q", buf.String())
	}
}

func TestIdentify_OverridesBodyUserAndFeedsTools(t *testing.T) {
	var gotUser string
	var gotData map[string]any
	registry := agent.NewToolRegistry()
	if err := registry.Register(&agent.Tool{
		Name: "peek", Description: "peek", Parameters: llm.Schema{"type": "object"},
		Execute: func(_ context.Context, tc *agent.ToolContext, _ map[string]any) (*agent.ToolResult, error) {
			gotUser, gotData = tc.UserID, tc.CustomData
			return &agent.ToolResult{Data: "ok"}, nil
		},
	}); err != nil {
		t.Fatal(err)
	}
	mock := llm.NewMockProvider(
		llm.MockTurn{ToolCalls: []llm.ToolCall{{ID: "c1", Name: "peek"}}},
		llm.MockTurn{Text: "done"},
	)
	engine := agent.NewAgentEngine(agent.AgentConfig{Provider: mock}, registry, agent.NewSessionManager())
	srv := httptest.NewServer(NewHandler(engine, Options{
		Identify: func(r *http.Request) (Identity, error) {
			if r.Header.Get("Authorization") != "Bearer ok" {
				return Identity{}, errors.New("no")
			}
			return Identity{UserID: "alice", CustomData: map[string]any{"tenant": "acme"}}, nil
		},
	}))
	defer srv.Close()

	res, _ := post(t, srv, map[string]any{"message": "go", "user_id": "mallory"}, http.Header{"Authorization": {"Bearer nope"}})
	if res.StatusCode != http.StatusUnauthorized {
		t.Fatalf("a failed identification must be 401, got %d", res.StatusCode)
	}

	post(t, srv, map[string]any{"message": "go", "user_id": "mallory"}, http.Header{"Authorization": {"Bearer ok"}})
	if gotUser != "alice" || gotData["tenant"] != "acme" {
		t.Fatalf("tool saw user %q, data %v", gotUser, gotData)
	}
}

func postApprove(t *testing.T, srv *httptest.Server, body any, header http.Header) *http.Response {
	t.Helper()
	raw, _ := json.Marshal(body)
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/v1/chat/approve", bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	for k, v := range header {
		req.Header[k] = v
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	return res
}

func TestApproval_ApproveDenyKeepAliveAndAuth(t *testing.T) {
	var writes int
	reg := agent.NewToolRegistry()
	_ = reg.Register(&agent.Tool{
		Name: "write_note", ReadOnly: false, Parameters: llm.Schema{"type": "object"},
		Execute: func(context.Context, *agent.ToolContext, map[string]any) (*agent.ToolResult, error) {
			writes++
			return &agent.ToolResult{Data: "saved"}, nil
		},
	})
	mock := llm.NewMockProvider(
		llm.MockTurn{ToolCalls: []llm.ToolCall{{ID: "c1", Name: "write_note", Args: map[string]any{"path": "a.md"}}}},
		llm.MockTurn{Text: "Saved."},
		llm.MockTurn{ToolCalls: []llm.ToolCall{{ID: "c2", Name: "write_note", Args: map[string]any{"path": "b.md"}}}},
		llm.MockTurn{Text: "Skipped."},
	)
	engine := agent.NewAgentEngine(agent.AgentConfig{Provider: mock}, reg, agent.NewSessionManager())
	srv := httptest.NewServer(NewHandler(engine, Options{
		RequireApproval:   true,
		KeepAliveInterval: 15 * time.Millisecond,
		Identify: func(r *http.Request) (Identity, error) {
			u := r.Header.Get("X-User")
			if u == "" {
				return Identity{}, errors.New("missing user")
			}
			return Identity{UserID: u}, nil
		},
	}))
	defer srv.Close()

	alice := http.Header{"X-User": {"alice"}}
	mallory := http.Header{"X-User": {"mallory"}}

	// Open SSE stream for turn 1; wait for approval_required and a keepalive frame.
	raw, _ := json.Marshal(map[string]any{"session_id": "s1", "message": "save a"})
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/v1/chat/stream", bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-User", "alice")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()

	sc := bufio.NewScanner(res.Body)
	var sawKeepAlive, sawApproval, sawDone bool
	for sc.Scan() {
		line := sc.Text()
		if line == ": keepalive" {
			sawKeepAlive = true
			continue
		}
		payload, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		var chunk agent.StreamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			t.Fatal(err)
		}
		if chunk.Type == agent.ChunkTypeApprovalRequired {
			sawApproval = true
			// Wait briefly so at least one SSE keepalive comment is sent while paused.
			time.Sleep(35 * time.Millisecond)

			// Auth, validation and cross-user isolation on /v1/chat/approve.
			if r := postApprove(t, srv, map[string]any{"session_id": "s1", "tool_call_id": "c1", "approved": true}, nil); r.StatusCode != http.StatusUnauthorized {
				t.Fatalf("unauthenticated approve status = %d, want 401", r.StatusCode)
			}
			if r := postApprove(t, srv, map[string]any{"session_id": "", "tool_call_id": "c1"}, alice); r.StatusCode != http.StatusBadRequest {
				t.Fatalf("missing session_id status = %d, want 400", r.StatusCode)
			}
			if r := postApprove(t, srv, map[string]any{"session_id": "s1", "tool_call_id": "c1", "approved": true}, mallory); r.StatusCode != http.StatusNotFound {
				t.Fatalf("foreign user approve status = %d, want 404", r.StatusCode)
			}
			if r := postApprove(t, srv, map[string]any{"session_id": "s1", "tool_call_id": "c1", "approved": true}, alice); r.StatusCode != http.StatusOK {
				t.Fatalf("owner approve status = %d, want 200", r.StatusCode)
			}
			// Second submit on the same call id is 404.
			if r := postApprove(t, srv, map[string]any{"session_id": "s1", "tool_call_id": "c1", "approved": true}, alice); r.StatusCode != http.StatusNotFound {
				t.Fatalf("duplicate approve status = %d, want 404", r.StatusCode)
			}
		}
		if chunk.Type == agent.ChunkTypeDone {
			sawDone = true
		}
	}
	if !sawApproval || !sawKeepAlive || !sawDone || writes != 1 {
		t.Fatalf("turn 1: approval=%v keepalive=%v done=%v writes=%d", sawApproval, sawKeepAlive, sawDone, writes)
	}

	// Turn 2: deny with a reason; tool does not execute and model sees the reason.
	raw2, _ := json.Marshal(map[string]any{"session_id": "s1", "message": "save b"})
	req2, _ := http.NewRequest(http.MethodPost, srv.URL+"/v1/chat/stream", bytes.NewReader(raw2))
	req2.Header.Set("Content-Type", "application/json")
	req2.Header.Set("X-User", "alice")
	res2, err := http.DefaultClient.Do(req2)
	if err != nil {
		t.Fatal(err)
	}
	defer res2.Body.Close()
	sc2 := bufio.NewScanner(res2.Body)
	for sc2.Scan() {
		payload, ok := strings.CutPrefix(sc2.Text(), "data: ")
		if !ok {
			continue
		}
		var chunk agent.StreamChunk
		_ = json.Unmarshal([]byte(payload), &chunk)
		if chunk.Type == agent.ChunkTypeApprovalRequired {
			r := postApprove(t, srv, map[string]any{
				"session_id": "s1", "tool_call_id": chunk.ToolCallID,
				"approved": false, "reason": "wrong file",
			}, alice)
			if r.StatusCode != http.StatusOK {
				t.Fatalf("deny status = %d", r.StatusCode)
			}
		}
	}
	if writes != 1 {
		t.Fatalf("denied tool ran: writes = %d", writes)
	}
	reqs := mock.Requests()
	lastMsg := reqs[len(reqs)-1].Messages
	if !strings.Contains(lastMsg[len(lastMsg)-1].Parts[0].ToolResult.Content.(map[string]any)["error"].(string), "wrong file") {
		t.Fatalf("model did not see denial reason: %+v", lastMsg[len(lastMsg)-1])
	}
}

func TestApproval_TimeoutDeniesAndReleasesSession(t *testing.T) {
	reg := agent.NewToolRegistry()
	_ = reg.Register(&agent.Tool{
		Name: "mutate", ReadOnly: false, Parameters: llm.Schema{"type": "object"},
		Execute: func(context.Context, *agent.ToolContext, map[string]any) (*agent.ToolResult, error) {
			t.Error("timed-out tool must not run")
			return &agent.ToolResult{}, nil
		},
	})
	mock := llm.NewMockProvider(
		llm.MockTurn{ToolCalls: []llm.ToolCall{{ID: "c1", Name: "mutate"}}},
		llm.MockTurn{Text: "Timed out."},
		llm.MockTurn{Text: "Next turn works."},
	)
	engine := agent.NewAgentEngine(agent.AgentConfig{Provider: mock}, reg, agent.NewSessionManager())
	srv := httptest.NewServer(NewHandler(engine, Options{
		RequireApproval: true,
		ApprovalTimeout: 30 * time.Millisecond,
	}))
	defer srv.Close()

	_, chunks := post(t, srv, map[string]any{"session_id": "s1", "message": "go"}, nil)
	if len(chunks) == 0 || chunks[len(chunks)-1].Type != agent.ChunkTypeDone {
		t.Fatalf("expected turn to complete after approval timeout, got %+v", chunks)
	}
	// Session must not stay locked after the timeout.
	res, _ := post(t, srv, map[string]any{"session_id": "s1", "message": "again"}, nil)
	if res.StatusCode != http.StatusOK {
		t.Fatalf("follow-up turn status = %d, want 200", res.StatusCode)
	}
}

func TestApproval_ClientDisconnectClearsPendingSlot(t *testing.T) {
	reg := agent.NewToolRegistry()
	_ = reg.Register(&agent.Tool{
		Name: "mutate", ReadOnly: false, Parameters: llm.Schema{"type": "object"},
		Execute: func(context.Context, *agent.ToolContext, map[string]any) (*agent.ToolResult, error) {
			return &agent.ToolResult{}, nil
		},
	})
	mock := llm.NewMockProvider(
		llm.MockTurn{ToolCalls: []llm.ToolCall{{ID: "c1", Name: "mutate"}}},
		llm.MockTurn{Text: "never"},
	)
	engine := agent.NewAgentEngine(agent.AgentConfig{Provider: mock}, reg, agent.NewSessionManager())
	srv := httptest.NewServer(NewHandler(engine, Options{RequireApproval: true}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	raw, _ := json.Marshal(map[string]any{"session_id": "s1", "message": "go"})
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, srv.URL+"/v1/chat/stream", bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	sc := bufio.NewScanner(res.Body)
	for sc.Scan() {
		if strings.Contains(sc.Text(), `"approval_required"`) {
			cancel()
			break
		}
	}
	res.Body.Close()

	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		r := postApprove(t, srv, map[string]any{"session_id": "s1", "tool_call_id": "c1", "approved": true}, nil)
		if r.StatusCode == http.StatusNotFound {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("pending approval slot remained after client disconnect")
}
