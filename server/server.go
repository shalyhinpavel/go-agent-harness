// Package server is the HTTP face of the harness: one SSE endpoint over
// the engine. It has no authentication — the defaults keep it on loopback
// with no CORS; whoever exposes it puts an authenticating proxy in front.
package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"path"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/google/uuid"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/tools"
)

// HeaderSessionID carries the session a turn ran in. A request without a
// session_id gets a fresh one — never a shared default — and the client
// continues the conversation by sending it back.
const HeaderSessionID = "X-Session-ID"

const (
	// DefaultApprovalTimeout is how long a non-read-only tool call waits for
	// POST /v1/chat/approve before being denied as timed out.
	DefaultApprovalTimeout = 5 * time.Minute
	// DefaultKeepAliveInterval is how often an idle SSE stream sends a comment
	// frame so proxies do not close the connection while waiting.
	DefaultKeepAliveInterval = 15 * time.Second
)

// Options configures the HTTP layer.
type Options struct {
	// CORSOrigins lists the browser origins allowed to call the API;
	// empty = no CORS headers at all.
	CORSOrigins []string
	// MaxBodyMB caps a request body (attachments ride inside it base64).
	MaxBodyMB int
	// FilesDir is the file tools' workspace root. Non-image attachments
	// are stored there for the model to open; "" (file tools off) refuses
	// them — the model would be told about a file it cannot read.
	FilesDir string
	// Logger receives request diagnostics (nil = slog.Default()).
	Logger *slog.Logger
	// Identify derives the caller from the request — a verified token, a
	// session cookie. Its UserID replaces the one in the body (which any
	// client could forge), its CustomData reaches the tools, and an error
	// answers 401. nil = the body's user_id is trusted: fine on loopback,
	// wrong anywhere users must not see each other's sessions.
	Identify func(r *http.Request) (Identity, error)
	// RequireApproval pauses non-read-only tool calls (emitting an
	// "approval_required" SSE chunk) until confirmed via POST /v1/chat/approve.
	RequireApproval bool
	// ApprovalTimeout bounds how long a tool call waits for approval before
	// being denied as timed out (<=0 = DefaultApprovalTimeout).
	ApprovalTimeout time.Duration
	// KeepAliveInterval controls how often an idle SSE stream writes a
	// ": keepalive\n\n" frame (<=0 = DefaultKeepAliveInterval).
	KeepAliveInterval time.Duration
}

// Identity is who a request comes from, as established by Options.Identify.
type Identity struct {
	UserID     string
	CustomData map[string]any // becomes ToolContext.CustomData
}

// NewHandler builds the HTTP handler: GET /health, POST /v1/chat/stream and
// POST /v1/chat/approve. It is a plain http.Handler — mount it in any router,
// or wrap it with your own authentication middleware.
func NewHandler(engine *agent.AgentEngine, opts Options) http.Handler {
	approvalTimeout := opts.ApprovalTimeout
	if approvalTimeout <= 0 {
		approvalTimeout = DefaultApprovalTimeout
	}
	keepAlive := opts.KeepAliveInterval
	if keepAlive <= 0 {
		keepAlive = DefaultKeepAliveInterval
	}
	h := &handler{
		engine:          engine,
		filesDir:        opts.FilesDir,
		maxBody:         int64(opts.MaxBodyMB) << 20,
		log:             opts.Logger,
		identify:        opts.Identify,
		requireApproval: opts.RequireApproval,
		approvalTimeout: approvalTimeout,
		keepAlive:       keepAlive,
		approvals:       newApprovalBroker(),
	}
	if h.maxBody <= 0 {
		h.maxBody = 20 << 20
	}
	h.cors = map[string]bool{}
	for _, o := range opts.CORSOrigins {
		h.cors[o] = true
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("POST /v1/chat/stream", h.chatStream)
	mux.HandleFunc("POST /v1/chat/approve", h.approveChat)
	return h.middleware(mux)
}

type handler struct {
	engine          *agent.AgentEngine
	filesDir        string
	maxBody         int64
	identify        func(r *http.Request) (Identity, error)
	requireApproval bool
	approvalTimeout time.Duration
	keepAlive       time.Duration
	approvals       *approvalBroker
	cors            map[string]bool // allowed browser origins; empty = no CORS headers
	log             *slog.Logger
}

func (h *handler) logger() *slog.Logger {
	if h.log != nil {
		return h.log
	}
	return slog.Default()
}

// middleware recovers from panics and applies CORS. CORS headers appear only
// for origins on the explicit list, so an empty list means none at all.
func (h *handler) middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sw := &statusWriter{ResponseWriter: w}
		w = sw
		defer func() {
			if rec := recover(); rec != nil {
				h.logger().Error("recovered from a panic in a request handler", "panic", rec)
				// Once the headers are out (an SSE stream in flight) a JSON
				// body would land inside the stream; the cut-off is all we can do.
				if !sw.wrote {
					writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "internal error"})
				}
			}
		}()
		if origin := r.Header.Get("Origin"); origin != "" && h.cors[origin] {
			w.Header().Add("Vary", "Origin")
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Access-Control-Expose-Headers", HeaderSessionID)
			if r.Method == http.MethodOptions {
				w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
				w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
				w.WriteHeader(http.StatusNoContent)
				return
			}
		}
		next.ServeHTTP(w, r)
	})
}

// statusWriter remembers whether the response has started. Unwrap keeps
// http.ResponseController (flushing) working through it.
type statusWriter struct {
	http.ResponseWriter
	wrote bool
}

func (w *statusWriter) WriteHeader(code int) {
	w.wrote = true
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusWriter) Write(b []byte) (int, error) {
	w.wrote = true
	return w.ResponseWriter.Write(b)
}

func (w *statusWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// clientError is a request the client must fix (400), as opposed to a
// server-side failure (500).
type clientError struct{ msg string }

func (e *clientError) Error() string { return e.msg }

func badRequest(format string, args ...any) error {
	return &clientError{msg: fmt.Sprintf(format, args...)}
}

func (h *handler) chatStream(w http.ResponseWriter, r *http.Request) {
	var who Identity
	if h.identify != nil {
		var err error
		if who, err = h.identify(r); err != nil {
			writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "unauthorized"})
			return
		}
	}
	var req agent.ChatRequest
	r.Body = http.MaxBytesReader(w, r.Body, h.maxBody)
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			writeJSON(w, http.StatusRequestEntityTooLarge, map[string]string{"error": fmt.Sprintf("request body exceeds %d bytes", tooBig.Limit)})
			return
		}
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON body: " + err.Error()})
		return
	}
	if strings.TrimSpace(req.Message) == "" && len(req.Attachments) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "message is required"})
		return
	}
	if h.identify != nil {
		req.UserID, req.CustomData = who.UserID, who.CustomData
	}
	if req.UserID == "" {
		req.UserID = "anonymous"
	}
	if req.SessionID == "" {
		req.SessionID = uuid.NewString()
	}
	// Refused before anything is written: an upload into another user's
	// session, or into one mid-turn, would be left behind in its workspace.
	if err := h.engine.CheckSession(r.Context(), req.UserID, req.SessionID); err != nil {
		switch {
		case errors.Is(err, agent.ErrSessionOwner):
			writeJSON(w, http.StatusForbidden, map[string]string{"error": "session belongs to another user"})
		case errors.Is(err, agent.ErrSessionBusy):
			writeJSON(w, http.StatusConflict, map[string]string{"error": "session already has a turn in progress"})
		default:
			h.logger().Error("failed to check session", "err", err)
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to check session"})
		}
		return
	}
	if err := h.storeAttachments(&req); err != nil {
		var ce *clientError
		if errors.As(err, &ce) {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}
		h.logger().Error("failed to store attachment", "err", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to store attachment"})
		return
	}

	if h.requireApproval && req.Approver == nil {
		userID := req.UserID
		req.Approver = func(ctx context.Context, ar agent.ApprovalRequest) (agent.ApprovalDecision, error) {
			slot := h.approvals.ensure(ar.SessionID, userID, ar.ToolCallID)
			defer h.approvals.remove(ar.SessionID, ar.ToolCallID)

			timer := time.NewTimer(h.approvalTimeout)
			defer timer.Stop()

			select {
			case dec := <-slot.ch:
				return dec, nil
			case <-timer.C:
				return agent.ApprovalDecision{Approved: false, Reason: "approval timed out"}, nil
			case <-ctx.Done():
				return agent.ApprovalDecision{}, ctx.Err()
			}
		}
	}
	defer h.approvals.clearSession(req.SessionID)

	stream, err := h.engine.StreamChat(r.Context(), req)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}

	w.Header().Set(HeaderSessionID, req.SessionID)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no") // nginx must not buffer the stream
	w.WriteHeader(http.StatusOK)

	// The chunk itself is the SSE payload — its JSON tags cover content,
	// tool calls and results, usage and errors.
	rc := http.NewResponseController(w)
	ticker := time.NewTicker(h.keepAlive)
	defer ticker.Stop()
	for {
		select {
		case chunk, ok := <-stream:
			if !ok {
				return
			}
			// Ensure the pending slot exists before the SSE frame hits the wire so a
			// fast client posting to /v1/chat/approve never races the worker goroutine.
			if chunk.Type == agent.ChunkTypeApprovalRequired {
				h.approvals.ensure(req.SessionID, req.UserID, chunk.ToolCallID)
			}
			payload, err := json.Marshal(chunk)
			if err != nil {
				h.logger().Error("failed to marshal SSE chunk", "err", err)
				continue
			}
			if _, err := fmt.Fprintf(w, "data: %s\n\n", payload); err != nil {
				return // client went away
			}
			_ = rc.Flush()
		case <-ticker.C:
			if _, err := fmt.Fprint(w, ": keepalive\n\n"); err != nil {
				return
			}
			_ = rc.Flush()
		}
	}
}

type approveRequest struct {
	UserID     string `json:"user_id"`
	SessionID  string `json:"session_id"`
	ToolCallID string `json:"tool_call_id"`
	Approved   bool   `json:"approved"`
	Reason     string `json:"reason"`
}

func (h *handler) approveChat(w http.ResponseWriter, r *http.Request) {
	var who Identity
	if h.identify != nil {
		var err error
		if who, err = h.identify(r); err != nil {
			writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "unauthorized"})
			return
		}
	}
	var body approveRequest
	r.Body = http.MaxBytesReader(w, r.Body, 64<<10)
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON body: " + err.Error()})
		return
	}
	if strings.TrimSpace(body.SessionID) == "" || strings.TrimSpace(body.ToolCallID) == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "session_id and tool_call_id are required"})
		return
	}
	userID := body.UserID
	if h.identify != nil {
		userID = who.UserID
	}
	if userID == "" {
		userID = "anonymous"
	}
	if !h.approvals.resolve(body.SessionID, userID, body.ToolCallID, agent.ApprovalDecision{
		Approved: body.Approved,
		Reason:   body.Reason,
	}) {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "no pending approval found"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

type approvalKey struct {
	sessionID  string
	toolCallID string
}

type pendingApproval struct {
	userID  string
	ch      chan agent.ApprovalDecision
	decided bool
}

type approvalBroker struct {
	mu      sync.Mutex
	pending map[approvalKey]*pendingApproval
}

func newApprovalBroker() *approvalBroker {
	return &approvalBroker{pending: make(map[approvalKey]*pendingApproval)}
}

func (b *approvalBroker) ensure(sessionID, userID, toolCallID string) *pendingApproval {
	b.mu.Lock()
	defer b.mu.Unlock()
	key := approvalKey{sessionID: sessionID, toolCallID: toolCallID}
	if p, ok := b.pending[key]; ok {
		return p
	}
	p := &pendingApproval{userID: userID, ch: make(chan agent.ApprovalDecision, 1)}
	b.pending[key] = p
	return p
}

func (b *approvalBroker) resolve(sessionID, userID, toolCallID string, dec agent.ApprovalDecision) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	key := approvalKey{sessionID: sessionID, toolCallID: toolCallID}
	p, ok := b.pending[key]
	if !ok || p.userID != userID || p.decided {
		return false
	}
	p.decided = true
	delete(b.pending, key)
	p.ch <- dec
	return true
}

func (b *approvalBroker) remove(sessionID, toolCallID string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	delete(b.pending, approvalKey{sessionID: sessionID, toolCallID: toolCallID})
}

func (b *approvalBroker) clearSession(sessionID string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	for k := range b.pending {
		if k.sessionID == sessionID {
			delete(b.pending, k)
		}
	}
}

// storeAttachments writes non-image attachments into the session
// workspace, where the model opens them with the read_file tool; the engine
// receives only their names. Images stay inline as vision input.
func (h *handler) storeAttachments(req *agent.ChatRequest) error {
	for i := range req.Attachments {
		a := &req.Attachments[i]
		if strings.HasPrefix(a.MediaType, "image/") {
			continue
		}
		if h.filesDir == "" {
			return badRequest("attachment %q is not an image: storing it needs the file tools (tools.enabled: read_file)", a.Name)
		}
		name, err := attachmentName(a.Name)
		if err != nil {
			return err
		}
		data, err := base64.StdEncoding.DecodeString(a.Data)
		if err != nil {
			return badRequest("attachment %q: data is not valid base64: %v", a.Name, err)
		}
		dir, err := tools.SessionWorkspace(h.filesDir, req.UserID, req.SessionID)
		if err != nil {
			return err
		}
		if err := tools.StoreFile(dir, name, data); err != nil {
			return fmt.Errorf("attachment %q: %w", name, err)
		}
		a.Name, a.Data = name, ""
	}
	return nil
}

// attachmentName keeps the client's file name but never its path: the
// file lands directly in the session workspace.
func attachmentName(name string) (string, error) {
	base := strings.TrimSpace(path.Base(strings.ReplaceAll(name, "\\", "/")))
	if base == "" || base == "." || base == ".." || base == "/" {
		return "", badRequest("attachment %q has no usable file name", name)
	}
	if strings.IndexFunc(base, unicode.IsControl) >= 0 {
		return "", badRequest("attachment %q: control characters in the file name", name)
	}
	return base, nil
}
