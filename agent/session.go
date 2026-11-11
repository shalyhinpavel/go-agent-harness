package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// Session holds the state of one conversation.
type Session struct {
	ID      string
	UserID  string
	History []llm.Message
	// Usage accumulates reported token accounting across the session.
	Usage llm.Usage
	// LastInput/LastOutput are the latest call's reported sizes — the
	// basis for projecting the next request against the context limit.
	LastInput  int
	LastOutput int
	// windowStart is where the history window began on the last request
	// (see SessionManager.Window); kept so the prefix stays stable.
	windowStart int
	// reasoningBinding fingerprints the system prompt and tools the stored
	// reasoning was produced under; "" means unknown (see BindReasoning).
	reasoningBinding string
	CreatedAt        time.Time
	UpdatedAt        time.Time
}

// SessionManager creates, stores and evicts sessions in memory.
//
// Eviction is lazy, so no background goroutine is needed: a session whose
// last update is older than the TTL resets on next access, and creating a
// session beyond MaxSessions evicts the least recently updated one. With a
// SessionStore (WithSessionStore) eviction only drops the in-memory copy:
// the session comes back from the store on its next use.
type SessionManager struct {
	mu          sync.RWMutex
	sessions    map[string]*Session
	busy        map[string]bool // sessions with a turn in flight
	ttl         time.Duration
	maxSessions int
	store       SessionStore // nil = memory only
}

// SessionOption configures a SessionManager.
type SessionOption func(*SessionManager)

// WithSessionTTL resets a session that has been idle longer than d.
func WithSessionTTL(d time.Duration) SessionOption {
	return func(sm *SessionManager) { sm.ttl = d }
}

// WithMaxSessions caps the number of live sessions (LRU eviction).
func WithMaxSessions(n int) SessionOption {
	return func(sm *SessionManager) { sm.maxSessions = n }
}

// WithSessionStore writes every message through to store and loads sessions
// from it when they are not in memory.
func WithSessionStore(store SessionStore) SessionOption {
	return func(sm *SessionManager) { sm.store = store }
}

// NewSessionManager creates a manager. Zero options means unbounded storage.
func NewSessionManager(opts ...SessionOption) *SessionManager {
	sm := &SessionManager{sessions: make(map[string]*Session), busy: make(map[string]bool)}
	for _, opt := range opts {
		opt(sm)
	}
	return sm
}

// GetOrCreateSession returns the session or creates a fresh one; with a
// store, a session it holds is loaded back. An empty sessionID gets a
// generated UUID (returned via the session). The error is the store's.
func (sm *SessionManager) GetOrCreateSession(ctx context.Context, sessionID, userID string) (*Session, error) {
	if sessionID == "" {
		sessionID = uuid.New().String()
	}

	sm.mu.Lock()
	if sess, exists := sm.sessions[sessionID]; exists {
		if sm.ttl > 0 && !sm.busy[sessionID] && time.Since(sess.UpdatedAt) > sm.ttl {
			delete(sm.sessions, sessionID) // expired — reload or recreate below
		} else {
			sess.UpdatedAt = time.Now()
			sm.mu.Unlock()
			return sess, nil
		}
	}
	sm.mu.Unlock()

	// Store I/O happens outside the lock: it must not stall other sessions.
	var stored *StoredSession
	if sm.store != nil {
		var err error
		if stored, err = sm.store.Load(ctx, sessionID); err != nil {
			return nil, err
		}
	}
	fresh := &Session{
		ID:        sessionID,
		UserID:    userID,
		History:   make([]llm.Message, 0),
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	if stored != nil {
		fresh.UserID, fresh.CreatedAt = stored.UserID, stored.CreatedAt
		fresh.History = repairTail(stored.History)
	}

	sm.mu.Lock()
	defer sm.mu.Unlock()
	if sess, exists := sm.sessions[sessionID]; exists {
		sess.UpdatedAt = time.Now() // another caller got there first
		return sess, nil
	}
	if stored == nil && sm.store != nil {
		if err := sm.store.Create(ctx, sessionID, userID, fresh.CreatedAt); err != nil {
			return nil, err
		}
	}
	sm.evictForLocked()
	sm.sessions[sessionID] = fresh
	return fresh, nil
}

// evictForLocked makes room for one more session (LRU, never mid-turn).
func (sm *SessionManager) evictForLocked() {
	if sm.maxSessions <= 0 || len(sm.sessions) < sm.maxSessions {
		return
	}
	var oldestID string
	var oldest time.Time
	for id, s := range sm.sessions {
		if sm.busy[id] {
			continue // never evict a session mid-turn
		}
		if oldestID == "" || s.UpdatedAt.Before(oldest) {
			oldestID, oldest = id, s.UpdatedAt
		}
	}
	if oldestID != "" {
		delete(sm.sessions, oldestID)
	}
}

// repairTail drops an assistant message whose tool calls never got their
// results — a crash between the two writes. Providers reject such a history.
func repairTail(h []llm.Message) []llm.Message {
	if n := len(h); n > 0 && h[n-1].Role == llm.RoleAssistant {
		for _, p := range h[n-1].Parts {
			if p.Kind == llm.PartToolCall {
				return h[:n-1]
			}
		}
	}
	return h
}

// Delete forgets the session in memory and in the store.
func (sm *SessionManager) Delete(ctx context.Context, sessionID string) error {
	sm.mu.Lock()
	delete(sm.sessions, sessionID)
	sm.mu.Unlock()
	if sm.store != nil {
		return sm.store.Delete(ctx, sessionID)
	}
	return nil
}

// BeginTurn claims the session for one turn. Two interleaved turns would
// splice their tool calls and results into one history — a broken
// conversation every later request inherits — so a second concurrent
// turn is refused (ok=false) instead of queued. release must be called
// when the turn ends.
func (sm *SessionManager) BeginTurn(sessionID string) (release func(), ok bool) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	if sm.busy[sessionID] {
		return nil, false
	}
	sm.busy[sessionID] = true
	return func() {
		sm.mu.Lock()
		defer sm.mu.Unlock()
		delete(sm.busy, sessionID)
	}, true
}

// Owner reports the user a session belongs to, from memory or else from the
// store, so the answer survives a restart. found is false for a session that
// does not exist yet.
func (sm *SessionManager) Owner(ctx context.Context, sessionID string) (userID string, found bool, err error) {
	sm.mu.RLock()
	if sess, ok := sm.sessions[sessionID]; ok {
		userID = sess.UserID
		sm.mu.RUnlock()
		return userID, true, nil
	}
	sm.mu.RUnlock()
	if sm.store == nil {
		return "", false, nil
	}
	stored, err := sm.store.Load(ctx, sessionID)
	if err != nil || stored == nil {
		return "", false, err
	}
	return stored.UserID, true, nil
}

// Busy reports whether a turn is in flight on the session.
func (sm *SessionManager) Busy(sessionID string) bool {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	return sm.busy[sessionID]
}

// AppendMessage adds a message to the session history.
func (sm *SessionManager) AppendMessage(ctx context.Context, sessionID string, msg *llm.Message) error {
	return sm.AppendMessages(ctx, sessionID, msg)
}

// AppendMessages adds messages to the session history in one step; with a
// store they are written first, so history in memory never runs ahead of
// what is durable.
func (sm *SessionManager) AppendMessages(ctx context.Context, sessionID string, msgs ...*llm.Message) error {
	var add []llm.Message
	for _, m := range msgs {
		if m != nil {
			add = append(add, *m)
		}
	}
	if len(add) == 0 {
		return nil
	}
	if sm.store != nil {
		sm.mu.RLock()
		_, known := sm.sessions[sessionID]
		sm.mu.RUnlock()
		if known {
			if err := sm.store.Append(ctx, sessionID, add...); err != nil {
				return fmt.Errorf("agent: save messages of session %q: %w", sessionID, err)
			}
		}
	}

	sm.mu.Lock()
	defer sm.mu.Unlock()
	if sess, exists := sm.sessions[sessionID]; exists {
		sess.History = append(sess.History, add...)
		sess.UpdatedAt = time.Now()
	}
	return nil
}

// AddUsage accumulates a usage report into the session's totals and
// remembers the latest call's input/output sizes.
func (sm *SessionManager) AddUsage(sessionID string, u llm.Usage) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	if sess, exists := sm.sessions[sessionID]; exists {
		sess.Usage.Add(u)
		// An estimate (a stream cut short reports one, with no input count)
		// must not overwrite the last real sizes: the context guard reads
		// them, and a zero would switch it off for the next turn.
		if !u.Estimated && u.InputTokens > 0 {
			sess.LastInput = u.InputTokens
			sess.LastOutput = u.OutputTokens
		}
	}
}

// AddToolUsage accumulates what a tool spent on its own model calls into
// the session totals. It leaves the last-call sizes alone: those describe
// the agent's own context and feed the context guard.
func (sm *SessionManager) AddToolUsage(sessionID string, u llm.Usage) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	if sess, exists := sm.sessions[sessionID]; exists {
		sess.Usage.Add(u)
	}
}

// LastUsage returns the latest call's reported input and output tokens
// (0, 0 when nothing was reported yet).
func (sm *SessionManager) LastUsage(sessionID string) (input, output int) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	if sess, exists := sm.sessions[sessionID]; exists {
		return sess.LastInput, sess.LastOutput
	}
	return 0, 0
}

// SessionUsage returns the accumulated token accounting for the session.
func (sm *SessionManager) SessionUsage(sessionID string) llm.Usage {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	if sess, exists := sm.sessions[sessionID]; exists {
		return sess.Usage
	}
	return llm.Usage{}
}

// GetHistory returns a copy of the history slice (messages are treated as
// immutable once appended).
func (sm *SessionManager) GetHistory(sessionID string) []llm.Message {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	if sess, exists := sm.sessions[sessionID]; exists {
		out := make([]llm.Message, len(sess.History))
		copy(out, sess.History)
		return out
	}
	return nil
}

// WindowPolicy bounds the history sent to the model on each request.
//
// The window starts on an exchange boundary (never on a tool result, so a
// tool call always travels with its results) and is optionally preceded by
// the pinned task: the first user message of the session, which a long
// tool loop would otherwise push out. MaxItems and MaxChars cap the part
// after the pinned head; either may be zero (no cap).
//
// TrimTo adds hysteresis: when the window overflows a cap, it is cut down
// to TrimTo of the caps and then left alone until it overflows again, so
// the request prefix stays byte-stable for many iterations and providers
// with prefix caching can reuse it. TrimTo <= 0 or >= 1 slides the window
// by the minimum on every overflow (prefix changes on every request).
type WindowPolicy struct {
	MaxItems int
	MaxChars int
	TrimTo   float64
	PinTask  bool
}

// Window returns the history to send under p and remembers where the
// window starts, so the next call keeps the same prefix until a cap is
// exceeded. When older messages are dropped, assistant thinking parts are
// stripped: reasoning blocks are only valid for the exact prefix that
// produced them. The returned slice is a fresh copy, safe to mutate.
func (sm *SessionManager) Window(sessionID string, p WindowPolicy) []llm.Message {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	sess, ok := sm.sessions[sessionID]
	if !ok {
		return nil
	}
	head := 0
	if p.PinTask {
		head = pinnedHead(sess.History)
	}
	start := sess.windowStart
	if start < head || start > len(sess.History) {
		start = head
	}
	start = nextWindowStart(sess.History, head, start, p)
	sess.windowStart = start
	return assembleWindow(sess.History, head, start)
}

// CompactHistory returns at most maxItems latest messages (fewer when the
// cap falls inside a tool exchange: the window then starts at the next
// exchange boundary). Stateless, no pinned task; see Window for the
// policy the engine uses.
func (sm *SessionManager) CompactHistory(sessionID string, maxItems int) []llm.Message {
	history := sm.GetHistory(sessionID)
	start := nextWindowStart(history, 0, 0, WindowPolicy{MaxItems: maxItems})
	return assembleWindow(history, 0, start)
}

// pinnedHead is the length of the pinned prefix: everything up to and
// including the first user message that is not a tool result (normally
// just message 0, the task).
func pinnedHead(h []llm.Message) int {
	for i := range h {
		if h[i].Role == llm.RoleUser && !isToolResult(&h[i]) {
			return i + 1
		}
	}
	return 0
}

// nextWindowStart keeps start while h[start:] fits the caps; otherwise it
// moves forward to the first exchange boundary where the window fits
// TrimTo of the caps, or to the last boundary when even one exchange is
// over the caps (the context guard then decides).
func nextWindowStart(h []llm.Message, head, start int, p WindowPolicy) int {
	n := len(h)
	// suffix[i] = size of h[i:] in chars.
	suffix := make([]int, n+1)
	for i := n - 1; i >= head; i-- {
		suffix[i] = suffix[i+1] + messageChars(&h[i])
	}
	fits := func(s int, scale float64) bool {
		if p.MaxItems > 0 && float64(n-s) > scale*float64(p.MaxItems) {
			return false
		}
		if p.MaxChars > 0 && float64(suffix[s]) > scale*float64(p.MaxChars) {
			return false
		}
		return true
	}
	if fits(start, 1) {
		return start
	}
	scale := p.TrimTo
	if scale <= 0 || scale >= 1 {
		scale = 1
	}
	last := start
	for c := start + 1; c < n; c++ {
		if isToolResult(&h[c]) {
			continue
		}
		last = c
		if fits(c, scale) {
			return c
		}
	}
	return last
}

// assembleWindow copies h[:head] + h[start:]; when anything between them
// was dropped, thinking parts are stripped (new slices only: the history's
// arrays are shared).
func assembleWindow(h []llm.Message, head, start int) []llm.Message {
	out := make([]llm.Message, 0, head+len(h)-start)
	out = append(out, h[:head]...)
	out = append(out, h[start:]...)
	if start > head {
		return stripThinking(out)
	}
	return out
}

// stripThinking returns the messages without their thinking parts. A
// reasoning block is valid only for the conversation that produced it, so
// every rewrite that drops or moves earlier messages strips all of them:
// removing some and keeping the rest would leave those unreadable. The
// input's part slices are shared with the history and are not changed.
func stripThinking(msgs []llm.Message) []llm.Message {
	out := make([]llm.Message, len(msgs))
	copy(out, msgs)
	for i := range out {
		kept := make([]llm.Part, 0, len(out[i].Parts))
		for _, part := range out[i].Parts {
			if part.Kind != llm.PartThinking {
				kept = append(kept, part)
			}
		}
		if len(kept) != len(out[i].Parts) {
			out[i].Parts = kept
		}
	}
	return out
}

// hasThinking reports whether any message carries a thinking part.
func hasThinking(msgs []llm.Message) bool {
	for i := range msgs {
		for _, part := range msgs[i].Parts {
			if part.Kind == llm.PartThinking {
				return true
			}
		}
	}
	return false
}

// BindReasoning records the fingerprint of the system prompt and tools the
// next request carries. Stored reasoning produced under another fingerprint
// cannot be replayed under this one, so every thinking part is stripped from
// the history. The rewrite reaches the store first, as a compaction does: a
// refused write leaves the history as it was. A session with no recorded
// fingerprint (loaded after a restart) counts as changed, so its reasoning
// is dropped once rather than trusted.
func (sm *SessionManager) BindReasoning(ctx context.Context, sessionID, binding string) error {
	sm.mu.RLock()
	sess, ok := sm.sessions[sessionID]
	if !ok {
		sm.mu.RUnlock()
		return fmt.Errorf("agent: unknown session %q", sessionID)
	}
	if sess.reasoningBinding == binding {
		sm.mu.RUnlock()
		return nil
	}
	history := append([]llm.Message(nil), sess.History...)
	sm.mu.RUnlock()

	stripped := stripThinking(history)
	if sm.store != nil && hasThinking(history) {
		if err := sm.store.Compact(ctx, sessionID, stripped); err != nil {
			return fmt.Errorf("agent: rebind reasoning: %w", err)
		}
	}

	sm.mu.Lock()
	defer sm.mu.Unlock()
	if sess, ok := sm.sessions[sessionID]; ok {
		sess.History = stripped
		sess.reasoningBinding = binding
	}
	return nil
}

// messageChars approximates the payload size of a message: text, tool
// arguments and tool results as JSON. Images count by their data length.
func messageChars(m *llm.Message) int {
	size := 0
	for _, part := range m.Parts {
		size += len(part.Text)
		if part.ToolCall != nil {
			size += len(part.ToolCall.Name)
			if raw, err := json.Marshal(part.ToolCall.Args); err == nil {
				size += len(raw)
			}
		}
		if part.ToolResult != nil {
			if raw, err := json.Marshal(part.ToolResult.Content); err == nil {
				size += len(raw)
			}
		}
		if part.Image != nil {
			size += len(part.Image.Data) + len(part.Image.URL)
		}
	}
	return size
}

// isToolResult reports whether the message carries tool outcomes.
func isToolResult(msg *llm.Message) bool {
	if msg == nil {
		return false
	}
	for _, p := range msg.Parts {
		if p.Kind == llm.PartToolResult {
			return true
		}
	}
	return false
}
