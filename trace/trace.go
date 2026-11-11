// Package trace records agent trajectories as JSONL sidecar files —
// observability for debugging, cost accounting and eval replay.
//
// The trace shows what the agent did (steps, tool I/O, token usage,
// latency); it does not explain why the model decided anything —
// reconstructed "reasons" would be confabulation. Thinking deltas are
// recorded as the closest available window into the model's process.
package trace

import (
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/shalyhinpavel/go-agent-harness/internal/safename"
	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// Event types recorded on a trajectory.
const (
	EventTurnStart  = "turn_start"  // user message opened a turn
	EventText       = "text"        // assistant text delta
	EventThinking   = "thinking"    // reasoning delta (trace-only, never in history)
	EventToolCall   = "tool_call"   // tool invocation requested by the model
	EventToolResult = "tool_result" // tool outcome (full payload, unlike context)
	EventLLMCall    = "llm_call"    // one provider call: usage + latency
	EventTurnEnd    = "turn_end"    // turn finished (totals)
	EventError      = "error"       // fatal for the turn
	EventWarning    = "warning"     // the turn goes on, but something was off
)

// Event is one JSONL record of an agent trajectory.
type Event struct {
	Time       time.Time      `json:"ts"`
	SessionID  string         `json:"session_id"`
	TurnID     string         `json:"turn_id,omitempty"`
	Type       string         `json:"type"`
	Iteration  int            `json:"iteration,omitempty"`
	Text       string         `json:"text,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	ToolName   string         `json:"tool_name,omitempty"`
	ToolArgs   map[string]any `json:"tool_args,omitempty"`
	ToolResult any            `json:"tool_result,omitempty"`
	Usage      *llm.Usage     `json:"usage,omitempty"`
	StopReason string         `json:"stop_reason,omitempty"`
	LatencyMS  int64          `json:"latency_ms,omitempty"`
	Error      string         `json:"error,omitempty"`
	// Agent names a sub-agent whose steps land in the session's trace
	// (e.g. "researcher"); empty for the session's own agent.
	Agent string `json:"agent,omitempty"`
	// Request carries the exact model request behind an llm_call that
	// produced the turn's final answer: system, messages, tools. Recorded
	// only when trace.requests is on, because it repeats the turn's
	// history; replaying the model's context needs it verbatim instead of
	// a reconstruction.
	Request any `json:"request,omitempty"`
}

// Sink consumes trajectory events. Record must be safe for concurrent use
// and must not block the agent loop: implementations log write failures and
// move on.
type Sink interface {
	Record(event Event)
}

// maxOpenFiles bounds the session files a DirSink holds open at once. A
// server sees a new session per request, so without a bound the sink would
// run out of file descriptors; an evicted session's file is simply reopened
// for append if it speaks again.
const maxOpenFiles = 64

// DirSink writes one JSONL file per session inside a directory. Files are
// opened lazily and kept open for append, the least recently used one
// closed when more than maxOpenFiles are open; Close closes the rest.
type DirSink struct {
	// Logger receives write problems; trace failures never fail a turn
	// (nil = slog.Default()).
	Logger *slog.Logger

	mu    sync.Mutex
	dir   string
	files map[string]*openFile
	clock uint64
}

type openFile struct {
	f    *os.File
	used uint64
}

func (s *DirSink) logger() *slog.Logger {
	if s.Logger != nil {
		return s.Logger
	}
	return slog.Default()
}

// NewDirSink creates a sink writing to dir (created if missing).
func NewDirSink(dir string) (*DirSink, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	return &DirSink{dir: dir, files: make(map[string]*openFile)}, nil
}

// Record appends the event to the session's JSONL file.
func (s *DirSink) Record(event Event) {
	if event.Time.IsZero() {
		event.Time = time.Now().UTC()
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	f, err := s.file(event.SessionID)
	if err != nil {
		s.logger().Warn("trace: write skipped", "err", err, "session", event.SessionID)
		return
	}

	line, err := json.Marshal(event)
	if err != nil {
		s.logger().Warn("trace: marshal event", "err", err)
		return
	}
	if _, err := f.Write(append(line, '\n')); err != nil {
		s.logger().Warn("trace: write failed", "err", err, "session", event.SessionID)
	}
}

// file returns (opening on first use) the session's log file. Callers hold
// the mutex.
func (s *DirSink) file(sessionID string) (*os.File, error) {
	s.clock++
	if of, ok := s.files[sessionID]; ok {
		of.used = s.clock
		return of.f, nil
	}
	if len(s.files) >= maxOpenFiles {
		s.evictOldest()
	}

	name := sanitizeName(sessionID) + ".jsonl"
	f, err := os.OpenFile(filepath.Join(s.dir, name), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, err
	}
	s.files[sessionID] = &openFile{f: f, used: s.clock}
	return f, nil
}

// evictOldest closes the least recently used file. Callers hold the mutex.
func (s *DirSink) evictOldest() {
	var oldestID string
	var oldest uint64
	for id, of := range s.files {
		if oldestID == "" || of.used < oldest {
			oldestID, oldest = id, of.used
		}
	}
	if oldestID == "" {
		return
	}
	if err := s.files[oldestID].f.Close(); err != nil {
		s.logger().Warn("trace: close", "err", err, "session", oldestID)
	}
	delete(s.files, oldestID)
}

// Close closes every open session file.
func (s *DirSink) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()

	for id, of := range s.files {
		if err := of.f.Close(); err != nil {
			s.logger().Warn("trace: close", "err", err, "session", id)
		}
		delete(s.files, id)
	}
}

// sanitizeName keeps the file name inside the trace directory even for
// hostile session ids.
func sanitizeName(id string) string { return safename.Of(id, "session") }

// OpenFiles reports how many session files are open right now.
func (s *DirSink) OpenFiles() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.files)
}
