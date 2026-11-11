package agent

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/shalyhinpavel/go-agent-harness/internal/safename"
	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// SessionStore keeps conversations beyond the process. Plug one into a
// SessionManager with WithSessionStore: every message is written through,
// and a session the manager does not hold in memory (after a restart, or
// evicted by TTL or the session cap) is loaded back from it.
//
// The usage counters are per process and are not stored.
type SessionStore interface {
	// Load returns the stored session, or (nil, nil) when there is none.
	Load(ctx context.Context, sessionID string) (*StoredSession, error)
	// Create records a new, empty session and its owner.
	Create(ctx context.Context, sessionID, userID string, createdAt time.Time) error
	// Append adds messages to the end of the session's history, in order.
	Append(ctx context.Context, sessionID string, msgs ...llm.Message) error
	// Delete removes the session; a missing one is not an error.
	Delete(ctx context.Context, sessionID string) error
	// Compact replaces the stored history of an existing session with
	// history, keeping its owner and creation time. A failed write leaves
	// the old history in place.
	Compact(ctx context.Context, sessionID string, history []llm.Message) error
}

// StoredSession is a session as a SessionStore holds it.
type StoredSession struct {
	ID        string
	UserID    string
	CreatedAt time.Time
	History   []llm.Message
}

// FileSessionStore keeps each session as one JSONL file: a header line with
// the owner, then one line per message. Appending never rewrites what is
// there, so a crash costs at most the line being written.
type FileSessionStore struct {
	dir string
	mu  sync.Mutex
}

// NewFileSessionStore creates a store under dir (created if missing).
func NewFileSessionStore(dir string) (*FileSessionStore, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("agent: session store: %w", err)
	}
	return &FileSessionStore{dir: dir}, nil
}

// sessionLine is one JSONL record: a header ("session") or a "message".
type sessionLine struct {
	Type      string       `json:"type"`
	ID        string       `json:"id,omitempty"`
	UserID    string       `json:"user_id,omitempty"`
	CreatedAt time.Time    `json:"created_at,omitzero"`
	Message   *llm.Message `json:"message,omitempty"`
}

func (s *FileSessionStore) path(sessionID string) string {
	return filepath.Join(s.dir, safename.Of(sessionID, "session")+".jsonl")
}

// Load implements SessionStore. A torn last line (a crash mid-write) is
// dropped; damage anywhere else is an error rather than a silently shorter
// conversation.
func (s *FileSessionStore) Load(ctx context.Context, sessionID string) (*StoredSession, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.load(sessionID)
}

// load reads the session file; the caller holds s.mu.
func (s *FileSessionStore) load(sessionID string) (*StoredSession, error) {
	f, err := os.Open(s.path(sessionID))
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("agent: session store: %w", err)
	}
	defer f.Close()

	var sess *StoredSession
	r := bufio.NewReader(f)
	for {
		raw, readErr := r.ReadBytes('\n')
		torn := readErr == io.EOF // no newline: the write may be cut short
		if len(raw) > 0 {
			var line sessionLine
			if err := json.Unmarshal(raw, &line); err != nil {
				if torn {
					break
				}
				return nil, fmt.Errorf("agent: session store: %s is damaged: %w", filepath.Base(f.Name()), err)
			}
			switch {
			case line.Type == "session" && sess == nil:
				if line.ID != sessionID {
					return nil, fmt.Errorf("agent: session store: %s holds session %q, not %q", filepath.Base(f.Name()), line.ID, sessionID)
				}
				sess = &StoredSession{ID: line.ID, UserID: line.UserID, CreatedAt: line.CreatedAt}
			case line.Type == "message" && sess != nil && line.Message != nil:
				sess.History = append(sess.History, *line.Message)
			default:
				return nil, fmt.Errorf("agent: session store: %s: unexpected %q record", filepath.Base(f.Name()), line.Type)
			}
		}
		if readErr != nil {
			if readErr != io.EOF {
				return nil, fmt.Errorf("agent: session store: %w", readErr)
			}
			break
		}
	}
	if sess == nil {
		return nil, fmt.Errorf("agent: session store: %s has no header", filepath.Base(f.Name()))
	}
	return sess, nil
}

// Create implements SessionStore.
func (s *FileSessionStore) Create(ctx context.Context, sessionID, userID string, createdAt time.Time) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	f, err := os.OpenFile(s.path(sessionID), os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return fmt.Errorf("agent: session store: %w", err)
	}
	return writeLines(f, sessionLine{Type: "session", ID: sessionID, UserID: userID, CreatedAt: createdAt.UTC()})
}

// Append implements SessionStore.
func (s *FileSessionStore) Append(ctx context.Context, sessionID string, msgs ...llm.Message) error {
	if len(msgs) == 0 {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	f, err := os.OpenFile(s.path(sessionID), os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return fmt.Errorf("agent: session store: %w", err)
	}
	lines := make([]sessionLine, len(msgs))
	for i := range msgs {
		lines[i] = sessionLine{Type: "message", Message: &msgs[i]}
	}
	return writeLines(f, lines...)
}

// Delete implements SessionStore.
func (s *FileSessionStore) Delete(ctx context.Context, sessionID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := os.Remove(s.path(sessionID)); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("agent: session store: %w", err)
	}
	return nil
}

// Compact implements SessionStore. The rewritten file is built beside the
// original and renamed over it, so a crash or a failed write leaves the old
// history in place.
func (s *FileSessionStore) Compact(ctx context.Context, sessionID string, history []llm.Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	stored, err := s.load(sessionID)
	if err != nil {
		return err
	}
	if stored == nil {
		return fmt.Errorf("agent: session store: session %q is not stored", sessionID)
	}
	lines := []sessionLine{{Type: "session", ID: sessionID, UserID: stored.UserID, CreatedAt: stored.CreatedAt.UTC()}}
	for i := range history {
		lines = append(lines, sessionLine{Type: "message", Message: &history[i]})
	}
	buf, err := encodeLines(lines...)
	if err != nil {
		return err
	}

	tmp, err := os.CreateTemp(s.dir, "compact-*.tmp")
	if err != nil {
		return fmt.Errorf("agent: session store: %w", err)
	}
	defer os.Remove(tmp.Name()) // a no-op once the rename has moved the file

	_, err = tmp.Write(buf)
	if serr := tmp.Sync(); err == nil {
		err = serr
	}
	if cerr := tmp.Close(); err == nil {
		err = cerr
	}
	if err == nil {
		err = os.Rename(tmp.Name(), s.path(sessionID))
	}
	if err != nil {
		return fmt.Errorf("agent: session store: %w", err)
	}
	return nil
}

// encodeLines renders records as JSONL: one line each.
func encodeLines(lines ...sessionLine) ([]byte, error) {
	var buf []byte
	for _, l := range lines {
		raw, err := json.Marshal(l)
		if err != nil {
			return nil, fmt.Errorf("agent: session store: %w", err)
		}
		buf = append(append(buf, raw...), '\n')
	}
	return buf, nil
}

// writeLines writes the records in one write call and closes f, so a batch
// (an assistant message with its tool results) lands together or not at all
// in all but a crash at the worst moment.
func writeLines(f *os.File, lines ...sessionLine) error {
	buf, err := encodeLines(lines...)
	if err != nil {
		f.Close()
		return err
	}
	_, err = f.Write(buf)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return fmt.Errorf("agent: session store: %w", err)
	}
	return nil
}
