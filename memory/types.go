package memory

import (
	"context"
	"time"
)

// UserFact is one atomic vectorizable fact about a user.
type UserFact struct {
	ID         string    `json:"id"`
	UserID     string    `json:"user_id"`
	Category   string    `json:"category"` // e.g. "work", "preference", "budget"
	Fact       string    `json:"fact"`     // e.g. "Works as a data engineer in Berlin"
	Tags       []string  `json:"tags,omitempty"`
	Confidence float64   `json:"confidence"` // 0.0 - 1.0
	Embedding  []float32 `json:"embedding,omitempty"`
	CreatedAt  time.Time `json:"created_at"`
	UpdatedAt  time.Time `json:"updated_at"`
	Score      float64   `json:"score,omitempty"` // final score including time decay
}

// UserDossier is the compact "hot" profile of a user — a strictly bounded
// markdown block (~500 tokens) injected into every system prompt.
type UserDossier struct {
	UserID    string    `json:"user_id"`
	Content   string    `json:"content"`
	UpdatedAt time.Time `json:"updated_at"`
}

// MemoryStore is the storage port behind the memory module. Implement it
// with PostgreSQL+pgvector, SQLite, Firestore — anything that can hold
// JSON facts and answer a similarity search. The query text and the
// optional embedding both reach the store so backends can run hybrid
// (lexical + vector) search.
type MemoryStore interface {
	GetDossier(ctx context.Context, userID string) (*UserDossier, error)
	SaveDossier(ctx context.Context, dossier *UserDossier) error
	InsertFact(ctx context.Context, fact *UserFact) error
	// SearchFacts returns up to limit facts ranked for the query; the
	// embedding is used when the backend supports vectors (nil = lexical
	// mode only).
	SearchFacts(ctx context.Context, userID, query string, embedding []float32, limit int) ([]*UserFact, error)
	DeleteFact(ctx context.Context, userID, factID string) error
}
