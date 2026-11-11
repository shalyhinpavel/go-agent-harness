package memory

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// InMemoryStore is the built-in memory store: good for local development
// and tests. Swap it for a real backend via the MemoryStore interface.
type InMemoryStore struct {
	mu       sync.RWMutex
	dossiers map[string]*UserDossier
	facts    map[string][]*UserFact
}

// NewInMemoryStore creates an empty in-memory store.
func NewInMemoryStore() *InMemoryStore {
	return &InMemoryStore{
		dossiers: make(map[string]*UserDossier),
		facts:    make(map[string][]*UserFact),
	}
}

func (s *InMemoryStore) GetDossier(_ context.Context, userID string) (*UserDossier, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if d, exists := s.dossiers[userID]; exists {
		return d, nil
	}
	return &UserDossier{UserID: userID, Content: ""}, nil
}

func (s *InMemoryStore) SaveDossier(_ context.Context, dossier *UserDossier) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	dossier.UpdatedAt = time.Now()
	s.dossiers[dossier.UserID] = dossier
	return nil
}

func (s *InMemoryStore) InsertFact(_ context.Context, fact *UserFact) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if fact.ID == "" {
		fact.ID = uuid.New().String()
	}
	if fact.CreatedAt.IsZero() {
		fact.CreatedAt = time.Now()
	}
	fact.UpdatedAt = time.Now()

	s.facts[fact.UserID] = append(s.facts[fact.UserID], fact)
	return nil
}

// SearchFacts ranks facts by decayed score. Substring matches on the
// query score higher, so ranking is query-then-recency — good enough for
// development; semantic search wants a real vector store.
func (s *InMemoryStore) SearchFacts(_ context.Context, userID, query string, embedding []float32, limit int) ([]*UserFact, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	userFacts, exists := s.facts[userID]
	if !exists || len(userFacts) == 0 {
		return nil, nil
	}

	if limit <= 0 {
		limit = 5 // the same default as FileStore
	}

	q := strings.ToLower(query)
	result := make([]*UserFact, 0, len(userFacts))
	for _, f := range userFacts {
		fCopy := *f
		sim := 0.5 // base similarity for unrelated facts
		if q != "" && strings.Contains(strings.ToLower(f.Fact), q) {
			sim = 0.95 // direct textual hit
		}
		fCopy.Score = CalculateScore(sim, f.CreatedAt, 30.0)
		result = append(result, &fCopy)
	}

	sort.Slice(result, func(i, j int) bool {
		return result[i].Score > result[j].Score
	})

	if len(result) > limit {
		result = result[:limit]
	}

	return result, nil
}

func (s *InMemoryStore) DeleteFact(_ context.Context, userID, factID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	userFacts, exists := s.facts[userID]
	if !exists {
		return fmt.Errorf("no facts found for user %s", userID)
	}

	newFacts := make([]*UserFact, 0, len(userFacts))
	found := false
	for _, f := range userFacts {
		if f.ID != factID {
			newFacts = append(newFacts, f)
		} else {
			found = true
		}
	}
	if !found {
		return fmt.Errorf("fact %s not found for user %s", factID, userID)
	}

	s.facts[userID] = newFacts
	return nil
}
