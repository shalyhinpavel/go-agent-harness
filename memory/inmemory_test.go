package memory

import (
	"context"
	"testing"
	"time"
)

func TestInMemoryStore_GetDossier_NonExistent(t *testing.T) {
	store := NewInMemoryStore()

	dossier, err := store.GetDossier(context.Background(), "nonexistent_user")

	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if dossier.UserID != "nonexistent_user" {
		t.Errorf("Expected user ID 'nonexistent_user', got %s", dossier.UserID)
	}
	if dossier.Content != "" {
		t.Errorf("Expected empty content for nonexistent user, got %s", dossier.Content)
	}
}

func TestInMemoryStore_SaveAndGetDossier(t *testing.T) {
	store := NewInMemoryStore()

	dossier := &UserDossier{
		UserID:  "user1",
		Content: "- Test content",
	}

	err := store.SaveDossier(context.Background(), dossier)
	if err != nil {
		t.Fatalf("SaveDossier failed: %v", err)
	}

	retrieved, err := store.GetDossier(context.Background(), "user1")
	if err != nil {
		t.Fatalf("GetDossier failed: %v", err)
	}

	if retrieved.Content != dossier.Content {
		t.Errorf("Expected content %s, got %s", dossier.Content, retrieved.Content)
	}

	if retrieved.UpdatedAt.IsZero() {
		t.Error("Expected UpdatedAt to be set")
	}
}

func TestInMemoryStore_InsertFact(t *testing.T) {
	store := NewInMemoryStore()

	fact := &UserFact{
		UserID:     "user1",
		Category:   "test",
		Fact:       "Test fact",
		Confidence: 0.9,
	}

	err := store.InsertFact(context.Background(), fact)
	if err != nil {
		t.Fatalf("InsertFact failed: %v", err)
	}

	if fact.ID == "" {
		t.Error("Expected ID to be generated")
	}

	if fact.CreatedAt.IsZero() {
		t.Error("Expected CreatedAt to be set")
	}

	if fact.UpdatedAt.IsZero() {
		t.Error("Expected UpdatedAt to be set")
	}
}

func TestInMemoryStore_InsertFact_WithExistingID(t *testing.T) {
	store := NewInMemoryStore()

	fact := &UserFact{
		ID:       "custom-id",
		UserID:   "user1",
		Category: "test",
		Fact:     "Test fact",
	}

	_ = store.InsertFact(context.Background(), fact)

	if fact.ID != "custom-id" {
		t.Errorf("Expected ID to remain 'custom-id', got %s", fact.ID)
	}
}

func TestInMemoryStore_InsertFact_WithExistingCreatedAt(t *testing.T) {
	store := NewInMemoryStore()

	customTime := time.Now().Add(-24 * time.Hour)
	fact := &UserFact{
		UserID:    "user1",
		Category:  "test",
		Fact:      "Test fact",
		CreatedAt: customTime,
	}

	_ = store.InsertFact(context.Background(), fact)

	if !fact.CreatedAt.Equal(customTime) {
		t.Error("Expected CreatedAt to remain unchanged")
	}
}

func TestInMemoryStore_SearchFacts_NonExistentUser(t *testing.T) {
	store := NewInMemoryStore()

	facts, err := store.SearchFacts(context.Background(), "nonexistent_user", "", []float32{0.1}, 10)

	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if facts != nil {
		t.Errorf("Expected nil facts for nonexistent user, got %v", facts)
	}
}

func TestInMemoryStore_SearchFacts_EmptyFacts(t *testing.T) {
	store := NewInMemoryStore()

	// The user exists but has no facts.
	store.facts["user1"] = []*UserFact{}

	facts, err := store.SearchFacts(context.Background(), "user1", "", []float32{0.1}, 10)

	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if facts != nil {
		t.Errorf("Expected nil facts for empty user, got %v", facts)
	}
}

func TestInMemoryStore_SearchFacts_Limit(t *testing.T) {
	store := NewInMemoryStore()

	for i := 0; i < 5; i++ {
		_ = store.InsertFact(context.Background(), &UserFact{
			UserID:    "user1",
			Category:  "test",
			Fact:      "Fact",
			CreatedAt: time.Now(),
		})
	}

	facts, err := store.SearchFacts(context.Background(), "user1", "python", []float32{0.1}, 3)

	if err != nil {
		t.Fatalf("SearchFacts failed: %v", err)
	}
	if len(facts) != 3 {
		t.Errorf("Expected 3 facts, got %d", len(facts))
	}
}

func TestInMemoryStore_SearchFacts_SortedByScore(t *testing.T) {
	store := NewInMemoryStore()

	oldFact := &UserFact{
		UserID:    "user1",
		Category:  "test",
		Fact:      "Old fact",
		CreatedAt: time.Now().Add(-60 * 24 * time.Hour),
	}
	newFact := &UserFact{
		UserID:    "user1",
		Category:  "test",
		Fact:      "New fact",
		CreatedAt: time.Now(),
	}

	_ = store.InsertFact(context.Background(), oldFact)
	_ = store.InsertFact(context.Background(), newFact)

	facts, err := store.SearchFacts(context.Background(), "user1", "", []float32{0.1}, 10)

	if err != nil {
		t.Fatalf("SearchFacts failed: %v", err)
	}
	if len(facts) != 2 {
		t.Fatalf("Expected 2 facts, got %d", len(facts))
	}

	// Same similarity, so the newer fact must rank first.
	if facts[0].Score <= facts[1].Score {
		t.Errorf("Expected first fact to have higher score: %f <= %f", facts[0].Score, facts[1].Score)
	}
}

func TestInMemoryStore_DeleteFact_Success(t *testing.T) {
	store := NewInMemoryStore()

	_ = store.InsertFact(context.Background(), &UserFact{ID: "fact1", UserID: "user1", Fact: "Fact 1"})
	_ = store.InsertFact(context.Background(), &UserFact{ID: "fact2", UserID: "user1", Fact: "Fact 2"})

	err := store.DeleteFact(context.Background(), "user1", "fact1")

	if err != nil {
		t.Fatalf("DeleteFact failed: %v", err)
	}

	facts, _ := store.SearchFacts(context.Background(), "user1", "", nil, 10)
	if len(facts) != 1 {
		t.Errorf("Expected 1 fact after deletion, got %d", len(facts))
	}
	if facts[0].ID != "fact2" {
		t.Errorf("Expected remaining fact to be fact2, got %s", facts[0].ID)
	}
}

func TestInMemoryStore_DeleteFact_NotFound(t *testing.T) {
	store := NewInMemoryStore()

	_ = store.InsertFact(context.Background(), &UserFact{ID: "fact1", UserID: "user1", Fact: "Fact 1"})

	err := store.DeleteFact(context.Background(), "user1", "nonexistent")

	if err == nil {
		t.Error("Expected error when deleting nonexistent fact")
	}

	facts, _ := store.SearchFacts(context.Background(), "user1", "", nil, 10)
	if len(facts) != 1 {
		t.Errorf("Expected 1 fact to remain, got %d", len(facts))
	}
}

func TestInMemoryStore_DeleteFact_UserNotFound(t *testing.T) {
	store := NewInMemoryStore()

	err := store.DeleteFact(context.Background(), "nonexistent_user", "fact1")

	if err == nil {
		t.Error("Expected error when deleting from nonexistent user")
	}
}

func TestInMemoryStore_NonPositiveLimitDefaults(t *testing.T) {
	store := NewInMemoryStore()
	for _, f := range []string{"a", "b"} {
		if err := store.InsertFact(context.Background(), &UserFact{UserID: "u", Fact: f}); err != nil {
			t.Fatal(err)
		}
	}
	for _, limit := range []int{-1, 0} {
		got, err := store.SearchFacts(context.Background(), "u", "a", nil, limit)
		if err != nil || len(got) != 2 {
			t.Fatalf("limit %d: want both facts, got %d, err %v", limit, len(got), err)
		}
	}
}
