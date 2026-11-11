package memory

import (
	"testing"
	"time"
)

func TestCalculateScore_NoDecay(t *testing.T) {
	now := time.Now()
	similarity := 0.9

	// A fresh fact barely decays.
	score := CalculateScore(similarity, now, 30.0)

	if score < 0.89 {
		t.Errorf("Expected score close to similarity (0.9), got %f", score)
	}
}

func TestCalculateScore_HalfLifeDecay(t *testing.T) {
	// The fact is exactly one half-life old.
	halfLifeDays := 30.0
	createdAt := time.Now().Add(-time.Duration(halfLifeDays) * 24 * time.Hour)
	similarity := 1.0

	score := CalculateScore(similarity, createdAt, halfLifeDays)

	if score < 0.49 || score > 0.51 {
		t.Errorf("Expected score ~0.5 after one half-life, got %f", score)
	}
}

func TestCalculateScore_TwoHalfLives(t *testing.T) {
	halfLifeDays := 30.0
	createdAt := time.Now().Add(-time.Duration(2*halfLifeDays) * 24 * time.Hour)
	similarity := 1.0

	score := CalculateScore(similarity, createdAt, halfLifeDays)

	if score < 0.24 || score > 0.26 {
		t.Errorf("Expected score ~0.25 after two half-lives, got %f", score)
	}
}

func TestCalculateScore_DefaultHalfLife(t *testing.T) {
	createdAt := time.Now().Add(-30 * 24 * time.Hour)
	similarity := 1.0

	// A zero half-life falls back to the 30-day default.
	score := CalculateScore(similarity, createdAt, 0)

	if score < 0.49 || score > 0.51 {
		t.Errorf("Expected score ~0.5 with default half-life, got %f", score)
	}
}

func TestCalculateScore_NegativeHalfLife(t *testing.T) {
	createdAt := time.Now().Add(-30 * 24 * time.Hour)
	similarity := 1.0

	// So does a negative one.
	score := CalculateScore(similarity, createdAt, -10.0)

	if score < 0.49 || score > 0.51 {
		t.Errorf("Expected score ~0.5 with negative half-life, got %f", score)
	}
}

func TestCalculateScore_ZeroSimilarity(t *testing.T) {
	createdAt := time.Now()
	similarity := 0.0

	score := CalculateScore(similarity, createdAt, 30.0)

	if score != 0.0 {
		t.Errorf("Expected score 0.0 for zero similarity, got %f", score)
	}
}

func TestCalculateScore_OldFactLowerScore(t *testing.T) {
	oldFact := time.Now().Add(-60 * 24 * time.Hour)
	newFact := time.Now()
	similarity := 0.9
	halfLifeDays := 30.0

	oldScore := CalculateScore(similarity, oldFact, halfLifeDays)
	newScore := CalculateScore(similarity, newFact, halfLifeDays)

	if oldScore >= newScore {
		t.Errorf("Old fact should have lower score than new fact: old=%f, new=%f", oldScore, newScore)
	}
}
