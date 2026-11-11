package memory

import (
	"math"
	"time"
)

// CalculateScore scores a fact against a query: cosine similarity decayed
// exponentially with the fact's age (half-life in days, default 30).
func CalculateScore(cosineSimilarity float64, createdAt time.Time, halfLifeDays float64) float64 {
	if halfLifeDays <= 0 {
		halfLifeDays = 30.0
	}

	ageDays := time.Since(createdAt).Hours() / 24.0
	decayFactor := math.Pow(0.5, ageDays/halfLifeDays)

	return cosineSimilarity * decayFactor
}
