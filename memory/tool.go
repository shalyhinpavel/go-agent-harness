package memory

import (
	"context"
	"fmt"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// NewSearchTool builds the memory_search tool over a store — the
// tool-based memory access port. The model calls it when the conversation
// references long-term context that is not in the current window.
//
// embeddingFn converts a query into the store's vector space; nil falls
// back to keyword storage order (the built-in InMemoryStore scores by
// recency and ignores the vector).
func NewSearchTool(store MemoryStore, embeddingFn func(ctx context.Context, query string) ([]float32, error)) *agent.Tool {
	return &agent.Tool{
		Name:        "memory_search",
		ReadOnly:    true,
		Description: "Search the user's long-term memory for facts relevant to a query (earlier preferences, projects, constraints).",
		Parameters: llm.Schema{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{
					"type":        "string",
					"description": "What to look for in long-term memory",
				},
				"limit": map[string]any{
					"type":        "number",
					"description": "Maximum facts to return (default 5)",
				},
			},
			"required": []string{"query"},
		},
		Execute: func(ctx context.Context, tc *agent.ToolContext, args map[string]any) (*agent.ToolResult, error) {
			query, _ := args["query"].(string)
			limit := 5
			if l, ok := args["limit"].(float64); ok && l > 0 {
				limit = int(l)
			}

			var embedding []float32
			if embeddingFn != nil {
				vec, err := embeddingFn(ctx, query)
				if err != nil {
					return nil, fmt.Errorf("memory: embed query: %w", err)
				}
				embedding = vec
			}

			facts, err := store.SearchFacts(ctx, tc.UserID, query, embedding, limit)
			if err != nil {
				return nil, fmt.Errorf("memory: search: %w", err)
			}

			results := make([]map[string]any, 0, len(facts))
			for _, f := range facts {
				results = append(results, map[string]any{
					"category":   f.Category,
					"fact":       f.Fact,
					"confidence": f.Confidence,
					"score":      f.Score,
				})
			}
			return &agent.ToolResult{Data: map[string]any{
				"query": query,
				"facts": results,
				"count": len(results),
			}}, nil
		},
	}
}
