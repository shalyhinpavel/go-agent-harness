// Customtool shows how little a tool is: a name, a description written for
// the model, a JSON Schema and a function. It defines two:
//
//   - calculator: pure code, no dependencies;
//
//   - web_search: the plug point for the search API of your choice. The
//     harness ships no search (any good one needs a key); this one talks to a
//     SearXNG instance you run yourself, and the same shape fits Brave,
//     Tavily or any other HTTP API.
//
//     go run ./examples/customtool "What is 1234 times 5678?"
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/examples/internal/demo"
	"github.com/shalyhinpavel/go-agent-harness/llm"
)

func calculator() *agent.Tool {
	return &agent.Tool{
		Name:        "calculator",
		ReadOnly:    true, // changes nothing outside its result
		Description: "Does exact arithmetic on two numbers. Use it instead of computing in your head.",
		Parameters: llm.Schema{
			"type": "object",
			"properties": map[string]any{
				"a":  map[string]any{"type": "number"},
				"b":  map[string]any{"type": "number"},
				"op": map[string]any{"type": "string", "enum": []string{"add", "sub", "mul", "div"}},
			},
			"required": []string{"a", "b", "op"},
		},
		Execute: func(ctx context.Context, tc *agent.ToolContext, args map[string]any) (*agent.ToolResult, error) {
			a, okA := args["a"].(float64)
			b, okB := args["b"].(float64)
			op, _ := args["op"].(string)
			if !okA || !okB {
				return nil, fmt.Errorf("calculator: a and b must be numbers")
			}
			var result float64
			switch op {
			case "add":
				result = a + b
			case "sub":
				result = a - b
			case "mul":
				result = a * b
			case "div":
				if b == 0 {
					return nil, fmt.Errorf("calculator: division by zero")
				}
				result = a / b
			default:
				return nil, fmt.Errorf("calculator: unknown op %q", op)
			}
			// An error return becomes an error result the model can read and
			// react to; Data is what it sees on success.
			return &agent.ToolResult{Data: map[string]any{"result": result}}, nil
		},
	}
}

// webSearch queries a SearXNG instance (JSON output enabled in its settings).
func webSearch(baseURL string) *agent.Tool {
	client := &http.Client{Timeout: 15 * time.Second}
	return &agent.Tool{
		Name:        "web_search",
		ReadOnly:    true,
		Description: "Search the web. Returns titles, URLs and snippets; open a result with web_fetch.",
		Parameters: llm.Schema{
			"type":       "object",
			"properties": map[string]any{"query": map[string]any{"type": "string"}},
			"required":   []string{"query"},
		},
		Execute: func(ctx context.Context, tc *agent.ToolContext, args map[string]any) (*agent.ToolResult, error) {
			query, _ := args["query"].(string)
			if strings.TrimSpace(query) == "" {
				return nil, fmt.Errorf("web_search: query is required")
			}
			req, err := http.NewRequestWithContext(ctx, http.MethodGet,
				baseURL+"/search?format=json&q="+url.QueryEscape(query), nil)
			if err != nil {
				return nil, err
			}
			resp, err := client.Do(req)
			if err != nil {
				return nil, fmt.Errorf("web_search: %w", err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				return nil, fmt.Errorf("web_search: HTTP %d", resp.StatusCode)
			}
			var body struct {
				Results []struct {
					Title   string `json:"title"`
					URL     string `json:"url"`
					Content string `json:"content"`
				} `json:"results"`
			}
			if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
				return nil, fmt.Errorf("web_search: bad response: %w", err)
			}
			if len(body.Results) > 8 {
				body.Results = body.Results[:8]
			}
			return &agent.ToolResult{Data: body.Results}, nil
		},
	}
}

func main() {
	provider, err := demo.Provider()
	if err != nil {
		log.Fatal(err)
	}

	registry := agent.NewToolRegistry()
	must(registry.Register(calculator()))
	if searx := os.Getenv("SEARXNG_URL"); searx != "" {
		must(registry.Register(webSearch(strings.TrimRight(searx, "/"))))
	}

	engine := agent.NewAgentEngine(agent.AgentConfig{
		Provider:      provider,
		SystemPrompt:  "You are a careful assistant. Use the calculator for any arithmetic.",
		MaxIterations: 6,
	}, registry, agent.NewSessionManager())

	question := "What is 1234 times 5678?"
	if len(os.Args) > 1 {
		question = strings.Join(os.Args[1:], " ")
	}
	if err := demo.Say(context.Background(), engine, "session-1", question); err != nil {
		log.Fatal(err)
	}
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
