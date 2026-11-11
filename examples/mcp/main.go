// Mcp connects the agent to a Model Context Protocol server and lets it use
// that server's tools like its own. Any MCP server works — files, GitHub,
// databases, a browser, a search API. This example starts one over stdio:
//
//	go run ./examples/mcp npx -y @modelcontextprotocol/server-filesystem /tmp
//
// (everything after the program name is the server's command line).
package main

import (
	"context"
	"log"
	"os"
	"time"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/examples/internal/demo"
	"github.com/shalyhinpavel/go-agent-harness/mcpclient"
)

func main() {
	if len(os.Args) < 2 {
		log.Fatal("usage: go run ./examples/mcp <server command> [args...]")
	}
	provider, err := demo.Provider()
	if err != nil {
		log.Fatal(err)
	}

	registry := agent.NewToolRegistry()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	pool, err := mcpclient.ConnectAll(ctx, []mcpclient.ServerConfig{{
		Name:    "demo",
		Command: os.Args[1],
		Args:    os.Args[2:],
	}}, registry)
	cancel()
	if err != nil {
		log.Fatal(err)
	}
	defer pool.Close()

	for _, t := range registry.List() {
		log.Printf("tool from the server: %s", t.Name)
	}

	engine := agent.NewAgentEngine(agent.AgentConfig{
		Provider:      provider,
		SystemPrompt:  "You are a helpful assistant. Use the available tools to answer.",
		MaxIterations: 10,
		// An environment whose calls depend on order wants one call at a time.
		MaxParallelTools: 1,
	}, registry, agent.NewSessionManager())

	if err := demo.Say(context.Background(), engine, "session-1", "List what you can do with your tools, then do one harmless thing."); err != nil {
		log.Fatal(err)
	}
}
