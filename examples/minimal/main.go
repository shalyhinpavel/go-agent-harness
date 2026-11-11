// Minimal is the smallest useful agent: a model, one built-in tool, one
// question.
//
//	go run ./examples/minimal "What is on https://go.dev/doc/ ?"
package main

import (
	"context"
	"log"
	"os"
	"strings"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/examples/internal/demo"
	"github.com/shalyhinpavel/go-agent-harness/tools"
)

func main() {
	provider, err := demo.Provider()
	if err != nil {
		log.Fatal(err)
	}

	registry := agent.NewToolRegistry()
	if err := registry.Register(tools.NewWebFetch(tools.WebFetchConfig{BlockPrivate: true})); err != nil {
		log.Fatal(err)
	}

	engine := agent.NewAgentEngine(agent.AgentConfig{
		Provider:      provider,
		SystemPrompt:  "You are a helpful assistant. Use web_fetch to read pages you are asked about.",
		MaxIterations: 8,
	}, registry, agent.NewSessionManager())

	question := "Say hello in one sentence."
	if len(os.Args) > 1 {
		question = strings.Join(os.Args[1:], " ")
	}
	if err := demo.Say(context.Background(), engine, "session-1", question); err != nil {
		log.Fatal(err)
	}
}
