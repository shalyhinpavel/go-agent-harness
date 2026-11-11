// Package demo holds the two helpers every example shares: picking a model
// provider from the environment and printing a turn as it streams. It is
// not part of the harness API — in your own program you build the provider
// you want directly.
package demo

import (
	"context"
	"fmt"
	"os"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/llm"
	"github.com/shalyhinpavel/go-agent-harness/llm/anthropic"
	"github.com/shalyhinpavel/go-agent-harness/llm/openai"
)

// Provider picks a provider from the environment:
//
//	ANTHROPIC_API_KEY                  Anthropic (MODEL, default claude-sonnet-5-5)
//	OPENAI_API_KEY [+ OPENAI_BASE_URL] OpenAI or any OpenAI-compatible server
//	                                   (MODEL is required for a custom base URL)
//
// With neither set it returns a scripted mock, so every example runs
// without keys — it just answers with a fixed line.
func Provider() (llm.LLMProvider, error) {
	model := os.Getenv("MODEL")
	switch {
	case os.Getenv("ANTHROPIC_API_KEY") != "":
		if model == "" {
			model = "claude-sonnet-5-5"
		}
		return anthropic.New(anthropic.Config{Model: model})
	case os.Getenv("OPENAI_API_KEY") != "":
		base := os.Getenv("OPENAI_BASE_URL")
		if model == "" {
			if base != "" {
				return nil, fmt.Errorf("demo: set MODEL for the endpoint %s", base)
			}
			model = "gpt-5"
		}
		return openai.New(openai.Config{
			APIKey: os.Getenv("OPENAI_API_KEY"), BaseURL: base, Model: model,
			MaxCompletionTokens: base == "",
		})
	default:
		fmt.Fprintln(os.Stderr, "(no ANTHROPIC_API_KEY or OPENAI_API_KEY: using the mock provider)")
		return llm.NewMockProvider(llm.MockTurn{Text: "This is the mock provider. Set a key to talk to a real model."}), nil
	}
}

// Say runs one chat turn and prints what the agent does as it happens:
// text as it streams, every tool call and its outcome, and the token bill.
func Say(ctx context.Context, engine *agent.AgentEngine, sessionID, message string) error {
	stream, err := engine.StreamChat(ctx, agent.ChatRequest{UserID: "demo", SessionID: sessionID, Message: message})
	if err != nil {
		return err
	}
	for chunk := range stream {
		switch chunk.Type {
		case agent.ChunkTypeText:
			fmt.Print(chunk.Content)
		case agent.ChunkTypeToolCall:
			fmt.Printf("\n[tool call] %s %v\n", chunk.ToolName, chunk.ToolArgs)
		case agent.ChunkTypeToolResult:
			if chunk.ToolResult.Error != "" {
				fmt.Printf("[tool error] %s\n", chunk.ToolResult.Error)
			} else {
				fmt.Printf("[tool result] %s ok\n", chunk.ToolName)
			}
		case agent.ChunkTypeError:
			return fmt.Errorf("turn failed: %s", chunk.Error)
		case agent.ChunkTypeDone:
			if u := chunk.Usage; u != nil {
				fmt.Printf("\n\n(tokens: %d in, %d out)\n", u.InputTokens, u.OutputTokens)
			}
		}
	}
	return nil
}
