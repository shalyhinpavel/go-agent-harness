// Package bootstrap wires providers, prompts, tools and engines from a
// config.Config for the harness command.
package bootstrap

import (
	"fmt"
	"os"

	"github.com/shalyhinpavel/go-agent-harness/internal/config"
	"github.com/shalyhinpavel/go-agent-harness/llm"
	"github.com/shalyhinpavel/go-agent-harness/llm/anthropic"
	"github.com/shalyhinpavel/go-agent-harness/llm/openai"
)

// Default models for a provider section that names none.
var defaultModels = map[string]string{
	"anthropic":  "claude-sonnet-5-5",
	"openai":     "gpt-5",
	"openrouter": "openrouter/auto",
}

// NoKeyMessage is what the mock says when auto found no API key: the first
// reply a new user sees tells them how to get a real one.
const NoKeyMessage = "No API key found, so this is the built-in mock talking. " +
	"Set ANTHROPIC_API_KEY, OPENAI_API_KEY or OPENROUTER_API_KEY (in the environment or a .env file) " +
	"to chat with a real model; for a local server set OPENAI_BASE_URL and AGENT_MODEL."

// Resolve settles what Provider will build: kind "auto" picks the backend
// by the first API key found in the environment (Anthropic, then OpenAI or
// an OpenAI-compatible server at OPENAI_BASE_URL, then OpenRouter; none
// means the mock), and an empty model takes the kind's default. Commands
// call it to show the user what they are talking to.
func Resolve(cfg config.Provider) config.Provider {
	if cfg.Kind == "auto" {
		switch {
		case os.Getenv("ANTHROPIC_API_KEY") != "":
			cfg.Kind = "anthropic"
		case os.Getenv("OPENAI_BASE_URL") != "" || cfg.BaseURL != "":
			cfg.Kind = "openai-compat"
		case os.Getenv("OPENAI_API_KEY") != "":
			cfg.Kind = "openai"
		case os.Getenv("OPENROUTER_API_KEY") != "":
			cfg.Kind = "openrouter"
		default:
			cfg.Kind = "mock"
		}
	}
	if cfg.Model == "" {
		cfg.Model = defaultModels[cfg.Kind]
	}
	return cfg
}

// Provider builds the LLM provider described by cfg (after Resolve). API
// keys resolve via the config's api_key_env, falling back to the
// conventional variable of each provider kind.
func Provider(cfg config.Provider) (llm.LLMProvider, error) {
	cfg = Resolve(cfg)
	kind, model := cfg.Kind, cfg.Model
	key := func(fallback string) string {
		if cfg.APIKeyEnv != "" {
			if v := os.Getenv(cfg.APIKeyEnv); v != "" {
				return v
			}
		}
		if fallback != "" {
			return os.Getenv(fallback)
		}
		return ""
	}

	base := func(fallbackEnv string) string {
		if cfg.BaseURL != "" {
			return cfg.BaseURL
		}
		if cfg.BaseURLEnv != "" {
			return os.Getenv(cfg.BaseURLEnv)
		}
		if fallbackEnv != "" {
			return os.Getenv(fallbackEnv)
		}
		return ""
	}

	sampling := openai.Sampling{
		Temperature:     cfg.Params.Temperature,
		TopP:            cfg.Params.TopP,
		TopK:            cfg.Params.TopK,
		ReasoningEffort: cfg.Params.ReasoningEffort,
		ExtraBody:       cfg.Params.ExtraBody,
	}

	switch kind {
	case "mock":
		return llm.NewMockProvider(llm.MockTurn{Text: NoKeyMessage}), nil
	case "openai":
		// api.openai.com unless a base URL says otherwise (a proxy in front of
		// it): reasoning models reject max_tokens either way.
		return openai.New(openai.Config{
			APIKey:              key("OPENAI_API_KEY"),
			BaseURL:             base("OPENAI_BASE_URL"),
			Model:               model,
			MaxCompletionTokens: true,
			Sampling:            sampling,
		})
	case "openai-compat":
		if model == "" {
			return nil, fmt.Errorf("provider.model (or AGENT_MODEL) is required for an OpenAI-compatible server")
		}
		oc := openai.Config{
			APIKey:   key("OPENAI_API_KEY"),
			Model:    model,
			Sampling: sampling,
		}
		if u := base("OPENAI_BASE_URL"); u != "" {
			oc.BaseURL = u
		}
		return openai.New(oc)
	case "openrouter":
		return openai.New(openai.Config{
			APIKey:   key("OPENROUTER_API_KEY"),
			BaseURL:  "https://openrouter.ai/api/v1",
			Model:    model,
			Sampling: sampling,
		})
	case "anthropic":
		// Temperature and top_p are supported; silently dropping the other
		// knobs would run the model under settings the config does not
		// describe.
		rest := cfg.Params
		rest.Temperature, rest.TopP = nil, nil
		if !rest.IsZero() {
			return nil, fmt.Errorf("provider.params: only temperature and top_p are supported for the anthropic kind")
		}
		return anthropic.New(anthropic.Config{
			APIKey:      key("ANTHROPIC_API_KEY"),
			Model:       model,
			BaseURL:     base("ANTHROPIC_BASE_URL"),
			Temperature: cfg.Params.Temperature,
			TopP:        cfg.Params.TopP,
		})
	default:
		return nil, fmt.Errorf("unknown provider kind %q", kind)
	}
}
