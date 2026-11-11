package bootstrap

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/shalyhinpavel/go-agent-harness/internal/config"
	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// TestProvider_ParamsReachTheWire: YAML params travel through config and
// bootstrap into the request body, nested extra_body included.
func TestProvider_ParamsReachTheWire(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer srv.Close()

	path := filepath.Join(t.TempDir(), "config.yaml")
	yaml := "provider:\n  kind: openai-compat\n  model: m\n  base_url: " + srv.URL + `
  params:
    top_k: 20
    reasoning_effort: low
    extra_body:
      chat_template_kwargs:
        enable_thinking: true
`
	if err := os.WriteFile(path, []byte(yaml), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	p, err := Provider(cfg.Provider)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := p.Generate(context.Background(), &llm.ChatRequest{Messages: []llm.Message{*llm.UserText("q")}}); err != nil {
		t.Fatal(err)
	}
	kw, _ := body["chat_template_kwargs"].(map[string]any)
	if body["top_k"] != 20.0 || body["reasoning_effort"] != "low" || kw["enable_thinking"] != true {
		t.Fatalf("params did not reach the wire: %+v", body)
	}
}

// TestProvider_AnthropicRefusesParams: a kind that cannot apply the params
// must say so instead of running under settings the config does not describe.
func TestProvider_AnthropicRefusesParams(t *testing.T) {
	_, err := Provider(config.Provider{Kind: "anthropic", Params: config.ProviderParams{ReasoningEffort: "high"}})
	if err == nil {
		t.Fatal("anthropic with params must fail")
	}
	if _, err := Provider(config.Provider{Kind: "anthropic"}); err != nil {
		t.Fatalf("anthropic without params: %v", err)
	}
	temp := 0.2
	if _, err := Provider(config.Provider{Kind: "anthropic", Params: config.ProviderParams{Temperature: &temp}}); err != nil {
		t.Fatalf("anthropic takes temperature: %v", err)
	}
}

func TestResolve_AutoPicksByKey(t *testing.T) {
	keys := []string{"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENROUTER_API_KEY"}
	cases := []struct {
		name      string
		env       map[string]string
		kind      string
		model     string
		wantKind  string
		wantModel string
	}{
		{"no key", nil, "auto", "", "mock", ""},
		{"anthropic first", map[string]string{"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "o"}, "auto", "", "anthropic", "claude-sonnet-5-5"},
		{"openai", map[string]string{"OPENAI_API_KEY": "o", "OPENROUTER_API_KEY": "r"}, "auto", "", "openai", "gpt-5"},
		{"local server", map[string]string{"OPENAI_BASE_URL": "http://localhost:11434/v1"}, "auto", "qwen3", "openai-compat", "qwen3"},
		{"openrouter", map[string]string{"OPENROUTER_API_KEY": "r"}, "auto", "", "openrouter", "openrouter/auto"},
		{"explicit kind wins", map[string]string{"ANTHROPIC_API_KEY": "a"}, "openrouter", "m", "openrouter", "m"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			for _, k := range keys {
				t.Setenv(k, c.env[k])
			}
			got := Resolve(config.Provider{Kind: c.kind, Model: c.model})
			if got.Kind != c.wantKind || got.Model != c.wantModel {
				t.Fatalf("Resolve = %s/%q, want %s/%q", got.Kind, got.Model, c.wantKind, c.wantModel)
			}
		})
	}
}

// TestProvider_OpenAIHonoursBaseURL: an explicit openai kind reaches the base
// URL it was given, from the config or from OPENAI_BASE_URL, so a proxy in
// front of the API is not bypassed.
func TestProvider_OpenAIHonoursBaseURL(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "test")
	check := func(t *testing.T, cfg config.Provider, hit <-chan struct{}) {
		t.Helper()
		p, err := Provider(cfg)
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if _, _, err := p.Generate(ctx, &llm.ChatRequest{Messages: []llm.Message{*llm.UserText("q")}}); err != nil {
			t.Fatalf("the request must reach the configured server: %v", err)
		}
		select {
		case <-hit:
		default:
			t.Fatal("the configured server saw no request")
		}
	}
	serve := func(t *testing.T) (*httptest.Server, <-chan struct{}) {
		hit := make(chan struct{}, 1)
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			select {
			case hit <- struct{}{}:
			default:
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
		}))
		t.Cleanup(srv.Close)
		return srv, hit
	}

	t.Run("config base_url", func(t *testing.T) {
		srv, hit := serve(t)
		check(t, config.Provider{Kind: "openai", Model: "m", BaseURL: srv.URL}, hit)
	})
	t.Run("OPENAI_BASE_URL", func(t *testing.T) {
		srv, hit := serve(t)
		t.Setenv("OPENAI_BASE_URL", srv.URL)
		check(t, config.Provider{Kind: "openai", Model: "m"}, hit)
	})
}

func TestProvider_CompatNeedsModel(t *testing.T) {
	if _, err := Provider(config.Provider{Kind: "openai-compat", BaseURL: "http://localhost:1/v1"}); err == nil {
		t.Fatal("an OpenAI-compatible server without a model must fail at startup, not on the first request")
	}
}
