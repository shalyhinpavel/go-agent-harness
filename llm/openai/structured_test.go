package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// TestGenerateStructured_FormatFallback: a backend without structured-
// output support (400 on response_format) must still produce decoded JSON
// through the prompt-guided retry with lenient parsing.
func TestGenerateStructured_FormatFallback(t *testing.T) {
	var sawFormatFreeRequest bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)

		if _, has := body["response_format"]; has {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"message":"response_format json_schema is not supported by this model"}}`))
			return
		}
		sawFormatFreeRequest = true
		content := "Thought about it.\n```json\n{\"choice\":\"deep\",\"probabilities\":{\"deep\":0.9}}\n```"
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": content}}},
		})
	}))
	defer srv.Close()

	p, err := New(Config{APIKey: "test", BaseURL: srv.URL, Model: "weak-model"})
	if err != nil {
		t.Fatal(err)
	}

	var out struct {
		Choice        string             `json:"choice"`
		Probabilities map[string]float64 `json:"probabilities"`
	}
	if _, err := p.GenerateStructured(context.Background(), &llm.ChatRequest{
		Messages: []llm.Message{*llm.UserText("pick one")},
	}, llm.Schema{"type": "object"}, &out); err != nil {
		t.Fatalf("format fallback must decode: %v", err)
	}
	if out.Choice != "deep" {
		t.Fatalf("choice = %q", out.Choice)
	}
	if !sawFormatFreeRequest {
		t.Fatal("second attempt must go out without response_format")
	}
}

// TestGenerateStructured_NoJSONAnywhere: both attempts return prose with
// no JSON — the error must say so instead of silently succeeding.
func TestGenerateStructured_NoJSONAnywhere(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": "I cannot do JSON, sorry."}}},
		})
	}))
	defer srv.Close()

	p, err := New(Config{APIKey: "test", BaseURL: srv.URL, Model: "weak-model"})
	if err != nil {
		t.Fatal(err)
	}

	var out map[string]any
	_, err = p.GenerateStructured(context.Background(), &llm.ChatRequest{
		Messages: []llm.Message{*llm.UserText("pick one")},
	}, llm.Schema{"type": "object"}, &out)
	if err == nil || !strings.Contains(err.Error(), "no JSON found") {
		t.Fatalf("prose-only reply must fail with no JSON found, got: %v", err)
	}
}

// TestGenerateStructured_PromptRetriesAfterProse: a backend that ignores
// response_format answers in prose on the native attempt and on the first
// prompt-guided try; the second prompt-guided try is JSON and must decode.
func TestGenerateStructured_PromptRetriesAfterProse(t *testing.T) {
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		content := "Here are my thoughts in prose."
		if calls == 3 {
			content = `{"choice":"deep"}`
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": content}}},
		})
	}))
	defer srv.Close()

	p, err := New(Config{APIKey: "test", BaseURL: srv.URL, Model: "ignores-format"})
	if err != nil {
		t.Fatal(err)
	}
	var out struct {
		Choice string `json:"choice"`
	}
	if _, err := p.GenerateStructured(context.Background(), &llm.ChatRequest{
		Messages: []llm.Message{*llm.UserText("pick one")},
	}, llm.Schema{"type": "object"}, &out); err != nil {
		t.Fatalf("third attempt must decode: %v", err)
	}
	if out.Choice != "deep" || calls != 3 {
		t.Fatalf("choice = %q after %d calls, want deep after 3", out.Choice, calls)
	}
}

// TestGenerateStructured_TruncatedNativeFallsThrough: a native reply that
// starts a JSON object and stops (undecodable, not empty) must not end the
// call — the prompt-guided attempt still runs.
func TestGenerateStructured_TruncatedNativeFallsThrough(t *testing.T) {
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		content := `{"choice": "dee` // native attempt: an object cut off mid-string
		if _, native := body["response_format"]; !native {
			content = `{"choice":"deep"}`
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": content}}},
		})
	}))
	defer srv.Close()

	p, err := New(Config{APIKey: "test", BaseURL: srv.URL, Model: "truncates"})
	if err != nil {
		t.Fatal(err)
	}
	var out struct {
		Choice string `json:"choice"`
	}
	if _, err := p.GenerateStructured(context.Background(), &llm.ChatRequest{
		Messages: []llm.Message{*llm.UserText("pick one")},
	}, llm.Schema{"type": "object"}, &out); err != nil {
		t.Fatalf("truncated native reply must fall through to the prompt attempt: %v", err)
	}
	if out.Choice != "deep" || calls != 2 {
		t.Fatalf("choice = %q after %d calls, want deep after 2", out.Choice, calls)
	}
}
