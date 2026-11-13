# go-agent-harness

`go-agent-harness` is an embeddable, provider-agnostic AI agent engine for Go,
shipped with a ready-to-run CLI and HTTP/SSE server (`cmd/harness`).

## What it is and why it exists

Building a tool-using agent in Go usually forces an awkward choice:

- **Raw vendor SDKs** leave you rewriting the same loop in every project:
  streaming deltas, parallel tool execution, JSON Schema wiring, signed
  thinking replay, context-window guards, retries on dropped streams, and
  token accounting across failed calls and sub-agents.
- **Heavy agent frameworks** take over your application structure, bundle
  opinionated memory or search stacks, and get in the way when you just want
  to call a model from ordinary Go code.

`go-agent-harness` sits in between. It is a small library with one neutral
provider port (`llm`), plain Go functions as tools, per-session state,
lifecycle hooks, human-in-the-loop approvals, context compaction, and JSONL
trajectories. Vendor SDK types never leave `llm/<provider>/`. Search APIs,
authentication, and long-term memory engines stay outside the core and plug in
through clean ports or MCP servers.

Use it in two ways:

1. **Embed as a Go library** — import `agent`, `llm/*`, `tools`, `mcpclient`,
   `server`, `memory`, or `trace` directly in your service.
2. **Run `harness` from the command line** — interactive terminal chat,
   one-shot prompts, an HTTP + SSE server, and a live backend conformance
   checker.

Requires Go 1.26+. Pre-v1: minor releases may adjust APIs as they settle.

---

## Quick start (CLI)

Install the binary or run it from a clone:

```bash
go install github.com/shalyhinpavel/go-agent-harness/cmd/harness@latest
export ANTHROPIC_API_KEY=...    # or OPENAI_API_KEY, or OPENROUTER_API_KEY
harness
```

From a local clone (`go run ./cmd/harness`), keys are also read from `.env`
(see [`.env.example`](.env.example)) and runtime settings from
[`configs/config.yaml`](configs/config.yaml) when present.

```text
go-agent-harness · anthropic · claude-sonnet-5-5
tools: list_files, read_file, web_fetch, write_file (files in data/workspace)
/new starts a new conversation, /exit quits, Ctrl+C stops a reply

> What is on https://go.dev/doc/ ? Name three documents.
-> web_fetch {"url":"https://go.dev/doc/"}
ok web_fetch
Installing Go, Effective Go and A Tour of Go.
(4.8k in · 203 out · 4.6s)
```

Common CLI invocations:

```bash
harness                                       # interactive terminal chat
harness "Summarize https://go.dev/blog/"      # one turn, then exit
harness -model gpt-5 -thinking                # pick a model, stream its reasoning
harness -approve                              # ask [y/N] before non-read-only tools
OPENAI_BASE_URL=http://localhost:11434/v1 \
  AGENT_MODEL=qwen3 harness                   # local model (Ollama, vLLM, LM Studio)
harness serve                                 # HTTP + SSE server on 127.0.0.1:8080
harness check -model qwen/qwen3.8-27b         # run live provider conformance checks
```

Without any API key, `harness` falls back to a built-in mock provider and
prints a notice so you can inspect the CLI and HTTP wiring offline.

---

## Embed as a Go library

```bash
go get github.com/shalyhinpavel/go-agent-harness
```

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/llm/openai"
	"github.com/shalyhinpavel/go-agent-harness/tools"
)

func main() {
	provider, err := openai.New(openai.Config{
		APIKey:              os.Getenv("OPENAI_API_KEY"),
		Model:               "gpt-5",
		MaxCompletionTokens: true, // api.openai.com expects max_completion_tokens
		// BaseURL: "http://localhost:11434/v1", // any OpenAI-compatible server
	})
	if err != nil {
		log.Fatal(err)
	}

	registry := agent.NewToolRegistry()
	if err := registry.Register(tools.NewWebFetch(tools.WebFetchConfig{BlockPrivate: true})); err != nil {
		log.Fatal(err)
	}

	sessions := agent.NewSessionManager()
	engine := agent.NewAgentEngine(agent.AgentConfig{
		Provider:      provider,
		SystemPrompt:  "You are a concise assistant.",
		MaxIterations: 10,
	}, registry, sessions)

	stream, err := engine.StreamChat(context.Background(), agent.ChatRequest{
		UserID:    "alice",
		SessionID: "session-1", // reuse the same id to continue the conversation
		Message:   "What is on https://go.dev/doc/ ?",
	})
	if err != nil {
		log.Fatal(err)
	}

	for chunk := range stream {
		switch chunk.Type {
		case agent.ChunkTypeText:
			fmt.Print(chunk.Content)
		case agent.ChunkTypeToolCall:
			fmt.Printf("\n[tool %s %v]\n", chunk.ToolName, chunk.ToolArgs)
		case agent.ChunkTypeError:
			log.Fatal(chunk.Error)
		case agent.ChunkTypeDone:
			if u := chunk.Usage; u != nil {
				fmt.Printf("\n(%d in, %d out)\n", u.InputTokens, u.OutputTokens)
			}
		}
	}
}
```

If you only want the final text instead of streaming chunks, call
`text, err := engine.Chat(ctx, req)`.

To switch to Anthropic, swap the provider construction and keep everything
else identical:

```go
provider, err := anthropic.New(anthropic.Config{
	APIKey: os.Getenv("ANTHROPIC_API_KEY"),
	Model:  "claude-sonnet-5-5",
})
```

For unit tests, `llm.NewMockProvider` scripts deterministic turns (text,
thinking, tool calls, usage, errors) with no network and no keys.

---

## Writing tools and connecting MCP servers

### Native Go tools

A tool is an `*agent.Tool` with a JSON Schema and a handler:

```go
err := registry.Register(&agent.Tool{
	Name:        "lookup_order",
	Description: "Look up an order by ID and return its current shipment status.",
	ReadOnly:    true, // safe to run without human approval
	Parameters: llm.Schema{
		"type": "object",
		"properties": map[string]any{
			"order_id": map[string]any{"type": "string"},
		},
		"required": []string{"order_id"},
	},
	Execute: func(ctx context.Context, tc *agent.ToolContext, args map[string]any) (*agent.ToolResult, error) {
		id, _ := args["order_id"].(string)
		tenant, _ := tc.CustomData["tenant"].(string)
		status, err := fetchOrder(ctx, tenant, tc.UserID, id)
		if err != nil {
			// Returned errors become tool error results that the model can read and recover from.
			return nil, fmt.Errorf("lookup_order: %w", err)
		}
		return &agent.ToolResult{Data: status}, nil
	},
})
```

Key details:

- **Parallelism:** when the model requests multiple tools in one turn, the
  engine runs them concurrently up to `MaxParallelTools` (default `8`; set `1`
  for sequential execution) and returns results in call order.
- **Request-scoped context:** set `ChatRequest.CustomData` from your auth
  layer (tenant ID, DB transaction handle, permissions). Tools receive a
  shallow copy in `ToolContext.CustomData` (treat it as read-only).
- **Sub-agents:** a tool can run another `AgentEngine` inside `Execute`, route
  events to `tc.Trace` (labeled via `AgentConfig.Name`), and return its token
  spend in `ToolResult.Usage` so the parent turn's usage includes nested calls.
- **Panic recovery:** panics inside tool handlers or document parsers are
  caught and converted into error tool results rather than crashing the host
  process.

### Built-in tools (`package tools`)

| Tool | ReadOnly | Behavior |
|---|---|---|
| `web_fetch` | `true` | Fetches a URL, converts HTML into readable text with links preserved, and pages output via `offset`. `BlockPrivate: true` blocks loopback, RFC1918, link-local, CGNAT, and multicast addresses (anti-SSRF). |
| `read_file` | `true` | Reads a file from the user+session workspace (`<root>/<user>/<session>`), protected against symlink escape via `os.Root`. Parses `.pdf`, `.docx`, `.xlsx`, `.csv`, and `.json` into plain text. |
| `list_files` | `true` | Lists files and subdirectories inside the user+session workspace. |
| `write_file` | `false` | Creates or overwrites a text file inside the user+session workspace. |
| `bash` | `false` | Runs `bash -c` inside the workspace with a scrubbed environment (no `.env` keys leaked), capped output, process-group kill on exit, and a per-command timeout. **Not an OS sandbox** — run inside a container if the model or prompt input is untrusted. |

### Model Context Protocol (`package mcpclient`)

Attach external MCP servers over `stdio`, `streamable-http`, or `sse`:

```go
pool, err := mcpclient.ConnectAll(ctx, []mcpclient.ServerConfig{
	{
		Name:    "fs",
		Command: "npx",
		Args:    []string{"-y", "@modelcontextprotocol/server-filesystem", "/tmp"},
	},
	{
		Name:       "docs",
		URL:        "http://127.0.0.1:8000/mcp",
		ToolPrefix: "docs_",
		ReadOnly:   []string{"search", "get_page"},
	},
}, registry)
if err != nil {
	log.Fatal(err)
}
defer pool.Close()
```

`stdio` subprocesses receive a minimal environment (`PATH`, `HOME`, `TMPDIR`,
locale, TLS cert vars) plus explicit `ServerConfig.Env` — never the host
process's API keys unless `InheritEnv: true` is set.

---

## Hooks and human approvals

### Lifecycle hooks

`AgentConfig.Hooks` lets host code inspect, mutate, deny, or redact model and
tool traffic without wrapping providers or tools:

```go
Hooks: agent.Hooks{
	BeforeModel: func(ctx context.Context, hc agent.HookContext, req *llm.ChatRequest) error {
		// Inspect or adjust req.System, req.Messages, req.Tools before each LLM call.
		// Returning a non-nil error aborts the turn with ChunkTypeError.
		return nil
	},
	BeforeTool: func(ctx context.Context, hc agent.HookContext, call llm.ToolCall) (agent.ToolDecision, error) {
		if call.Name == "write_file" && call.Args["path"] == "config.yaml" {
			// Soft deny: tool is skipped, model receives the reason and continues.
			return agent.ToolDecision{Deny: true, Reason: "config.yaml is read-only"}, nil
		}
		// Return a non-nil error instead to abort the entire turn immediately.
		return agent.ToolDecision{}, nil
	},
	AfterTool: func(ctx context.Context, hc agent.HookContext, call llm.ToolCall, res *agent.ToolResult, elapsed time.Duration) (*agent.ToolResult, error) {
		// Redact secrets or enrich results before truncation, tracing, and streaming.
		return res, nil
	},
}
```

### Human-in-the-loop approvals

Set `AgentConfig.Approver` (or per-turn `ChatRequest.Approver`) to pause
non-read-only tool calls until a human confirms them:

```go
Approver: func(ctx context.Context, req agent.ApprovalRequest) (agent.ApprovalDecision, error) {
	ok := askUser(ctx, req.ToolName, req.ToolArgs)
	if !ok {
		return agent.ApprovalDecision{Approved: false, Reason: "rejected by operator"}, nil
	}
	return agent.ApprovalDecision{Approved: true}, nil
}
```

- By default, tools with `ReadOnly: true` run unattended and tools with
  `ReadOnly: false` ask the `Approver`. Customize that rule with
  `AgentConfig.ApprovalPolicy`.
- Before calling `Approver`, the stream emits a chunk with
  `Type: agent.ChunkTypeApprovalRequired` carrying `tool_call_id`, `tool_name`,
  and `tool_args`.
- `Approved: false` sends an error `tool_result` back to the model so it can
  adjust course; returning a non-nil `error` aborts the turn.

---

## Sessions, context guards, and reasoning

- **Session isolation and persistence.** `agent.NewSessionManager` serializes
  turns per session, enforces user ownership (`ErrSessionOwner`), and manages
  in-memory TTL (`WithSessionTTL`) and LRU bounds (`WithMaxSessions`). Attach
  `agent.WithSessionStore(store)` — `agent.NewFileSessionStore(dir)` stores
  one JSONL file per session — so conversations and session ownership survive
  evictions and restarts.
- **History windowing.** `HistoryWindow` (message count) and `HistoryChars`
  (payload budget) trim older messages on clean tool-exchange boundaries.
  `HistoryTrimTo` (default `0.75`) trims in steps rather than on every turn so
  prompt prefixes remain cacheable. `PinTask: true` keeps the first user
  message pinned at the front of the window.
- **Context-limit guard and compaction.** When `ContextLimit` is set, the
  engine projects the next call's size from the last reported token usage and
  stops before overflowing the model window. With `AutoCompact: true` (or a
  custom `AgentConfig.Compactor`), older messages are summarized into a compact
  prefix and their token spend is added to the turn total.
- **Reasoning preservation.** With `PreserveReasoning: true`, OpenAI-compatible
  chains of thought (`reasoning_details`) and Anthropic signed `thinking` /
  `redacted_thinking` blocks are replayed across turns, and stripped
  automatically whenever compaction or a changed system/tool definition would
  invalidate signatures. Models that emit reasoning with no visible answer are
  nudged once to write the final answer.
- **Resilience and loop control.** Streams interrupted before emitting any
  output are retried (`StreamRetries`, default `2`). On `MaxIterations`, the
  final call forces `tool_choice: none` so the model answers instead of
  stopping mid-loop. `ChatRequest.Proceed` lets callers halt long loops cleanly
  between iterations when a budget expires.

---

## HTTP + SSE server (`package server`)

`server.NewHandler` returns a standard `http.Handler` with no framework
dependencies:

```go
handler := server.NewHandler(engine, server.Options{
	FilesDir:        "data/workspace",
	CORSOrigins:     []string{"https://app.example"},
	RequireApproval: true,
	ApprovalTimeout: 5 * time.Minute,
	Identify: func(r *http.Request) (server.Identity, error) {
		userID, tenant, err := verifyBearerToken(r.Header.Get("Authorization"))
		if err != nil {
			return server.Identity{}, err
		}
		return server.Identity{
			UserID:     userID,
			CustomData: map[string]any{"tenant": tenant},
		}, nil
	},
})
http.ListenAndServe("127.0.0.1:8080", handler)
```

### Endpoints

| Method & Path | Purpose |
|---|---|
| `GET /health` | Liveness check (`200 {"status":"ok"}`). |
| `POST /v1/chat/stream` | Runs a chat turn and streams `agent.StreamChunk` JSON objects as Server-Sent Events (`data: {...}\n\n`). Sends `: keepalive` comments while idle. |
| `POST /v1/chat/approve` | Approves or denies a pending tool call (`{"session_id","tool_call_id","approved","reason"}`). |

### Protocol details

- **Request body (`POST /v1/chat/stream`):**
  `{"user_id", "session_id", "message", "attachments": [{"name", "media_type", "data"}]}`.
  When `session_id` is omitted, a fresh UUID is generated and returned in the
  `X-Session-ID` response header.
- **Chunk sequence:** `text`, `thinking`, `tool_call`, `approval_required`
  (when `RequireApproval` is on), `tool_result`, ending with exactly one `done`
  or `error` chunk carrying token `usage`.
- **Pre-stream status codes:** `400` invalid input, `401` failed `Identify`,
  `403` session belongs to another user, `409` session already has a turn in
  progress, `413` body exceeds `MaxBodyMB` — all checked before storing
  attachments or opening the SSE stream.
- **Attachments:** base64-encoded. `image/*` attachments are sent inline to
  vision-capable models; other files are written to the user's session
  workspace so the model can inspect them with `read_file`.
- **Security:** `server.NewHandler` includes no built-in auth scheme. Keep it
  on `127.0.0.1` for local use, or wire `Options.Identify` / an authenticating
  reverse proxy before exposing it on a network.

---

## Observability and memory ports

- **Token accounting:** every `done` and `error` chunk reports the turn's
  `llm.Usage`, including failed/interrupted provider streams, context
  compaction calls, and sub-agent `ToolResult.Usage`.
- **JSONL traces (`package trace`):** pass `trace.NewDirSink(dir)` as
  `AgentConfig.TraceSink` to record one `<session>.jsonl` file per session with
  turn boundaries, text/thinking deltas, tool arguments and full results, per-call
  latencies, and optionally full final-answer prompts (`TraceRequests: true`).
- **Memory (`package memory`):** memory engines stay outside the core and
  attach via two ports:
  1. `AgentConfig.PromptAugmenter` — injects user dossiers or retrieved context
     into the system prompt per turn.
  2. `memory.NewSearchTool(store, embedder)` — exposes `memory_search` over any
     `memory.MemoryStore` implementation (in-memory and human-editable Markdown
     stores are included for local setups and tests).

---

## Examples

All examples run out of the box on the scripted mock provider, or against a
live backend when `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is set:

| Example | Demonstrates |
|---|---|
| [`examples/minimal`](examples/minimal/main.go) | Minimal library setup: one provider, `web_fetch`, one turn |
| [`examples/customtool`](examples/customtool/main.go) | Custom Go tools: a calculator and a SearXNG search tool |
| [`examples/mcp`](examples/mcp/main.go) | Connecting an external MCP server's tools to the engine |
| [`examples/http`](examples/http/main.go) | Mounting `server.NewHandler` behind bearer-token authentication |

```bash
go run ./examples/minimal "What is on https://go.dev/doc/ ?"
go run ./examples/mcp npx -y @modelcontextprotocol/server-filesystem /tmp
```

---

## Repository layout

```text
agent/        engine loop, sessions, session store, hooks, approvals, compaction
llm/          neutral message/schema types, LLMProvider port, MockProvider
  openai/     OpenAI and OpenAI-compatible adapter
  anthropic/  Anthropic Messages API adapter
tools/        built-in tools: web_fetch, workspace file tools + parsers, bash
mcpclient/    MCP client over stdio, streamable HTTP, and SSE
server/       net/http SSE handler and approval endpoint
memory/       MemoryStore port, in-memory & markdown file stores, memory_search
trace/        JSONL trajectory recorder with bounded open-file pool
examples/     runnable library examples
cmd/harness/  CLI: chat, serve, check
internal/     YAML config, default prompts, CLI bootstrap (not public API)
```

---

## Development

```bash
gofmt -l . && go build ./... && go vet ./... && go test -race ./...
```

All unit and integration tests run offline against `llm.NewMockProvider` and
`httptest` servers. To validate a live model backend end-to-end, run
`go run ./cmd/harness check -model <id>`.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [ROADMAP.md](ROADMAP.md).

## License

[MIT](LICENSE)
