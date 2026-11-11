# go-agent-harness

A provider-agnostic agent engine for Go: a library (`agent`, `llm`, `tools`,
`mcpclient`, `server`, `memory`, `trace`) plus the `harness` command in
`cmd/harness` (chat, serve, check). README.md describes the API; ROADMAP.md
lists what is next.

## Commands

```bash
gofmt -l . && go build ./... && go vet ./... && go test -race ./...  # must be clean before a commit
go run ./cmd/harness                      # terminal chat; provider picked from the key in .env
go run ./cmd/harness "What is 2+2?"       # one live turn, then exit
go run ./cmd/harness serve                # HTTP SSE on 127.0.0.1:8080
go run ./cmd/harness check -model <id>    # provider contract against a live backend
```

Keys live in `.env` (see `.env.example`), never in git.

## Conventions

- The neutral port is `llm`: Message / Part / ToolCall on JSON Schema and
  `LLMProvider` (StreamChat, Generate, GenerateStructured). Vendor SDK types
  stay inside `llm/<provider>/`.
- Tests use `llm.NewMockProvider` and `httptest` fixtures: CI has no keys and
  no network. A bug fix ships with a test that fails without it.
- Errors are `fmt.Errorf` with the package prefix (`tools: ...`). Comments are
  English, explain why, and match the density of the surrounding code.
- Every provider method reports the usage it spent, failed attempts included.
- No search tool and no memory engine in the core: both plug in as tools,
  MCP servers or through the memory ports.
- `internal/` (config, prompts, bootstrap) serves `cmd/harness` only and is
  not a library API. Runtime data goes under the ignored `data/` directory.
- One logical change per commit, each commit green.

See CONTRIBUTING.md for adding tools and providers.
