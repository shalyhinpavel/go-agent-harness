# Roadmap

The goal is a small, reliable single-agent engine that Go programs can
embed, plus a ready-made CLI and HTTP server for those who just want to run
it.

## In place

- Neutral provider port (`llm`): messages, parts, tool calls on JSON
  Schema, streaming, structured output, usage on every call, failed ones
  included
- Providers: OpenAI and OpenAI-compatible servers, OpenRouter, Anthropic
  (including native replay of signed and redacted thinking blocks), and a
  scripted mock for tests
- Engine: tool loop with bounded parallel tools, forced final answer on the
  last iteration, sessions with TTL and LRU eviction, history window,
  preserved reasoning, context-limit guard with automatic or custom
  compaction, lifecycle hooks (`BeforeModel`, `BeforeTool`, `AfterTool`),
  human-in-the-loop tool approvals, stream retry before first output, and
  JSONL traces
- Tools: `web_fetch`, user- and session-scoped file tools with document
  parsers (`pdf`, `docx`, `xlsx`, `csv`, `json`), opt-in `bash`, MCP servers
  over stdio, streamable HTTP and SSE
- Session store as a port, with a JSONL file implementation
- HTTP: a plain `net/http` SSE handler with sessions, pre-stream admission
  checks, tool-call approvals (`POST /v1/chat/approve`), attachments,
  keep-alive frames, CORS allow-list and body limits
- Memory ports: system-prompt injection and a `memory_search` tool over a
  pluggable store
- `harness` command: terminal chat (with `-thinking` and `-approve`), the
  HTTP server, and `check` — the provider contract against a live backend
- Examples: minimal agent, custom tools, MCP, HTTP with authentication

## Not planned

- A built-in search tool: every good search API needs a key; bring yours
  as a tool or an MCP server
- A memory engine in the core: memory plugs in through its two ports
- Media output (image or audio generation)
