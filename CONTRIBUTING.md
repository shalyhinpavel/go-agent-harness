# Contributing

Thanks for considering a contribution. The harness is deliberately small
and provider-agnostic — keep it that way.

## Ground rules

- **No vendor types outside adapters.** `agent`, `tools`, `memory` and
  `trace` speak only the neutral types from `llm`. SDK imports belong in
  `llm/<provider>/` and nowhere else.
- **Memory stays out of core.** It plugs in through the `PromptAugmenter`
  port or a tool over `MemoryStore`. Do not add a memory engine to the
  agent loop.
- **Traces observe, they do not explain.** Record what happened (steps,
  tool I/O, usage, latency). Never reconstruct motives.
- **Honest failure beats clever recovery.** A subsystem that cannot
  deliver reports an error the caller can act on; it does not invent a
  plausible-looking answer.

## Code style

- Comments in English, in the surrounding density — explain why, not what.
- `gofmt` clean, `go vet` clean, tests green under the race detector:
  `gofmt -l . && go build ./... && go vet ./... && go test -race ./...`
- Tests run on the mock provider (`llm.NewMockProvider`) and `httptest`
  fixtures — no API keys, no network. A change without tests does not
  ship; a bug fix ships with a regression test that fails without it.

## Adding a tool

See "Adding a tool" in the README for the contract. Built-ins go in
`tools/`, get registered in `buildCatalogue`, and become
selectable via `tools.enabled`. Dangerous tools (like `bash`) additionally
require an explicit opt-in flag in the config. Mark a tool `ReadOnly`
only if it changes nothing outside its own result: approval policies treat
read-only tools as safe to run unattended.

## Adding a provider

Implement `llm.LLMProvider` (`StreamChat`, `Generate`, `GenerateStructured`)
in `llm/<name>/`. Convert to and from the neutral types at the
boundary. Wire it into `internal/bootstrap.Provider` so the config can select it. Structured generation should
survive backends without native JSON-schema support — see the two-step
format chain in `llm/openai/provider.go` (`GenerateStructured`) for the pattern. Every
method reports the usage it spent, failed attempts included. Then run
`go run ./cmd/harness check` against a live backend.

## Verification of behavior claims

If a change claims a behavioral property (a guard fires, a fallback
kicks in, usage is counted), the test has to exercise it — not assert
the code path by inspection. Live checks against a real backend are
great for catching what mocks cannot; commit the findings in the PR
description, not in the tests. A test written for a fix must fail on the
code before the fix — check it by reverting the fix, not by reasoning.

## Pull requests

Small, single-purpose, green on CI. Say what changed and why; if it
closes an item from ROADMAP.md, say so.
