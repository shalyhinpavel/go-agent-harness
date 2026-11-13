# Security policy

## Supported versions

The harness is pre-1.0. Security fixes land on `main` and in the most recent
tagged release. Older `v0.x` tags are not patched.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Report it privately
through GitHub's security advisory form:

https://github.com/shalyhinpavel/go-agent-harness/security/advisories/new

Include what you found, the version or commit, and steps to reproduce. A
minimal test case helps most. We aim to acknowledge a report within seven days
and will keep you informed while a fix is prepared. Credit goes in the advisory
unless you prefer to stay anonymous.

## Scope

In scope: the library packages (`agent`, `llm`, `tools`, `mcpclient`,
`server`, `memory`, `trace`) and the `harness` command.

Out of scope:

- A deployment that exposes `harness serve` without authentication, which the
  command's usage text already warns about.
- Behaviour that needs an operator to enable something the defaults keep off,
  such as the `bash` tool.

## Deployment notes

- `harness serve` has no authentication. Keep it on loopback or behind a proxy
  that authenticates requests.
- The `bash` tool is opt-in (`tools.bash.enabled`) and is not a sandbox.
- `web_fetch` refuses private, loopback and link-local addresses when
  `tools.web_fetch.block_private` is set. The harness configuration sets it.
  When you embed the library, set `BlockPrivate: true` yourself: the zero value
  of `tools.WebFetchConfig` leaves it off.
