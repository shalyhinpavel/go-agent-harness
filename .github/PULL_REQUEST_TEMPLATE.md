## What and why

<!-- One logical change. Link the issue if there is one. -->

## Checklist

- [ ] `gofmt -l .` prints nothing, and `go build ./...`, `go vet ./...` and `go test -race -count=1 ./...` pass.
- [ ] A bug fix comes with a test that fails without the fix.
- [ ] Vendor SDK types stay inside `llm/<provider>/` (see CONTRIBUTING.md).
- [ ] README or CONTRIBUTING.md is updated if behaviour or the public API changed.
