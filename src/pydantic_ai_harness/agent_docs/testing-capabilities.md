# Testing Capabilities

Harness tests should exercise the behavior users rely on, not only private
helpers.

## Default Shape

- Use `pydantic_ai.models.TestModel` for model behavior.
- Keep real provider calls out of tests.
- Prefer `Agent(..., capabilities=[...])` tests for public behavior.
- Mirror source packages under `tests/<capability>/`.
- Use `pytest-anyio` for async capability/toolset behavior.

## Lower-Level Tests

Direct toolset tests are appropriate when you need to inspect:

- listed tools and schemas
- wrapper-toolset lifecycle
- retry behavior
- metadata and synthetic tool-call records
- `RunContext` or `ToolManager` interactions
- edge cases that are awkward to force through a full agent run

Use the `CodeMode` tests as the current reference for direct `RunContext` and
`ToolManager` setup.

## Boundary Reproductions

When a capability invokes another parser, process, service, or container, add
focused tests for each applicable contract:

- Reproduce the installed downstream parser's behavior when available, then
  test accepted abbreviations, aliases, `--name=value` forms, separators, and
  normalization through the capability's public entry point.
- Make cleanup return non-zero and raise. Assert that failure is surfaced and
  resource identity remains available until cleanup succeeds.
- Pass a non-default sentinel for each configurable address, endpoint, or path.
  Assert provisioning, readiness, invocation, and teardown all use it.
- Assert size limits against the final returned or serialized value, including
  headers, truncation markers, envelopes, and metadata.

## Coverage

The project enforces 100% branch coverage with `make testcov`. Tests for a new
capability should cover:

- default configuration
- important option combinations
- failure and retry paths
- composition with relevant Pydantic AI features
- docs examples when examples are executable

Use snapshots when behavior is protocol-shaped: messages, event streams,
schemas, telemetry spans, or structured tool metadata.

## Commands

Run focused checks first, then broaden:

```bash
uv run pytest tests/<capability>
make lint
make typecheck
make test
make testcov
```
