# Testing Capabilities

Harness tests should exercise the behavior users rely on, not only private
helpers.

## Default Shape

- Import `TestModel` from `pydantic_ai.models.test` for model behavior.
- Keep real provider calls out of tests.
- Prefer `Agent(..., capabilities=[...])` tests for public behavior.
- Mirror source packages under `tests/harness/<capability>/`.
- Use anyio's pytest plugin for async capability/toolset behavior.

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

## State Pydantic AI Produces

Don't hand-build state that Pydantic AI is supposed to produce: post-discovery
or post-load tool definitions, `ToolSearchReturnPart` contents, `RunContext`
fields such as `discovered_tool_names` or `loaded_capability_ids`, or
`ModelRequestParameters` reveal sets. A fixture like that encodes an
assumption about Pydantic AI's behavior, and it keeps passing after that
behavior changes, because the test never calls the code that changed.

- Prefer driving the real flow: an `Agent` with a `FunctionModel` that calls
  `search_tools` or `load_capability`, then assert on what the next request
  carries.
- Where the real flow is impractical, such as native tool-search parts that
  only a provider returns, keep the fixture and add a companion contract test
  that asserts Pydantic AI still produces or reads the state the fixture
  encodes, and reference that test from the fixture.

Hand-built state is fine for unit tests of harness-internal logic.

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

CI enforces 100% branch coverage after combining matrix artifacts. Do not run
coverage locally unless CI reports a specific coverage gap. To investigate a
reported gap, run coverage only for the flagged file or focused test.

Tests for a new capability should cover:

- default configuration
- important option combinations
- failure and retry paths
- composition with relevant Pydantic AI features
- docs examples when examples are executable

Use snapshots when behavior is protocol-shaped: messages, event streams,
schemas, telemetry spans, or structured tool metadata.

## Commands

Run Ruff across the repository. Run pytest and Pyright for the changed paths:

```bash
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync pytest -p no:cacheprovider tests/harness/<capability>
PYRIGHT_PYTHON_IGNORE_WARNINGS=1 uv run --no-sync pyright src/pydantic_ai_harness/pydantic_ai_harness/<capability>/ tests/harness/<capability>/
```

CI runs the repository-wide typecheck, test, and combined coverage gates.
