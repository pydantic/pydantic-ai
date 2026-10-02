# `tests/harness/` guidelines

Rules for the `pydantic-ai-harness` test suite. They add to [`tests/AGENTS.md`](../AGENTS.md) and to the package
rules in [`src/pydantic_ai_harness/AGENTS.md`](../../src/pydantic_ai_harness/AGENTS.md).

Read [`testing-capabilities.md`](../../src/pydantic_ai_harness/agent_docs/testing-capabilities.md) before you add or
change a capability's tests. Put them in `tests/harness/<capability>/`, named after the capability's package in
`src/pydantic_ai_harness/pydantic_ai_harness/`.

## Testing patterns

- Import `TestModel` from `pydantic_ai.models.test` for model behavior.
- Keep real provider calls out of tests.
- `ALLOW_MODEL_REQUESTS = False` is set globally in `tests/harness/conftest.py`
- Tests use `pytest-anyio` for async support
- Each capability test class follows: `TestCapabilityName` with methods `test_<scenario>`
- Prefer tests through `Agent(..., capabilities=[...])` when that is the public
  behavior. Use direct `Toolset`/`RunContext` tests for lower-level lifecycle,
  schema, retry, or wrapper behavior that is hard to isolate through `Agent`.
- Don't import private (`_`-prefixed) helpers into tests. Exercise them through
  the capability's public surface so tests survive internal refactors: drive the
  behavior through `Agent(..., capabilities=[...])`, or import the public class
  re-exported from the capability package's `__init__.py` (e.g.
  `from pydantic_ai_harness.filesystem import FileSystemToolset`, not
  `from pydantic_ai_harness.filesystem._toolset import _content_hash`). When a
  branch is only reachable by calling a private helper directly, mark it
  `# pragma: no cover` rather than reaching into the helper from a test.
- Test async behavior directly, following
  [`src/pydantic_ai_harness/agent_docs/concurrency.md`](../../src/pydantic_ai_harness/agent_docs/concurrency.md): assert
  the cancellation, the ordering, or the cleanup itself rather than the output
  it happens to produce; order steps with `Event`s instead of sleeps; and reach
  the real trigger (a real outer `anyio` cancel scope, the Trio parametrization
  where the suite has it) rather than a stand-in.
