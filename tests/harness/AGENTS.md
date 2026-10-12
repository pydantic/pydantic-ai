# `tests/harness/` guidelines

Rules for the `pydantic-ai-harness` test suite. They add to [`tests/AGENTS.md`](../AGENTS.md) and to the package
rules in [`src/pydantic_ai_harness/AGENTS.md`](../../src/pydantic_ai_harness/AGENTS.md). Where these rules or the
harness guides they link differ from `tests/AGENTS.md`, they apply under `tests/harness/`.

Read [`testing-capabilities.md`](../../src/pydantic_ai_harness/agent_docs/testing-capabilities.md) before you add or
change a capability's tests. For async behavior, follow the harness
[`concurrency.md`](../../src/pydantic_ai_harness/agent_docs/concurrency.md).

- `tests/harness/conftest.py` sets `ALLOW_MODEL_REQUESTS = False` for the suite.
- Name a capability test class `TestCapabilityName`, with `test_<scenario>` methods.
- Don't import private (`_`-prefixed) helpers into tests. Exercise them through
  the capability's public surface so tests survive internal refactors: drive the
  behavior through `Agent(..., capabilities=[...])`, or import the public class
  re-exported from the capability package's `__init__.py` (e.g.
  `from pydantic_ai_harness.filesystem import FileSystemToolset`, not
  `from pydantic_ai_harness.filesystem._toolset import _content_hash`). When a
  branch is only reachable by calling a private helper directly, mark it
  `# pragma: no cover` rather than reaching into the helper from a test.
