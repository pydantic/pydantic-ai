# Pydantic Harness

## Local setup

On first session, if `CLAUDE.local.md` does not exist, create it with the following content:

```markdown
This file is checked into the repository but ignored, any changes made stay local.

Use this file only to persist information about the specific workstation you are used on.

Capabilities branch is checked out at: <absolute path to the capabilities branch checkout on this system>
```

Fill in the absolute path where the `capabilities` branch of `pydantic-ai` is checked out on your system. If unknown, ask the user.

## Vocabulary

- **Capability**: an `AbstractCapability` subclass that bundles tools, hooks, instructions, and model settings into a reusable unit. This is the core abstraction of pydantic-harness
- **Hook**: a lifecycle method on `AbstractCapability` that intercepts agent graph execution (e.g. `before_model_request`, `wrap_run`, `after_tool_execute`)
- **Toolset**: a collection of tools that a capability can provide to the agent
- **Guard**: a type of capability that validates inputs/outputs or controls tool access (e.g. `InputGuardrail`, `CostGuard`)
- **Harness**: this package -- a collection of pre-made capabilities for pydantic-ai
- **AICA**: AI Code Assistant -- the automated agent that implements issues, reviews plans, and handles PR feedback
- **Ralph loop**: the state-machine-based workflow that drives AICA through phases (TRIAGE -> GOALS -> PLAN -> CODE -> VERIFY -> REVIEW -> PUBLISH)
- **DDD+ protocol**: classification system for PR review comments (do, dismiss, discuss, waiting, done)

## Capabilities API reference

When implementing a new capability, reference these docs in the pydantic-ai repo (path in `CLAUDE.local.md`):

- `docs/capabilities.md` -- main capabilities documentation, usage patterns, built-in capabilities
- `docs/hooks.md` -- lifecycle hooks reference, hook ordering, all hook categories
- `docs/extensibility.md` -- publishing capabilities as packages, spec serialization
- `docs/toolsets.md` -- toolset abstraction, building tools for capabilities
- `docs/tools-advanced.md` -- tool hooks, prepare_tools, tool validation
- `docs/agent.md` -- agent configuration, instructions, model settings
- `pydantic_ai_slim/pydantic_ai/capabilities/abstract.py` -- the `AbstractCapability` base class (all hook methods)
- `pydantic_ai_slim/pydantic_ai/capabilities/hooks.py` -- decorator-based `Hooks` capability
- `pydantic_ai_slim/pydantic_ai/capabilities/combined.py` -- `CombinedCapability` for composition

## Coding standards

- Python 3.10+ (target version for pyright and ruff)
- **pyright strict** mode -- no `Any` types, full type annotations
- **ruff**: line-length=120, single quotes, max-complexity=15
- **100% branch coverage** required (enforced by `make testcov`)
- docstrings use single backticks (markdown), not RST double backticks
- no typecasting (`as` in TypeScript, `cast()` in Python) -- use type narrowing instead
- prefer the most generic input types possible (reduce dependency chains)
- don't add comments that restate what the code does

## Package management

- Use `uv` for all dependency operations
- Never edit `pyproject.toml` or `uv.lock` directly -- use `uv add`, `uv remove`
- External PRs that change dependencies are auto-closed by CI

## Commands

```bash
make format     # ruff format
make lint       # ruff check
make typecheck  # pyright strict
make test       # pytest
make testcov    # pytest with branch coverage
```

Always run `make lint && make typecheck && make test` before committing.

## File structure

```
pydantic_harness/
  __init__.py          # public API re-exports
  <capability>/        # each capability gets its own package
    __init__.py        # public exports for the capability
    _capability.py     # capability class (AbstractCapability subclass)
    _toolset.py        # toolset implementation
    README.md          # standalone docs for the capability
tests/
  conftest.py          # shared fixtures (TestModel, test_agent)
  _<capability>/       # tests mirror source packages
    test_<capability>.py
```

## Testing patterns

- Use `pydantic_ai.models.TestModel` for all tests (no real API calls)
- `ALLOW_MODEL_REQUESTS = False` is set globally in `conftest.py`
- Tests use `pytest-anyio` for async support
- Each capability test class follows: `TestCapabilityName` with methods `test_<scenario>`

## Contributing rules for AICAs

- Never change `pyproject.toml` or `uv.lock` -- if a dependency is needed, open an issue
- Always link sources for any claims made during research
- Run `make lint && make typecheck && make test` before every commit
- Commit messages should summarize the "why", not the "what"
- All GitHub comments must start with "Claude here: "
