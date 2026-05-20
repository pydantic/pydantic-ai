# Harness Agent Docs

Use these guides when building or reviewing `pydantic-ai-harness` changes.

## Required Reads

For any code change:

1. `AGENTS.md`
2. This file
3. The guide below that matches the task
4. The public Pydantic AI docs for the integration points you touch

## Task Routing

- New or changed capability API: `capability-authoring.md`
- New or changed tests: `testing-capabilities.md`
- Unsure whether behavior belongs in harness or Pydantic AI core: `core-boundary.md`
- Review, pre-PR check, or final self-check: `review-checklist.md`

## Exemplar

Use `pydantic_ai_harness.code_mode` as the current exemplar for capability
shape:

- public re-export from `pydantic_ai_harness/__init__.py`
- package-level re-export from `pydantic_ai_harness/code_mode/__init__.py`
- public capability class in `_capability.py`
- implementation toolset in `_toolset.py`
- capability README next to implementation
- mirrored tests under `tests/code_mode/`

Do not copy `CodeMode` mechanically. Use it to understand package shape,
testing depth, docs placement, and how a harness capability composes with
Pydantic AI toolsets.

## Pydantic AI References

- Capabilities: <https://pydantic.dev/docs/ai/core-concepts/capabilities/>
- Hooks: <https://pydantic.dev/docs/ai/core-concepts/hooks/>
- Toolsets: <https://pydantic.dev/docs/ai/tools-toolsets/toolsets/>
- Advanced tools: <https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/>
- Agents: <https://pydantic.dev/docs/ai/core-concepts/agent/>
- Testing: <https://pydantic.dev/docs/ai/guides/testing/>
- Extensibility: <https://pydantic.dev/docs/ai/guides/extensibility/>
