---
title: Pydantic AI Docs
description: Give an agent a tool that locates and returns Pydantic AI documentation on demand instead of preloading it into the system prompt.
---

# Pydantic AI Docs

`PydanticAIDocs` gives an agent a single tool, `read_pyai_docs(topic)`, that locates a Pydantic AI documentation page and returns it verbatim. Nothing is bundled into context up front. Each call resolves the topic from a configured checkout inside the run workspace first, then falls back to fetching the page from `pydantic/pydantic-ai:main` (the remote fallback needs network access).

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/pydantic_ai_docs/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## The problem

An agent that authors Pydantic AI capabilities, hooks, tools, or toolsets needs the current docs for those APIs. Preloading the docs into the system prompt spends context the agent rarely needs in full, and pins a snapshot that drifts from `main`.

## The solution

`PydanticAIDocs` exposes one tool, `read_pyai_docs(topic)`, that locates the requested page and returns it verbatim. Each call resolves the topic from a configured checkout inside the run workspace first, then falls back to fetching the page from `pydantic/pydantic-ai:main` (the remote fallback needs network access).

The available topics are `capabilities`, `hooks`, `tools`, `tools-advanced`, `toolsets`, and `agent`.

## Usage

Construct an `Agent` with `PydanticAIDocs()` in its `capabilities`. Point `local_docs_path` at a Pydantic AI docs checkout inside the run workspace, or omit it to always fetch from the remote source. Relative paths use the workspace working directory:

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import PydanticAIDocs

agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[LocalWorkspace('.'), PydanticAIDocs(local_docs_path=Path('docs'))],
)

result = agent.run_sync('Read the toolsets docs, then explain how to build a FunctionToolset.')
print(result.output)
```

A configured local checkout is read through the run's [workspace](https://pydantic.dev/docs/ai/core-concepts/workspace/), so a run with a local path and no workspace fails at its start. With no local path configured, every call goes to the remote source and no workspace is needed.

The capability also adds a short static instruction telling the model that the `read_pyai_docs` tool exists and to read the relevant topic before authoring or modifying a Pydantic AI capability, hook, tool, or toolset, rather than relying on memory. The instruction is cache-stable, so it does not invalidate the prompt-cache prefix between turns.

## Resolution order

Each call resolves in this order:

1. **Workspace checkout** -- when `local_docs_path` (or the `PYDANTIC_AI_HARNESS_DOCS_PATH` environment variable) is set and the topic's page exists under it inside the run workspace (`{path}/{topic}.md`, or `{path}/capabilities/overview.md` for `capabilities`), that file is read and returned.
2. **Remote fetch** -- otherwise the page is fetched from the same path under `https://raw.githubusercontent.com/pydantic/pydantic-ai/main/docs/`.
3. **Neither resolves** -- a descriptive error naming the local path tried and the URL.

The capability never runs git. Keep the local checkout current yourself; the remote path always reads `main`, so it is the fresh fallback.

`local_docs_path` takes precedence over the `PYDANTIC_AI_HARNESS_DOCS_PATH` environment variable. `~` is not expanded -- use an absolute workspace path or a path relative to its working directory. With neither path set, every call goes straight to the remote source.

## Configuration

| Option | Default | Purpose |
| --- | --- | --- |
| `local_docs_path` | `None` | Pyai docs checkout inside the run workspace. Relative paths use the workspace working directory. Falls back to the `PYDANTIC_AI_HARNESS_DOCS_PATH` environment variable, then to the remote source. |
| `cache` | `True` | Memoize each returned doc for one agent run, so repeated reads within that run do not repeat workspace or network I/O. |

Caching is isolated per run so content read from one workspace is not reused in another. Set `cache=False` to re-read or re-fetch on every call within a run.

## Agent spec (YAML/JSON)

`PydanticAIDocs` works with Pydantic AI's [agent spec](../agent-spec.md) feature for defining agents in YAML or JSON. Its serialization name is `PydanticAIDocs`:

```yaml
# agent.yaml
model: anthropic:claude-sonnet-5
capabilities:
  - PydanticAIDocs: {}
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import PydanticAIDocs

agent = Agent.from_file('agent.yaml', custom_capability_types=[PydanticAIDocs])
result = agent.run_sync('...')
print(result.output)
```

Pass `custom_capability_types` so the spec loader knows how to instantiate `PydanticAIDocs`.

Specs saved before the rename from `PyaiDocs` use the old block name. To keep loading
them, pass the deprecated `PyaiDocs` class (imported from `pydantic_ai_harness.docs`,
which emits a deprecation warning) alongside or instead of `PydanticAIDocs` -- it keeps
the `PyaiDocs` serialization name. Re-save with `PydanticAIDocs` to migrate.

## API reference

::: pydantic_ai_harness.pydantic_ai_docs.PydanticAIDocs
