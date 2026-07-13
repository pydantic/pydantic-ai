---
title: Pydantic AI Docs
description: Give an agent a tool that locates and returns Pydantic AI documentation on demand instead of preloading it into the system prompt.
---

# Pydantic AI Docs

`PyaiDocs` gives an agent a single tool, `read_pyai_docs(topic)`, that locates a Pydantic AI documentation page and returns it verbatim. Nothing is bundled into context up front. Each call resolves the topic from a configured local checkout first, then falls back to fetching the page from `pydantic/pydantic-ai:main`, so it works whether or not you have a local checkout (the remote fallback needs network access).

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/docs/)

> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

## The problem

An agent that authors Pydantic AI capabilities, hooks, tools, or toolsets needs the current docs for those APIs. Preloading the docs into the system prompt spends context the agent rarely needs in full, and pins a snapshot that drifts from `main`.

## The solution

`PyaiDocs` exposes one tool, `read_pyai_docs(topic)`, that locates the requested page and returns it verbatim. Each call resolves the topic from a configured local checkout first, then falls back to fetching the page from `pydantic/pydantic-ai:main`, so it works whether or not you have a local checkout (the remote fallback needs network access).

The available topics are `capabilities`, `hooks`, `tools`, `tools-advanced`, `toolsets`, and `agent`.

## Usage

Construct an `Agent` with `PyaiDocs()` in its `capabilities`. Point `local_docs_path` at a local Pydantic AI docs checkout to read from disk first, or omit it to always fetch from the remote source:

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai_harness.docs import PyaiDocs

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[PyaiDocs(local_docs_path=Path('~/pydantic/ai/base/docs').expanduser())],
)

result = agent.run_sync('Read the toolsets docs, then explain how to build a FunctionToolset.')
print(result.output)
```

The capability also adds a short static instruction telling the model that the `read_pyai_docs` tool exists and to read the relevant topic before authoring or modifying a Pydantic AI capability, hook, tool, or toolset, rather than relying on memory. The instruction is cache-stable, so it does not invalidate the prompt-cache prefix between turns.

## Resolution order

Each call resolves in this order:

1. **Local checkout** -- when `local_docs_path` (or the `PYDANTIC_AI_HARNESS_DOCS_PATH` env var) is set and `{path}/{topic}.md` exists, that file is read and returned.
2. **Remote fetch** -- otherwise the page is fetched from `https://raw.githubusercontent.com/pydantic/pydantic-ai/main/docs/{topic}.md`.
3. **Neither resolves** -- a descriptive error naming the local path tried and the URL.

The capability never runs git. Keep the local checkout current yourself; the remote path always reads `main`, so it is the fresh fallback.

`local_docs_path` takes precedence over the `PYDANTIC_AI_HARNESS_DOCS_PATH` env var. Both have `~` expanded, so a raw `~/...` path resolves to the local checkout instead of silently falling through to the remote source. With neither set, every call goes straight to the remote source.

## Configuration

| Option | Default | Purpose |
| --- | --- | --- |
| `local_docs_path` | `None` | Local pyai docs checkout to read first. Falls back to the `PYDANTIC_AI_HARNESS_DOCS_PATH` env var, then to the remote source. |
| `cache` | `True` | Memoize each returned doc in-process for the capability's lifetime, so a topic is read or fetched at most once. |

Caching lives on the capability instance and is shared across the toolsets it builds, so a memoized topic survives multiple agent runs that reuse the same `PyaiDocs`. Set `cache=False` to re-read or re-fetch on every call -- useful when the local checkout changes underneath a long-lived capability.

## Agent spec (YAML/JSON)

`PyaiDocs` works with Pydantic AI's [agent spec](/ai/core-concepts/agent-spec/) feature for defining agents in YAML or JSON. Its serialization name is `PyaiDocs`:

```yaml
# agent.yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - PyaiDocs: {}
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.docs import PyaiDocs

agent = Agent.from_file('agent.yaml', custom_capability_types=[PyaiDocs])
result = agent.run_sync('...')
print(result.output)
```

Pass `custom_capability_types` so the spec loader knows how to instantiate `PyaiDocs`.

## API reference

::: pydantic_ai_harness.docs.PyaiDocs
