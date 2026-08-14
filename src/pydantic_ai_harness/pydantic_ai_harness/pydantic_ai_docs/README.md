# Pydantic AI Docs

Give an agent a tool that locates and returns Pydantic AI documentation on demand.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/pydantic_ai_docs/)

## The problem

An agent that authors Pydantic AI capabilities, hooks, tools, or toolsets needs the
current docs for those APIs. Preloading the docs into the system prompt spends context
the agent rarely needs in full and pins a snapshot that drifts from `main`.

## The solution

`PydanticAIDocs` exposes one tool, `read_pyai_docs(topic)`, that locates the requested page and
returns it verbatim -- nothing is bundled into context up front. Each call resolves the
topic from a configured local checkout first, then falls back to fetching the page from
`pydantic/pydantic-ai:main`, so it works whether or not you have a local checkout (the
remote fallback needs network access).

Topics: `capabilities`, `hooks`, `tools`, `tools-advanced`, `toolsets`, `agent`.

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai_harness import PydanticAIDocs

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[PydanticAIDocs(local_docs_path=Path('~/pydantic/ai/base/docs').expanduser())],
)
```

## Resolution order

1. **Local checkout** -- when `local_docs_path` (or the `PYDANTIC_AI_HARNESS_DOCS_PATH`
   env var) is set and `{path}/{topic}.md` exists, that file is read and returned.
2. **Remote fetch** -- otherwise the page is fetched from
   `https://raw.githubusercontent.com/pydantic/pydantic-ai/main/docs/{topic}.md`.
3. **Neither resolves** -- a descriptive error naming the local path tried and the URL.

The capability never runs git. Keep the local checkout current yourself; the remote path
always reads `main`, so it is the fresh fallback.

## Configuration

| Option | Default | Purpose |
| --- | --- | --- |
| `local_docs_path` | `None` | Local pyai docs checkout to read first. Falls back to the `PYDANTIC_AI_HARNESS_DOCS_PATH` env var, then to the remote source. |
| `cache` | `True` | Memoize each returned doc in-process for the capability's lifetime, so a topic is read or fetched at most once. |
