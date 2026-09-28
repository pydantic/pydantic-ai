# Pydantic AI Docs

Give an agent a tool that locates and returns Pydantic AI documentation on demand.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/pydantic_ai_docs/)

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

## Resolution order

Each call resolves in this order:

1. **Workspace checkout** -- when `local_docs_path` (or the `PYDANTIC_AI_HARNESS_DOCS_PATH`
   environment variable) is set and `{path}/{topic}.md` exists inside the run workspace, that file is read and returned.
2. **Remote fetch** -- otherwise the page is fetched from
   `https://raw.githubusercontent.com/pydantic/pydantic-ai/main/docs/{topic}.md`.
3. **Neither resolves** -- a descriptive error naming the local path tried and the URL.

The capability never runs git. Keep the local checkout current yourself; the remote path
always reads `main`, so it is the fresh fallback.

`local_docs_path` takes precedence over the `PYDANTIC_AI_HARNESS_DOCS_PATH` environment variable.
`~` is not expanded -- use an absolute workspace path or a path relative to its working
directory. With neither path set, every call goes straight to the remote source.

## Configuration

| Option | Default | Purpose |
| --- | --- | --- |
| `local_docs_path` | `None` | Pyai docs checkout inside the run workspace. Relative paths use the workspace working directory. Falls back to the `PYDANTIC_AI_HARNESS_DOCS_PATH` environment variable, then to the remote source. |
| `cache` | `True` | Memoize each returned doc for one agent run, so repeated reads within that run do not repeat workspace or network I/O. |

Caching is isolated per run so content read from one workspace is not reused in another. Set
`cache=False` to re-read or re-fetch on every call within a run.
