# PyaiDocs

> [!WARNING]
> **Experimental.** This capability lives under `pydantic_ai_harness.experimental` and may
> change or be removed in any release, without a deprecation period. Import it from the
> experimental path -- there is no top-level export:
>
> ```python
> from pydantic_ai_harness.experimental.docs import PyaiDocs
> ```
>
> Importing any experimental capability emits a `HarnessExperimentalWarning`. Silence **all**
> harness experimental warnings with a single filter (no per-capability lines needed):
>
> ```python
> import warnings
> from pydantic_ai_harness.experimental import HarnessExperimentalWarning
>
> warnings.filterwarnings('ignore', category=HarnessExperimentalWarning)
> ```

Give an agent a tool that locates and returns Pydantic AI documentation on demand.

## The problem

An agent that authors Pydantic AI capabilities, hooks, tools, or toolsets needs the
current docs for those APIs. Preloading the docs into the system prompt spends context
the agent rarely needs in full and pins a snapshot that drifts from `main`.

## The solution

`PyaiDocs` exposes one tool, `read_pyai_docs(topic)`, that locates the requested page and
returns it verbatim -- nothing is bundled into context up front. Each call resolves the
topic from a configured local checkout first, then falls back to fetching the page from
`pydantic/pydantic-ai:main`, so it works in any environment.

Topics: `capabilities`, `hooks`, `tools`, `tools-advanced`, `toolsets`, `agent`.

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai_harness.experimental.docs import PyaiDocs

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[PyaiDocs(local_docs_path=Path('~/pydantic/ai/base/docs').expanduser())],
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
