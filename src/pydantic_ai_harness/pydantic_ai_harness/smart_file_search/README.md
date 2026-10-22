# Smart File Search

Plain-English code search: the agent describes the behavior it is looking for in plain English and gets back ranked
source excerpts with file and line ranges, judged by any Pydantic AI model, with a decision model served over the
System One API, or TypeSafe's Jev, as the recommended judge.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/smart_file_search/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## The problem

A coding agent in an unfamiliar repository spends its first turns locating code: guessing identifiers,
chaining exploratory `grep` calls, and reading whole files to find the one function that matters. Text search
only finds what you can already name.

## The solution

`SmartFileSearch` adds a `smart_file_search` tool that lists the files in the run's workspace with `rg`, cuts them
into snippets (overlapping line windows, or the functions and methods of a Python file), shortlists them with
local BM25, and has a judge model rate each shortlisted snippet's relevance to the query.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import SmartFileSearch

agent = Agent(
    'openai:gpt-6-luna',
    capabilities=[LocalWorkspace('.'), SmartFileSearch(model='system-one:clm-latest')],
)
```

The judge is required and pluggable: pass any model or model name as `model`. A decision model over the System One
API, such as `'system-one:clm-latest'`, needs only `SYSTEM_ONE_BASE_URL` (and `SYSTEM_ONE_API_KEY` if the API needs
a key); TypeSafe's Jev is `'typesafe:jev-latest'`. Each judged snippet counts as one request against the run's
`request_limit` (50 by default), and a search spends at most half of what is left, so raise it for full
128-snippet searches.

A search sends the query, the shortlisted snippets and their paths to the judge model's provider, and only searches
inside the workspace's working directory. If no source may leave the machine, judge with a local model.

## Telemetry

Each search opens a `smart file search` span under its tool call, on the run's tracer:

| Attribute | Value |
|---|---|
| `smart_file_search.judge_model` | the judge model |
| `smart_file_search.files`, `.skipped_files`, `.snippets`, `.evaluated` | the coverage figures |
| `smart_file_search.selection_complete` | whether every snippet was judged |
| `smart_file_search.indexed_files` | files the search chunked and indexed; with `cache_index`, only new and changed ones |
| `smart_file_search.matches`, `.omitted_matches` | matches returned, and matches cut by `limit` |
| `smart_file_search.query` | the query, only when the run includes content in traces |

A failed search records no exception on this span; the enclosing tool span reports it, redacted by the run's
`include_content`. The judge runs themselves are not instrumented, as their prompts carry source, but their usage
is added to the run's.

See the [Smart File Search docs](https://pydantic.dev/docs/ai/harness/smart-file-search/) for the search
pipeline, the result shape, choosing the judge and tuning `threshold`.
