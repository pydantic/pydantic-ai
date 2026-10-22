# Smart Search

Plain-English code search: the agent describes the behavior it is looking for in plain English and gets back ranked
source excerpts with file and line ranges, judged by any Pydantic AI model, with a decision model served over the
System One API, or TypeSafe's Jev, as the recommended judge.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/smart_search/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## The problem

A coding agent in an unfamiliar repository spends its first turns locating code: guessing identifiers,
chaining exploratory `grep` calls, and reading whole files to find the one function that matters. Text search
only finds what you can already name.

## The solution

`SmartSearch` adds a `smart_search` tool that lists the files in the run's workspace with `rg`, cuts them
into snippets (overlapping line windows, or the functions and methods of a Python file), shortlists them with
local BM25, and has a judge model rate each shortlisted snippet's relevance to the query.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import SmartSearch

agent = Agent(
    'openai:gpt-6-luna',
    capabilities=[LocalWorkspace('.'), SmartSearch(model='system-one:clm-latest')],
)
```

The judge is pluggable: pass any model or model name as `model`. A decision model over the System One API, such
as `'system-one:clm-latest'`, needs only `SYSTEM_ONE_BASE_URL` (and `SYSTEM_ONE_API_KEY` if the API needs a key).
Left unset, the judge is `'typesafe:jev-latest'` when the `typesafe` SDK is installed and `TYPESAFE_API_KEY` is
set, and the run's own model otherwise.

A search sends the query, the shortlisted snippets and their paths to the judge model's provider, and only searches
inside the workspace's working directory. If no source may leave the machine, judge with a local model. With
instrumentation on, each search is a `smart search` span with the judge model and coverage figures as
attributes; the judge runs themselves are not traced, as their prompts carry source.

See the [Smart Search docs](https://pydantic.dev/docs/ai/harness/smart-search/) for the search
pipeline, the result shape, choosing the judge and tuning `threshold`.
