# Smart File Search

Plain-English code search: the agent describes the behavior it is looking for in plain English and gets back ranked
source excerpts with file and line ranges, judged by any Pydantic AI model, with a decision model served over the
System One API, or TypeSafe's Jev, as the recommended judge.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/smart_file_search/)

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

The judge is pluggable: pass any model or model name as `model`. A decision model over the System One API, such
as `'system-one:clm-latest'`, needs only `SYSTEM_ONE_BASE_URL` (and `SYSTEM_ONE_API_KEY` if the API needs a key).
Left unset, the judge is `'typesafe:jev-latest'` when the `typesafe` SDK is installed and `TYPESAFE_API_KEY` is
set, and the run's own model otherwise.

See the [Smart File Search docs](https://pydantic.dev/docs/ai/harness/smart-file-search/) for the search
pipeline, the result shape, choosing the judge and tuning `threshold`.
