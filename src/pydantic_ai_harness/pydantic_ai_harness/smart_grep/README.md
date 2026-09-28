# Smart Grep

Semantic code search: the agent describes the behavior it is looking for in plain English and gets back ranked
source excerpts with file and line ranges, judged by any Pydantic AI model, with TypeSafe's Jev decision model
as the recommended judge.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/smart_grep/)

## Installation

uv:

```bash
uv add "pydantic-ai-harness[smart-grep]"
```

pip:

```bash
pip install "pydantic-ai-harness[smart-grep]"
```

The extra adds tree-sitter grammars for syntax-aware chunking of fourteen languages beyond Python; without it,
those files are cut into overlapping line windows. Add `pydantic-ai-slim[typesafe]` and set `TYPESAFE_API_KEY`
to judge with Jev.

## The problem

A coding agent in an unfamiliar repository spends its first turns locating code: guessing identifiers,
chaining exploratory `grep` calls, and reading whole files to find the one function that matters. Text search
only finds what you can already name.

## The solution

`SmartGrep` adds a `smart_grep` tool that lists the files in the run's workspace with `rg`, cuts them into
syntax-aware snippets, shortlists them with local BM25, and has a judge model rate each shortlisted snippet's
relevance to the query.

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import SmartGrep

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[LocalWorkspace('.'), SmartGrep(model='anthropic:claude-haiku-4-5')],
)
```

The judge is pluggable: pass any model or model name as `model`. Left unset, it is `'typesafe:jev-latest'`
when the `typesafe` SDK is installed and `TYPESAFE_API_KEY` is set, and the run's own model otherwise.

See the [Smart Grep docs](https://pydantic.dev/docs/ai/harness/smart-grep/) for the search pipeline, the result
shape, choosing the judge and tuning `threshold`.
