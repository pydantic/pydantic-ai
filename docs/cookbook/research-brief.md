---
title: Research a decision with sources
description: Search, read, and delegate research while returning a validated, source-backed decision brief.
---

# Research a decision with sources

Use the Harness [`Researcher`](https://pydantic.dev/docs/ai/harness/researcher/) capability with a Pydantic output model when an application needs research it can store, render, or evaluate.

```bash
pip/uv-add "pydantic-ai-harness[researcher]"
```

```python {test="skip"}
from pydantic import BaseModel, HttpUrl
from pydantic_ai import Agent
from pydantic_ai_harness import Researcher


class Evidence(BaseModel):
    """A factual claim and the source that supports it."""

    claim: str
    source: HttpUrl


class ResearchBrief(BaseModel):
    """A source-backed answer to the research question."""

    answer: str
    evidence: list[Evidence]
    caveats: list[str]


agent = Agent(
    'openai:gpt-5.6-sol',
    output_type=ResearchBrief,
    capabilities=[Researcher()],
)

result = agent.run_sync(
    'Should a growing SaaS move its primary database from SQLite to PostgreSQL '
    'before it starts running multiple application instances? Use current official sources.'
)
print(result.output.model_dump_json(indent=2))
```

`Researcher` supplies web search, page fetching, focused delegation, source-backed research instructions, and bounded tool output. `ResearchBrief` makes the result a validated application value instead of prose that needs parsing.
