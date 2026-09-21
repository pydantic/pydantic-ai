---
title: Evaluate agent behavior in CI
description: Run a small regression dataset against an agent and fail when required behavior disappears.
---

# Evaluate agent behavior in CI

Use Pydantic Evals to express behavior as cases and evaluators rather than asserting an exact model sentence.

```bash
pip/uv-add "pydantic-ai-slim[openai]" pydantic-evals
export OPENAI_API_KEY=your-api-key
```

```python {dunder_name="not_main"}
import asyncio

from pydantic_ai import Agent
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Contains

agent = Agent('openai:gpt-5.6-sol', instructions='Answer support questions accurately and concisely.')

dataset = Dataset(
    name='support-regressions',
    cases=[
        Case(
            name='lost-api-key',
            inputs='I lost an API key. What should I do?',
            evaluators=[Contains(value='revoke', case_sensitive=False)],
        ),
        Case(
            name='suspected-leak',
            inputs='A secret may be in our logs. What is the first action?',
            evaluators=[Contains(value='rotate', case_sensitive=False)],
        ),
    ],
)


async def answer(question: str) -> str:
    return (await agent.run(question)).output


async def main() -> None:
    report = await dataset.evaluate(answer)
    averages = report.averages()
    assert averages and averages.assertions == 1
    print('All required behaviors passed.')
    #> All required behaviors passed.


if __name__ == '__main__':
    asyncio.run(main())
```

Keep deterministic unit tests for wiring and use a small behavioral dataset for outcomes that require a model. Evaluate required concepts or structured fields rather than exact prose, and record the model and date used by CI.

## Related

See [Evals](../evals.md) for evaluators, custom evaluation logic, and experiment reporting.
