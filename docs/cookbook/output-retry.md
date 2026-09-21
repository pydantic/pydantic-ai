---
title: Retry output that fails a business rule
description: Validate a typed result and give the model one focused chance to correct it.
---

# Retry output that fails a business rule

Use an output validator when valid JSON is not enough and the result must also satisfy an application rule.

```python {dunder_name="not_main"}
import asyncio

from pydantic import BaseModel

from pydantic_ai import Agent, ModelRetry, RunContext


class IncidentPlan(BaseModel):
    summary: str
    action_items: list[str]


agent = Agent(
    'openai:gpt-5.6-sol',
    output_type=IncidentPlan,
    retries=1,
)


@agent.output_validator
def require_action(ctx: RunContext[None], output: IncidentPlan) -> IncidentPlan:
    if not output.action_items:
        raise ModelRetry('Include at least one concrete action item.')
    return output


async def main() -> None:
    result = await agent.run(
        'Write a response plan for elevated API latency after a deployment.'
    )
    print(result.output.action_items)
    #> ['Roll back the deployment and compare latency with the previous release.']


if __name__ == '__main__':
    asyncio.run(main())
```

Pydantic validates the structure first. The output validator then enforces the business rule and sends its focused correction back to the model. Keep `retries` bounded so a persistently bad response fails clearly instead of looping.
