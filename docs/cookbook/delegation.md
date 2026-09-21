---
title: Delegate focused work to specialist agents
description: Keep specialist contexts and tools separate while one parent agent owns the final answer.
---

# Delegate focused work to specialist agents

Use Harness `SubAgents` when a parent should assign self-contained tasks without sharing its entire conversation with each specialist.

```python {test="skip"}
import asyncio

from pydantic_ai_harness import SubAgent, SubAgents

from pydantic_ai import Agent

security = Agent(
    'anthropic:claude-sonnet-4-6',
    name='security-reviewer',
    description='Reviews a proposed change for concrete security risks',
)
reliability = Agent(
    'anthropic:claude-sonnet-4-6',
    name='reliability-reviewer',
    description='Reviews failure modes, rollback, and operational risk',
)

reviewer = Agent(
    'anthropic:claude-opus-4-7',
    instructions='Delegate both specialist reviews, then resolve disagreements in the final recommendation.',
    capabilities=[SubAgents(agents=[SubAgent(security), SubAgent(reliability)])],
)


async def main() -> None:
    result = await reviewer.run('Review a plan to move session storage from Redis to PostgreSQL.')
    print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
```

Each delegate receives only the task the parent writes for it. Dependencies and usage limits can be forwarded, while per-delegate timeouts and call limits prevent one specialist from consuming the whole run budget.
