---
title: Share one usage budget across a conversation
description: Enforce an application-controlled request budget across multiple agent runs instead of resetting it each turn.
---

# Share one usage budget across a conversation

Pass the same `RunUsage` object to every run when a limit should cover a conversation, batch, or delegated workflow rather than one model response.

```bash
pip/uv-add "pydantic-ai-slim[openai]"
export OPENAI_API_KEY=your-api-key
```

```python {dunder_name="not_main"}
import asyncio

from pydantic_ai import Agent, RunUsage, UsageLimitExceeded, UsageLimits

agent = Agent('openai:gpt-5.6-sol', instructions='Answer incident questions concisely.')


async def main() -> None:
    usage = RunUsage()
    limits = UsageLimits(request_limit=2)

    for question in (
        'Summarize rollback readiness.',
        'Who should approve the rollback?',
        'Write the status update.',
    ):
        try:
            result = await agent.run(question, usage=usage, usage_limits=limits)
        except UsageLimitExceeded:
            print(f'Budget exhausted after {usage.requests} model requests.')
            #> Budget exhausted after 2 model requests.
            break
        else:
            print(result.output)
#> Rollback is ready once traffic-shift checks pass.
#> The incident commander should approve the rollback.


if __name__ == '__main__':
    asyncio.run(main())
```

Store `RunUsage` alongside persisted message history when the budget must survive process or request boundaries. Request and tool-call limits are deterministic; cost limits require models that report or can calculate cost.

## Related

See [Usage limits](../agent.md#usage-limits) for available limits and [Storage](../storage.md#what-a-history-alone-doesnt-carry) for persisting conversation-wide usage.
