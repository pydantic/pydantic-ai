---
title: Put a hard limit on tool execution
description: Stop an agent before a model-generated batch exceeds the allowed number of tool calls.
---

# Put a hard limit on tool execution

Pass `UsageLimits` to each run when model requests, tokens, cost, or tool execution must stay inside an application-controlled budget.

```python {dunder_name="not_main"}
import asyncio

from pydantic_ai import Agent, UsageLimitExceeded, UsageLimits

agent = Agent('openai:gpt-5.6-sol')
sent_emails: list[str] = []


@agent.tool_plain
def send_email(address: str, subject: str) -> str:
    """Send one email."""
    sent_emails.append(address)
    return 'sent'


async def main() -> None:
    try:
        await agent.run(
            'Email the launch update to alice@example.com and bob@example.com.',
            usage_limits=UsageLimits(request_limit=3, tool_calls_limit=1),
        )
    except UsageLimitExceeded:
        print('Stopped before exceeding the tool-call limit.')
        #> Stopped before exceeding the tool-call limit.

    print(sent_emails)
    #> []


if __name__ == '__main__':
    asyncio.run(main())
```

The model requests two calls in one batch, but the limit allows only one. Pydantic AI rejects the entire batch before either tool executes, avoiding a misleading partial result. Choose limits at the application boundary rather than relying on the prompt to control spending or side effects.
