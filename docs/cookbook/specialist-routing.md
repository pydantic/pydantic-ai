---
title: Route a request to a specialist agent
description: Classify a support request, hand the same conversation to a typed specialist, and enforce one shared usage budget.
---

# Route a request to a specialist agent

Use application-controlled routing when each specialist needs its own instructions and output type, but your application must retain control of the hand-off and total model usage.

```bash
pip/uv-add "pydantic-ai-slim[openai]"
export OPENAI_API_KEY=your-api-key
```

```python {dunder_name="not_main"}
import asyncio
from typing import Literal

from pydantic import BaseModel

from pydantic_ai import Agent, RunUsage, UsageLimits


class Route(BaseModel):
    destination: Literal['billing', 'technical', 'general']
    reason: str


class BillingReply(BaseModel):
    answer: str
    needs_refund_review: bool


class TechnicalReply(BaseModel):
    answer: str
    diagnostic_code: str | None


class GeneralReply(BaseModel):
    answer: str


router = Agent(
    'openai:gpt-5.6-mini',
    output_type=Route,
    instructions='Route the request. Do not answer it.',
)
billing = Agent(
    'openai:gpt-5.6',
    output_type=BillingReply,
    instructions='Resolve billing requests. Flag refunds for review; never promise one.',
)
technical = Agent(
    'openai:gpt-5.6',
    output_type=TechnicalReply,
    instructions='Give one safe diagnostic step and include a known diagnostic code when relevant.',
)
general = Agent(
    'openai:gpt-5.6-mini',
    output_type=GeneralReply,
    instructions='Answer support requests that need no specialist.',
)

SupportReply = BillingReply | TechnicalReply | GeneralReply
limits = UsageLimits(request_limit=3)


async def answer(request: str) -> SupportReply:
    usage = RunUsage()
    routed = await router.run(request, usage=usage, usage_limits=limits)

    specialists = {
        'billing': billing,
        'technical': technical,
        'general': general,
    }
    specialist = specialists[routed.output.destination]
    result = await specialist.run(
        f'Respond to the request. Routing reason: {routed.output.reason}',
        # Preserve the user's request and the routing decision for the specialist.
        message_history=routed.new_messages(),
        usage=usage,
        usage_limits=limits,
    )
    return result.output


async def main() -> None:
    reply = await answer('I was charged twice for invoice INV-42. Can you reverse one charge?')
    assert isinstance(reply, BillingReply)
    print(reply.needs_refund_review)
    #> True
    print(reply.answer)
    #> I found the duplicate charge and sent it for refund review.


if __name__ == '__main__':
    asyncio.run(main())
```

The router does not call specialists as tools: application code owns the branch, so the return type remains explicit and every run contributes to the same [`RunUsage`][pydantic_ai.usage.RunUsage]. Add a deterministic fallback to `general` if your routing policy can decline a request.

## Related

See [Programmatic agent hand-off](../multi-agent-applications.md#programmatic-agent-hand-off) for interactive and delegated variants, and [Usage limits](../agent.md#usage-limits) for all available limits.
