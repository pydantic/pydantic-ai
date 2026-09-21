---
title: Require approval for a sensitive tool
description: Pause an agent before it executes a consequential action, then resume with a human decision.
---

# Require approval for a sensitive tool

Set `requires_approval=True` on a tool that must never run solely because the model requested it.

```python {dunder_name="not_main"}
import asyncio

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults

agent = Agent(
    'openai:gpt-5.6-sol',
    output_type=[str, DeferredToolRequests],
)


@agent.tool_plain(requires_approval=True)
def refund_payment(payment_id: str, amount_cents: int) -> str:
    """Refund part or all of a captured payment."""
    return f'Refunded {amount_cents} cents from {payment_id}'


async def main() -> None:
    result = await agent.run('Refund $49.99 from payment pay_123.')
    messages = result.all_messages()

    assert isinstance(result.output, DeferredToolRequests)
    approvals = DeferredToolResults()
    for call in result.output.approvals:
        print(f'Requested: {call.tool_name} {call.args}')
        #> Requested: refund_payment {'payment_id': 'pay_123', 'amount_cents': 4999}
        approved = input('Approve? [y/N] ').lower() == 'y'
        approvals.approvals[call.tool_call_id] = approved

    result = await agent.run(
        message_history=messages,
        deferred_tool_results=approvals,
    )
    print(result.output)
    #> The $49.99 refund for payment pay_123 was completed.


if __name__ == '__main__':
    asyncio.run(main())
```

The first run returns the proposed call without executing `refund_payment`. The second run executes it only when the corresponding tool-call ID is approved. Authentication and authorization still belong inside the application; approval protects against autonomous model action rather than an untrusted client.
