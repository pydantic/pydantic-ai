---
title: Authorize a refund during a realtime call
description: Keep caller identity outside the transcript and deny an action against an order the authenticated account does not own.
---

# Authorize a refund during a realtime call

Keep caller identity and resource ownership in typed dependencies rather than trusting what was spoken. Resolve approval-gated tools with application policy because a realtime session cannot pause for a later human response.

```bash
pip/uv-add "pydantic-ai-slim[openai-realtime]"
export OPENAI_API_KEY=your-api-key
```

```python {dunder_name="not_main"}
import asyncio
from dataclasses import dataclass, field

from pydantic_ai import (
    Agent,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    ToolDenied,
)
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.messages import ToolReturnPart


@dataclass
class Account:
    customer_id: str
    order_ids: set[str]
    remaining_refundable: dict[str, float]
    completed_tool_calls: set[str] = field(default_factory=set)


agent = Agent(deps_type=Account, instructions='Help the caller with their authenticated account.')
executed_refunds: list[str] = []


@agent.tool(requires_approval=True, sequential=True)
async def refund_authenticated_order(
    ctx: RunContext[Account], order_id: str, amount: float
) -> str:
    assert ctx.tool_call_id is not None
    if ctx.tool_call_id in ctx.deps.completed_tool_calls:
        return 'This refund request was already completed.'

    remaining = ctx.deps.remaining_refundable.get(order_id, 0)
    if order_id not in ctx.deps.order_ids or amount <= 0 or amount > remaining:
        return 'Refund refused: the amount is not currently refundable.'

    ctx.deps.remaining_refundable[order_id] = remaining - amount
    ctx.deps.completed_tool_calls.add(ctx.tool_call_id)
    executed_refunds.append(f'{ctx.deps.customer_id}:{order_id}')
    return f'Refunded ${amount:.2f}.'


async def refund_policy(
    ctx: RunContext[Account], requests: DeferredToolRequests
) -> DeferredToolResults:
    decisions = DeferredToolResults()
    for call in requests.approvals:
        args = call.args_as_dict()
        if args['order_id'] not in ctx.deps.order_ids:
            decision = ToolDenied('That order is not in the authenticated account.')
        elif args['amount'] <= 0:
            decision = ToolDenied('The refund amount must be positive.')
        elif args['amount'] > 100:
            decision = ToolDenied('Refunds over $100 need a human agent.')
        elif args['amount'] > ctx.deps.remaining_refundable.get(args['order_id'], 0):
            decision = ToolDenied('That amount exceeds the remaining refundable balance.')
        else:
            decision = True
        decisions.approvals[call.tool_call_id] = decision
    return decisions


async def main() -> None:
    realtime = agent.realtime(
        'openai:gpt-realtime',
        deps=Account(
            customer_id='cus_123',
            order_ids={'B200'},
            remaining_refundable={'B200': 75},
        ),
        capabilities=[HandleDeferredToolCalls(handler=refund_policy)],
    )
    async with realtime.session() as session:
        transcripts = session.stream_transcripts()
        await session.send('Refund $50 for order A100.')
        async for part in transcripts:
            print(f'{part.speaker}: {part.transcript}')
            #> user: Refund $50 for order A100.
            #> assistant: I cannot access that order; I can connect you to an agent.
            if part.speaker == 'assistant':
                break

    refund_result = next(
        part
        for message in session.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'refund_authenticated_order'
    )
    print(refund_result.outcome, executed_refunds)
    #> denied []


if __name__ == '__main__':
    asyncio.run(main())
```

The tool receives `customer_id` and the remaining refundable balance from the authenticated application boundary, not from model-generated arguments. The handler screens ownership, amount, and balance before execution; the sequential tool checks them again, updates the balance, and records the tool-call ID for idempotency. The denied call is retained in history without executing the refund. In production, perform the balance update and idempotency insert in one database transaction. Use a standard agent run when a person must approve asynchronously after the call has paused.

## Related

See [Realtime tools](../realtime/tools.md#deferred-and-approval-required-tools) for inline policy handling and the [realtime handoff example](../examples/realtime-handoff.md) for transferring a call to a text workflow.
