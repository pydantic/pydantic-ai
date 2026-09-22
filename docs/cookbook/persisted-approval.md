---
title: Resume a tool approval in a later request
description: Persist a paused run, validate the approved call server-side, and resume it from a separate application request.
---

# Resume a tool approval in a later request

Persist the message history and pending tool-call ID when approval happens outside the agent request that proposed the action.

```bash
pip/uv-add "pydantic-ai-slim[openai]"
export OPENAI_API_KEY=your-api-key
```

```python {dunder_name="not_main"}
import asyncio
from pathlib import Path

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, RunContext
from pydantic_ai.messages import ModelMessagesTypeAdapter

agent = Agent('openai:gpt-5.6-sol', output_type=[str, DeferredToolRequests])
completed_refunds: dict[str, str] = {}


@agent.tool(requires_approval=True)
def refund_payment(ctx: RunContext, payment_id: str, amount_cents: int) -> str:
    # Use the tool-call ID as the idempotency key at the payment boundary.
    assert ctx.tool_call_id is not None
    if previous := completed_refunds.get(ctx.tool_call_id):
        return previous
    outcome = f'Refunded {amount_cents} cents from {payment_id}'
    completed_refunds[ctx.tool_call_id] = outcome
    return outcome


async def request_refund() -> str:
    result = await agent.run('Refund $49.99 from payment pay_123.')
    assert isinstance(result.output, DeferredToolRequests)
    call = result.output.approvals[0]
    Path('refund-history.json').write_bytes(ModelMessagesTypeAdapter.dump_json(result.all_messages()))
    Path('pending-call.txt').write_text(call.tool_call_id)
    print(f'Awaiting approval: {call.tool_name} {call.args}')
    """
    Awaiting approval: refund_payment {'payment_id': 'pay_123', 'amount_cents': 4999}
    """
    return call.tool_call_id


async def approve_refund(tool_call_id: str) -> None:
    # Creating this directory atomically claims the one pending approval.
    claim = Path('refund-claim')
    try:
        claim.mkdir()
    except FileExistsError:
        raise ValueError('This approval is already being processed.') from None

    pending = Path('pending-call.txt')
    claimed = claim / 'pending-call.txt'
    claimed_pending = False
    try:
        pending.replace(claimed)
        claimed_pending = True
        if tool_call_id != claimed.read_text():
            raise ValueError('This tool call is not awaiting approval.')

        history_path = Path('refund-history.json')
        history = ModelMessagesTypeAdapter.validate_json(history_path.read_bytes())
        approvals = DeferredToolResults(approvals={tool_call_id: True})
        result = await agent.run(message_history=history, deferred_tool_results=approvals)
        claimed.unlink()
        history_path.unlink()
        print(result.output)
        #> The $49.99 refund for payment pay_123 was completed.
    except Exception:
        if claimed_pending and claimed.exists():
            claimed.replace(pending)
        raise
    finally:
        claim.rmdir()


async def main() -> None:
    tool_call_id = await request_refund()  # first HTTP request
    try:
        await approve_refund('call_from_another_tenant')
    except ValueError as error:
        print(error)
        #> This tool call is not awaiting approval.
    await approve_refund(tool_call_id)  # later valid HTTP request


if __name__ == '__main__':
    asyncio.run(main())
```

Store the pending call and history under an authenticated tenant, atomically claim the approval before resuming, and consume it after success. The in-memory dictionary illustrates the required idempotency key; use the tool-call ID with an atomic uniqueness constraint at the real payment boundary. Authorization belongs in the tool as well; approval confirms intent but does not grant the caller new permissions.

## Related

See [Deferred tools](../deferred-tools.md) for rejection, external execution, and durable workflow integrations.
