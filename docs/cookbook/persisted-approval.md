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

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults
from pydantic_ai.messages import ModelMessagesTypeAdapter

agent = Agent('openai:gpt-5.6-sol', output_type=[str, DeferredToolRequests])


@agent.tool_plain(requires_approval=True)
def refund_payment(payment_id: str, amount_cents: int) -> str:
    return f'Refunded {amount_cents} cents from {payment_id}'


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
    expected_call_id = Path('pending-call.txt').read_text()
    if tool_call_id != expected_call_id:
        raise ValueError('This tool call is not awaiting approval.')

    history = ModelMessagesTypeAdapter.validate_json(Path('refund-history.json').read_bytes())
    approvals = DeferredToolResults(approvals={tool_call_id: True})
    result = await agent.run(message_history=history, deferred_tool_results=approvals)
    print(result.output)
    #> The $49.99 refund for payment pay_123 was completed.


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

Store the pending call and history under an authenticated tenant and verify the submitted call ID against that server-side record. Authorization belongs in the tool as well; approval confirms intent but does not grant the caller new permissions.

## Related

See [Deferred tools](../deferred-tools.md) for rejection, external execution, and durable workflow integrations.
