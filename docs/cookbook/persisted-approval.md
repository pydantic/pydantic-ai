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
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from uuid import uuid4

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
    with closing(sqlite3.connect('approvals.db')) as database, database:
        database.execute(
            'CREATE TABLE IF NOT EXISTS approvals ('
            'tool_call_id TEXT PRIMARY KEY, consumed INTEGER NOT NULL DEFAULT 0, '
            'claim_owner TEXT, claim_until REAL)'
        )
        database.execute('INSERT INTO approvals (tool_call_id) VALUES (?)', (call.tool_call_id,))
    print(f'Awaiting approval: {call.tool_name} {call.args}')
    """
    Awaiting approval: refund_payment {'payment_id': 'pay_123', 'amount_cents': 4999}
    """
    return call.tool_call_id


async def approve_refund(tool_call_id: str) -> None:
    owner = uuid4().hex
    now = time.time()
    with closing(sqlite3.connect('approvals.db')) as database, database:
        # BEGIN IMMEDIATE serializes claim decisions. An expired lease is claimable after a crash.
        database.execute('BEGIN IMMEDIATE')
        row = database.execute(
            'SELECT consumed, claim_until FROM approvals WHERE tool_call_id = ?',
            (tool_call_id,),
        ).fetchone()
        if row is None or row[0]:
            raise ValueError('This tool call is not awaiting approval.')
        if row[1] is not None and row[1] > now:
            raise ValueError('This approval is already being processed.')
        database.execute(
            'UPDATE approvals SET claim_owner = ?, claim_until = ? WHERE tool_call_id = ?',
            (owner, now + 60, tool_call_id),
        )

    try:
        history_path = Path('refund-history.json')
        history = ModelMessagesTypeAdapter.validate_json(history_path.read_bytes())
        approvals = DeferredToolResults(approvals={tool_call_id: True})
        result = await agent.run(message_history=history, deferred_tool_results=approvals)
        with closing(sqlite3.connect('approvals.db')) as database, database:
            updated = database.execute(
                'UPDATE approvals SET consumed = 1, claim_owner = NULL, claim_until = NULL '
                'WHERE tool_call_id = ? AND claim_owner = ?',
                (tool_call_id, owner),
            )
            if updated.rowcount != 1:
                raise RuntimeError('The approval lease expired before completion.')
        history_path.unlink()
        print(result.output)
        #> The $49.99 refund for payment pay_123 was completed.
    except Exception:
        with closing(sqlite3.connect('approvals.db')) as database, database:
            database.execute(
                'UPDATE approvals SET claim_owner = NULL, claim_until = NULL '
                'WHERE tool_call_id = ? AND claim_owner = ?',
                (tool_call_id, owner),
            )
        raise


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

Store the pending call and history under an authenticated tenant, atomically claim the approval with a renewable database lease, and consume it after success. An abandoned lease becomes claimable after its deadline. The in-memory dictionary illustrates the required idempotency key; use the tool-call ID with an atomic uniqueness constraint at the real payment boundary. Authorization belongs in the tool as well; approval confirms intent but does not grant the caller new permissions.

## Related

See [Deferred tools](../deferred-tools.md) for rejection, external execution, and durable workflow integrations.
