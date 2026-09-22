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
from contextlib import closing, suppress
from uuid import uuid4

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, RunContext
from pydantic_ai.messages import ModelMessagesTypeAdapter

agent = Agent('openai:gpt-5.6-sol', output_type=[str, DeferredToolRequests])


def open_database() -> sqlite3.Connection:
    database = sqlite3.connect('approvals.db')
    database.execute(
        'CREATE TABLE IF NOT EXISTS approvals ('
        'tenant_id TEXT NOT NULL, tool_call_id TEXT NOT NULL, history BLOB NOT NULL, '
        'consumed INTEGER NOT NULL DEFAULT 0, claim_owner TEXT, claim_until REAL, '
        'PRIMARY KEY (tenant_id, tool_call_id))'
    )
    database.execute(
        'CREATE TABLE IF NOT EXISTS completed_refunds ('
        'tool_call_id TEXT PRIMARY KEY, outcome TEXT NOT NULL)'
    )
    return database


@agent.tool(requires_approval=True)
def refund_payment(ctx: RunContext, payment_id: str, amount_cents: int) -> str:
    # Pass this key to the real payment provider as its durable idempotency key.
    assert ctx.tool_call_id is not None
    outcome = f'Refunded {amount_cents} cents from {payment_id}'
    with closing(open_database()) as database, database:
        database.execute(
            'INSERT OR IGNORE INTO completed_refunds VALUES (?, ?)',
            (ctx.tool_call_id, outcome),
        )
        stored = database.execute(
            'SELECT outcome FROM completed_refunds WHERE tool_call_id = ?',
            (ctx.tool_call_id,),
        ).fetchone()
    assert stored is not None
    return str(stored[0])


async def request_refund(tenant_id: str) -> str:
    result = await agent.run('Refund $49.99 from payment pay_123.')
    assert isinstance(result.output, DeferredToolRequests)
    call = result.output.approvals[0]
    history = ModelMessagesTypeAdapter.dump_json(result.all_messages())
    with closing(open_database()) as database, database:
        database.execute(
            'INSERT INTO approvals (tenant_id, tool_call_id, history) VALUES (?, ?, ?)',
            (tenant_id, call.tool_call_id, history),
        )
    print(f'Awaiting approval: {call.tool_name} {call.args}')
    """
    Awaiting approval: refund_payment {'payment_id': 'pay_123', 'amount_cents': 4999}
    """
    return call.tool_call_id


async def renew_lease(tenant_id: str, tool_call_id: str, owner: str) -> None:
    while True:
        await asyncio.sleep(20)
        with closing(open_database()) as database, database:
            renewed = database.execute(
                'UPDATE approvals SET claim_until = ? '
                'WHERE tenant_id = ? AND tool_call_id = ? AND claim_owner = ? AND consumed = 0',
                (time.time() + 60, tenant_id, tool_call_id, owner),
            )
        if renewed.rowcount != 1:
            return


async def approve_refund(tenant_id: str, tool_call_id: str) -> None:
    owner = uuid4().hex
    now = time.time()
    with closing(open_database()) as database, database:
        # BEGIN IMMEDIATE serializes claim decisions. An expired lease is claimable after a crash.
        database.execute('BEGIN IMMEDIATE')
        row = database.execute(
            'SELECT consumed, claim_until, history FROM approvals '
            'WHERE tenant_id = ? AND tool_call_id = ?',
            (tenant_id, tool_call_id),
        ).fetchone()
        if row is None or row[0]:
            raise ValueError('This tool call is not awaiting approval for this tenant.')
        if row[1] is not None and row[1] > now:
            raise ValueError('This approval is already being processed.')
        database.execute(
            'UPDATE approvals SET claim_owner = ?, claim_until = ? '
            'WHERE tenant_id = ? AND tool_call_id = ?',
            (owner, now + 60, tenant_id, tool_call_id),
        )

    renewal = asyncio.create_task(renew_lease(tenant_id, tool_call_id, owner))
    try:
        history = ModelMessagesTypeAdapter.validate_json(row[2])
        approvals = DeferredToolResults(approvals={tool_call_id: True})
        result = await agent.run(message_history=history, deferred_tool_results=approvals)
        with closing(open_database()) as database, database:
            updated = database.execute(
                'UPDATE approvals SET consumed = 1, claim_owner = NULL, claim_until = NULL '
                'WHERE tenant_id = ? AND tool_call_id = ? AND claim_owner = ?',
                (tenant_id, tool_call_id, owner),
            )
            if updated.rowcount != 1:
                raise RuntimeError('The approval lease expired before completion.')
        print(result.output)
        #> The $49.99 refund for payment pay_123 was completed.
    except Exception:
        with closing(open_database()) as database, database:
            database.execute(
                'UPDATE approvals SET claim_owner = NULL, claim_until = NULL '
                'WHERE tenant_id = ? AND tool_call_id = ? AND claim_owner = ?',
                (tenant_id, tool_call_id, owner),
            )
        raise
    finally:
        renewal.cancel()
        with suppress(asyncio.CancelledError):
            await renewal


async def main() -> None:
    tool_call_id = await request_refund('tenant_123')  # first HTTP request
    try:
        await approve_refund('another_tenant', tool_call_id)
    except ValueError as error:
        print(error)
        #> This tool call is not awaiting approval for this tenant.
    await approve_refund('tenant_123', tool_call_id)  # later valid HTTP request


if __name__ == '__main__':
    asyncio.run(main())
```

Store each pending call and its history under an authenticated tenant, atomically claim the approval with a renewable database lease, and consume it after success. An abandoned lease becomes claimable after its deadline without allowing healthy long runs to overlap. The SQLite refund record illustrates durable deduplication; pass the tool-call ID to the real payment provider as its idempotency key. Authorization belongs in the tool as well; approval confirms intent but does not grant the caller new permissions.

## Related

See [Deferred tools](../deferred-tools.md) for rejection, external execution, and durable workflow integrations.
