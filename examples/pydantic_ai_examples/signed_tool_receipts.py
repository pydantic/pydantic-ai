"""Typed, hash-chained receipts for tool execution hooks.

Run with:

    uv run -m pydantic_ai_examples.signed_tool_receipts

This example uses the Hooks capability to emit one receipt before a tool runs
and one receipt after it returns. The receipt shape is defined with Pydantic
models, canonicalized with JSON, MAC-signed with HMAC-SHA256, and chained with
previous_receipt_hash.

HMAC keeps this example dependency-free. For third-party verification, replace
`sign_receipt` with an asymmetric signature such as Ed25519 and publish the
verification key with your agent or audit bundle.
"""

from __future__ import annotations as _annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks, ValidatedToolArgs
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition

SECRET_KEY = b'demo-signing-key-replace-me'


class ReceiptSignature(BaseModel):
    alg: Literal['HMAC-SHA256'] = 'HMAC-SHA256'
    kid: str = 'demo-key-1'
    sig: str


class ToolReceipt(BaseModel):
    receipt_version: Literal['pydantic-ai-tool-receipt-v0'] = (
        'pydantic-ai-tool-receipt-v0'
    )
    sequence: int
    phase: Literal['pre_execution', 'post_execution']
    tool_name: str
    tool_call_id: str | None
    arguments_digest: str
    result_digest: str | None = None
    previous_receipt_hash: str | None = None
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    signature: ReceiptSignature | None = None

    def unsigned_payload(self) -> dict[str, Any]:
        return self.model_dump(mode='json', exclude={'signature'})


receipt_chain: list[ToolReceipt] = []
hooks = Hooks()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf-8')


def sha256_hex(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def receipt_hash(receipt: ToolReceipt) -> str:
    return f'sha256:{sha256_hex(receipt.unsigned_payload())}'


def sign_receipt(receipt: ToolReceipt) -> ToolReceipt:
    signature = hmac.new(
        SECRET_KEY, canonical_json(receipt.unsigned_payload()), hashlib.sha256
    ).hexdigest()
    return receipt.model_copy(update={'signature': ReceiptSignature(sig=signature)})


def append_receipt(
    *,
    phase: Literal['pre_execution', 'post_execution'],
    call: ToolCallPart,
    args: ValidatedToolArgs,
    result: Any | None = None,
) -> ToolReceipt:
    previous_hash = receipt_hash(receipt_chain[-1]) if receipt_chain else None
    receipt = ToolReceipt(
        sequence=len(receipt_chain) + 1,
        phase=phase,
        tool_name=call.tool_name,
        tool_call_id=call.tool_call_id,
        arguments_digest=f'sha256:{sha256_hex(args)}',
        result_digest=f'sha256:{sha256_hex(result)}' if result is not None else None,
        previous_receipt_hash=previous_hash,
    )
    signed = sign_receipt(receipt)
    receipt_chain.append(signed)
    return signed


def verify_chain(receipts: list[ToolReceipt]) -> bool:
    previous_hash: str | None = None

    for receipt in receipts:
        if receipt.signature is None:
            return False
        if receipt.previous_receipt_hash != previous_hash:
            return False

        expected = hmac.new(
            SECRET_KEY,
            canonical_json(receipt.unsigned_payload()),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, receipt.signature.sig):
            return False

        previous_hash = receipt_hash(receipt)

    return True


@hooks.on.before_tool_execute
async def sign_pre_execution(
    ctx: RunContext[None],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: ValidatedToolArgs,
) -> ValidatedToolArgs:
    receipt = append_receipt(phase='pre_execution', call=call, args=args)
    print(
        f'[receipt] pre  #{receipt.sequence} {tool_def.name}: {receipt_hash(receipt)}'
    )
    return args


@hooks.on.after_tool_execute
async def sign_post_execution(
    ctx: RunContext[None],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: ValidatedToolArgs,
    result: Any,
) -> Any:
    receipt = append_receipt(
        phase='post_execution', call=call, args=args, result=result
    )
    print(
        f'[receipt] post #{receipt.sequence} {tool_def.name}: {receipt_hash(receipt)}'
    )
    return result


agent = Agent(
    TestModel(call_tools=['get_invoice', 'draft_refund']),
    instructions='Look up the invoice and draft a refund for the full amount.',
    capabilities=[hooks],
)


@agent.tool_plain
def get_invoice(invoice_id: str) -> dict[str, Any]:
    """Look up an invoice by ID."""
    return {
        'invoice_id': invoice_id,
        'customer': 'Acme Co',
        'amount_usd': 125,
        'status': 'paid',
    }


@agent.tool_plain
def draft_refund(invoice_id: str, amount_usd: int) -> str:
    """Draft a refund for an invoice."""
    return f'Drafted refund for {invoice_id}: ${amount_usd}'


if __name__ == '__main__':
    result = agent.run_sync(
        'Look up invoice INV-1001 and draft a refund for the full amount.'
    )

    print('\nResponse:', result.output)
    print('\n--- Signed receipt chain ---')
    print(json.dumps([r.model_dump(mode='json') for r in receipt_chain], indent=2))
    print(f'\nReceipt chain valid: {verify_chain(receipt_chain)}')
