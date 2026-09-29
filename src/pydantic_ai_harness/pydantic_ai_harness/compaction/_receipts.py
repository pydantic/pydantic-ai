"""Compaction receipts: a deterministic, honest note left in the surviving history.

After a strategy that crosses a compaction *boundary* (older history summarized or dropped)
rewrites the conversation, it can append a standard receipt so the model knows its memory of
everything before that point is secondhand and can re-verify rather than confabulate. The
receipt carries no timestamp -- it is a pure function of its inputs, so its bytes are
deterministic and testable -- and, when a handle provider is attached and discoverable, an
identifier for persisted run history.

Wording note: the exact receipt text is content, so it is shipped minimal/neutral and flagged
pending the benchmark eval-rig pass. The mechanism (presence, determinism, handle discovery,
OTel span event) is structural and lands now, gated behind each strategy's `receipts` flag.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.messages import TextContent, UserPromptPart
from pydantic_ai.tools import RunContext

_RECEIPT_MARKER = '[History before this point'
"""Prefix identifying a receipt part, for detection and de-accumulation across compactions."""

RECEIPT_EVENT_NAME = 'compaction.receipt'
"""Name of the OTel span event emitted alongside the `compact_messages` span."""


# ---------------------------------------------------------------------------
# Transcript-handle discovery
# ---------------------------------------------------------------------------


@runtime_checkable
class TranscriptHandleProvider(Protocol):
    """A capability that can hand out a handle for its persisted run history.

    Any capability implementing this method is discovered from `RunContext.capabilities`; the
    first non-`None` handle is used. `StepPersistence` implements it by returning its `run_id`.
    """

    def compaction_transcript_handle(self) -> str | None: ...  # pragma: no cover


def discover_transcript_handle(ctx: RunContext[AgentDepsT]) -> str | None:
    """Return a transcript handle from the first provider, else `None`."""
    capabilities = getattr(ctx, 'capabilities', None)
    if not capabilities:
        return None
    for capability in capabilities.values():
        if isinstance(capability, TranscriptHandleProvider):
            handle = capability.compaction_transcript_handle()
            if handle:
                return handle
    return None


# ---------------------------------------------------------------------------
# Receipt formatting
# ---------------------------------------------------------------------------


def format_receipt(
    *,
    dropped_messages: int,
    dropped_tokens: int,
    by: str,
    handle: str | None,
    has_summary: bool = True,
) -> str:
    """Render the standard, deterministic receipt string.

    No timestamp is included -- the output is a pure function of the arguments. *has_summary*
    tells the truth about what survived: a summarizing strategy leaves a summary the model can
    read (secondhand), while a drop-only strategy leaves nothing, so the caveat differs.
    """
    if has_summary:
        core = (
            f'was summarized by {by}. The summary above is secondhand; '
            're-verify critical facts against primary sources.'
        )
    else:
        core = (
            f'was dropped by {by}. That context is no longer in the window; '
            're-verify critical facts against primary sources.'
        )
    transcript = f' Persisted run handle: {handle}.' if handle else ''
    return f'{_RECEIPT_MARKER} ({dropped_messages} messages, ~{dropped_tokens} tokens) {core}{transcript}]'


_RECEIPT_METADATA = 'pydantic-ai-harness.compaction.receipt.v1'
"""Model-invisible metadata identifying a receipt content item."""


def make_receipt_part(text: str) -> UserPromptPart:
    """Wrap receipt *text* in a `UserPromptPart` for placement in the surviving history.

    The text remains user-role context while its metadata lets compaction distinguish it from a
    user turn.
    """
    return UserPromptPart(content=[TextContent(text, metadata=_RECEIPT_METADATA)])


def is_receipt_part(part: object) -> bool:
    """Return True if *part* carries the receipt metadata marker."""
    return (
        isinstance(part, UserPromptPart)
        and not isinstance(part.content, str)
        and any(isinstance(item, TextContent) and item.metadata == _RECEIPT_METADATA for item in part.content)
    )


# ---------------------------------------------------------------------------
# Span-event plumbing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReceiptInfo:
    """Structured receipt data recorded by a strategy and drained onto the compaction span."""

    strategy: str
    dropped_messages: int
    dropped_tokens: int
    by: str
    handle: str | None


_pending_receipts: ContextVar[list[ReceiptInfo] | None] = ContextVar(
    'pydantic_ai_harness.compaction.pending_receipts',
    default=None,
)
"""Async-context-local sink: `compact_with_span` opens a scope, strategies append, span drains."""


def open_receipt_scope() -> Token[list[ReceiptInfo] | None]:
    """Start a fresh receipt scope; returns the token to reset it with."""
    return _pending_receipts.set([])


def drain_receipts() -> list[ReceiptInfo]:
    """Return the receipts recorded in the current scope (empty when none/no scope)."""
    return list(_pending_receipts.get() or [])


def reset_receipt_scope(token: Token[list[ReceiptInfo] | None]) -> None:
    """Reset the receipt scope opened by `open_receipt_scope`."""
    _pending_receipts.reset(token)


def record_receipt(info: ReceiptInfo) -> None:
    """Record a receipt in the current scope; a no-op when called outside a scope."""
    pending = _pending_receipts.get()
    if pending is not None:
        pending.append(info)
