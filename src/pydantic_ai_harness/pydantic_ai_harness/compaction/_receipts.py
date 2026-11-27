"""Compaction receipts: a deterministic note left in the surviving history.

After a strategy that crosses a compaction *boundary* (older history summarized or dropped)
rewrites the conversation, it can append a standard receipt so the model knows its memory of
everything before that point is secondhand and can re-verify rather than confabulate. The
receipt carries no timestamp -- it is a pure function of its inputs, so its bytes are
deterministic and testable -- and, when a handle provider is attached and discoverable, an
identifier for persisted run history. Each strategy gates it behind its `receipts` flag.

The receipt is a `CompactionReceiptPart` in the application's voice, so its text is only ever the
capability's own wording with config and counts filled in: the strategy name, who compacted, how
much, and the run handle. Never the summary, the user's words, a tool's output or a file's text.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    InstructionCapabilityPart,
    ModelMessage,
    ModelRequest,
    OperatorSource,
    OperatorText,
    TextContent,
    UserPromptPart,
)
from pydantic_ai.tools import AgentDepsT, RunContext

_RECEIPT_MARKER = '[History before this point'
"""How a receipt's text opens. Never used to recognize one: that's by type."""

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
    for capability in ctx.capabilities.values():
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
"""Metadata on the user-prompt receipts older versions wrote, which is how they're recognized."""


@dataclass(kw_only=True)
class CompactionReceiptPart(InstructionCapabilityPart, namespace='compaction', name='receipt'):
    """The receipt a compaction strategy with `receipts=True` leaves at its cut point.

    It's in the application's voice and conversation-scoped, so it stays with the compacted
    history. Find receipts in history by this type.
    """


def make_receipt_part(text: str, *, capability_id: str | None) -> CompactionReceiptPart:
    """Wrap receipt *text*, from `format_receipt`, for placement in the surviving history.

    *capability_id* is the run id of the strategy that wrote it, when it's registered on the run.
    """
    return CompactionReceiptPart(
        # The capability's own wording, with only config and counts filled in.
        content=OperatorText(text),
        source=OperatorSource(),
        scope='conversation',
        capability_id=capability_id,
    )


def registered_id(ctx: RunContext[AgentDepsT], capability: AbstractCapability[AgentDepsT]) -> str | None:
    """The run id *capability* is registered under, or `None` when it isn't, such as a tier of `TieredCompaction`."""
    return next((run_id for run_id, registered in ctx.capabilities.items() if registered is capability), None)


def is_receipt_part(part: object) -> bool:
    """Whether *part* is a compaction receipt: a `CompactionReceiptPart`, or the legacy user-prompt form."""
    if isinstance(part, CompactionReceiptPart):
        return True
    # Legacy form: older versions wrote the receipt as a user prompt. It's recognized by its model-invisible
    # metadata only, never by its text, so a user who types the receipt's wording isn't taken for one.
    return (
        isinstance(part, UserPromptPart)
        and not isinstance(part.content, str)
        and any(isinstance(item, TextContent) and item.metadata == _RECEIPT_METADATA for item in part.content)
    )


def is_receipt_message(message: ModelMessage) -> bool:
    """Return True if *message* is a request made up entirely of receipt parts."""
    return isinstance(message, ModelRequest) and bool(message.parts) and all(is_receipt_part(p) for p in message.parts)


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
