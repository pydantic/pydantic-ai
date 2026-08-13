"""Shared utilities for the compaction capabilities.

Token estimation, the `CompactionStrategy` protocol, tool-pair-safe cutoff logic, first-user
preservation, and in-place tool-result clearing -- anything used by more than one capability.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from json import dumps
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from weakref import ReferenceType, ref

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.messages import (
    CompactionPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    NativeToolCallPart,
    NativeToolReturnPart,
    RetryPromptPart,
    SpeechPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import AbstractModel, Model
from pydantic_ai.tools import RunContext
from typing_extensions import Self, assert_never

from pydantic_ai_harness.compaction._context_window import DEFAULT_CONTEXT_WINDOW, resolve_context_window
from pydantic_ai_harness.compaction._pinning import is_pinned
from pydantic_ai_harness.compaction._receipts import (
    RECEIPT_EVENT_NAME,
    drain_receipts,
    is_receipt_part,
    open_receipt_scope,
    reset_receipt_scope,
)

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
    from pydantic_ai.tools import ToolDefinition

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN = 4
"""Rough approximation: ~4 characters per token on average."""

_COMPACTION_RECLAIM: ContextVar[tuple[ReferenceType[object], int] | None] = ContextVar(
    'pydantic_ai_harness.compaction.reclaim', default=None
)
"""Heuristic reclaim from compaction that ran earlier in this request's hook chain."""


def _collect_message_text(messages: Sequence[ModelMessage]) -> list[str]:
    """Collect all text segments from a sequence of messages, excluding instructions.

    Every part that carries text the provider is sent counts, including the ones a run only
    grows under load: a retry prompt, an extended-thinking block, the result of a
    provider-side tool. Leaving those out was tolerable while the budget was an absolute
    `max_tokens` the caller had calibrated against this same estimator; a fraction of the
    real window is only meaningful if the numerator measures the same request the
    denominator describes.

    `FilePart` is deliberately absent: its payload is binary, and counting its length as
    characters would be a number with no relation to what the provider bills.
    """
    segments: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for request_part in msg.parts:
                segments.extend(_request_part_text(request_part))
        else:
            for response_part in msg.parts:
                segments.extend(_response_part_text(response_part))
    return segments


def _collect_text(messages: Sequence[ModelMessage]) -> list[str]:
    """Collect all text segments from a sequence of messages, instructions included."""
    segments = _collect_message_text(messages)
    segments.extend(_instructions_text(messages))
    return segments


def _request_part_text(part: ModelRequestPart) -> list[str]:
    """Text segments a single request part contributes to the token estimate."""
    if isinstance(part, UserPromptPart):
        return [_user_prompt_text_for_counting(part)]
    elif isinstance(part, SystemPromptPart):
        return [part.content]
    elif isinstance(part, (ToolReturnPart, RetryPromptPart)):
        # Both are sent in full. The tool-search and capability-load returns subclass
        # `ToolReturnPart`, so they arrive here too.
        return [str(part.content)]
    # Control bookkeeping rather than message text: it records which tools became available, and
    # the schemas themselves travel in the request's tool definitions. Those schemas are not free
    # -- this estimator counts no tool definitions at all, so revealing tools mid-run costs
    # context it does not see (tracked separately); skipping the part is not a claim that the
    # reveal was free.
    elif isinstance(part, ToolAvailabilityDeltaPart):
        return []
    elif isinstance(part, SpeechPart):  # pyright: ignore[reportUnnecessaryIsInstance]
        # A realtime turn arrives as spoken audio plus a transcript. Count the transcript -- the
        # words the provider bills against the window -- not the binary audio, which has no
        # character-count meaning (as with `FilePart`).
        #
        # `SpeechPart` is the last member of the union today, so pyright sees this `isinstance` as
        # redundant, but it is kept explicit so the `else` stays a real branch at runtime rather
        # than dead code -- see the `else` comment.
        return [part.transcript or '']
    else:
        # A part pydantic-ai added after this was written. Skip rather than guess at its payload:
        # this runs on every request to decide whether to compact, so an unrecognised part must
        # not take the run down -- which is exactly what the old `str(part.content)` fallback did
        # once a part without `content` existed (#577).
        #
        # `assert_never` sits under `TYPE_CHECKING` rather than in the branch body on purpose. It
        # still makes the chain exhaustive at type-check time, so a new upstream part fails
        # `make typecheck` and becomes a decision we make; but unlike the usual runtime form it
        # cannot re-crash a user who upgrades pydantic-ai ahead of us, which is the failure this
        # change exists to remove.
        if TYPE_CHECKING:
            assert_never(part)
        return []  # pragma: no cover - reachable only on a pydantic-ai release newer than this one


def _response_part_text(part: ModelResponsePart) -> list[str]:
    """Text segments a single response part contributes to the token estimate."""
    if isinstance(part, TextPart):
        return [part.content]
    elif isinstance(part, ToolCallPart):
        return [part.tool_name, str(part.args)]
    elif isinstance(part, (ThinkingPart, CompactionPart)):
        # A redacted thinking block carries a signature and no text.
        return [part.content or '']
    elif isinstance(part, NativeToolCallPart):
        return [part.tool_name, str(part.args)]
    elif isinstance(part, NativeToolReturnPart):
        return [part.tool_name, str(part.content)]
    elif isinstance(part, SpeechPart):
        # `SpeechPart` is in both unions; a realtime assistant turn lands here. Count its
        # transcript for the same reason as on the request side.
        return [part.transcript or '']
    # Any other response part (e.g. `FilePart`) contributes no counted text, as before.
    return []


def _instructions_text(messages: Sequence[ModelMessage]) -> list[str]:
    """The instructions this history would be sent with, counted once.

    Every `ModelRequest` in a history carries the instructions in force when it was made, but
    a request sends one set. Summing them would multiply a system prompt by the number of
    turns, which is the opposite error from ignoring it. The most recent non-empty set is the
    one that goes, mirroring how a model synthesizes instructions from history.
    """
    for msg in reversed(messages):
        if isinstance(msg, ModelRequest) and msg.instructions:
            return [msg.instructions]
    return []


def _user_prompt_text_for_counting(part: UserPromptPart) -> str:
    """Extract text content from a user prompt part for counting."""
    if isinstance(part.content, str):
        return part.content
    texts: list[str] = []
    for item in part.content:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, TextContent):
            texts.append(item.content)
    return ''.join(texts)


def estimate_text_tokens(text: str, tokenizer: Callable[[str], int] | None = None) -> int:
    """Approximate the token count of a single string.

    Uses *tokenizer* when given, otherwise the ~4 characters-per-token heuristic.
    """
    if tokenizer is not None:
        return tokenizer(text)
    return len(text) // _CHARS_PER_TOKEN


def estimate_token_count(
    messages: Sequence[ModelMessage],
    tokenizer: Callable[[str], int] | None = None,
) -> int:
    """Approximate token count for a sequence of messages.

    Args:
        messages: Messages to count tokens for.
        tokenizer: Optional callable that returns the token count for a string.
            When ``None``, falls back to a ~4 characters-per-token heuristic.
    """
    segments = _collect_text(messages)
    if tokenizer is not None:
        return sum(tokenizer(s) for s in segments)
    return sum(len(s) for s in segments) // _CHARS_PER_TOKEN


def estimate_context_tokens(
    messages: Sequence[ModelMessage],
    tokenizer: Callable[[str], int] | None = None,
    *,
    model_request_parameters: ModelRequestParameters | None = None,
) -> int:
    """Best-available token count for the request this history would produce.

    Anchors on the most recent `ModelResponse` carrying provider-reported usage: its
    `input_tokens` measured everything the provider was actually sent for that request --
    instructions, tool definitions, and every prior message -- and its `output_tokens` measured
    the response's own parts, so their sum is ground truth for the history up to and including
    that response. Only the messages after the anchor are estimated with the character
    heuristic (or *tokenizer*), without re-counting instructions, which the anchor already
    covers. Tool schemas named by availability deltas after the anchor are estimated from the
    pending request parameters. This is what makes the estimate robust where the pure heuristic
    is not: token-dense content (minified JSON, base64, non-Latin scripts) and the tool
    definitions the heuristic cannot see at all are both inside the provider's number.

    Without provider usage, the history's message text falls back to `estimate_token_count`,
    while schemas named by availability deltas are still estimated from the pending request
    parameters.

    A history rewritten *after* the anchor's request went out (a compaction strategy editing
    older messages mid-cycle) is overestimated, because the anchor still describes the
    pre-rewrite request; the next real response re-anchors it. `TieredCompaction` compensates
    inside its escalation loop by subtracting each tier's estimated reclaim from its anchored
    baseline. Compacting slightly early is the cheap failure mode; the heuristic's multi-x
    underestimate on dense content, which lets the history blow the context window, is the
    expensive one.
    """
    if anchor := _latest_usage_anchor(messages):
        index, message = anchor
        anchored = message.usage.input_tokens + message.usage.output_tokens
        segments = _collect_message_text(messages[index + 1 :])
        # The anchor paid for the instructions in force when its request was made. When the
        # latest instructions differ (dynamic instructions, or a persisted history resumed
        # under a new prompt), the new set is absent from both the anchor and the message
        # text, so count it; when unchanged, counting it would double what the anchor holds.
        current_instructions = _instructions_text(messages)
        if current_instructions != _instructions_text(messages[: index + 1]):
            segments = [*segments, *current_instructions]
        segments.extend(_revealed_tool_schema_text(messages[index + 1 :], model_request_parameters))
        if tokenizer is not None:
            return anchored + sum(tokenizer(s) for s in segments)
        return anchored + sum(len(s) for s in segments) // _CHARS_PER_TOKEN
    segments = [*_collect_text(messages), *_revealed_tool_schema_text(messages, model_request_parameters)]
    if tokenizer is not None:
        return sum(tokenizer(s) for s in segments)
    return sum(len(s) for s in segments) // _CHARS_PER_TOKEN


def has_context_usage_anchor(messages: Sequence[ModelMessage]) -> bool:
    """Return whether `estimate_context_tokens` uses provider-reported usage for *messages*."""
    return _latest_usage_anchor(messages) is not None


def _latest_usage_anchor(messages: Sequence[ModelMessage]) -> tuple[int, ModelResponse] | None:
    """Return the most recent response with provider-reported input usage."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, ModelResponse) and message.usage.input_tokens:
            return index, message
    return None


def _revealed_tool_names(messages: Sequence[ModelMessage]) -> set[str]:
    """Return tool names recorded as newly available in *messages*."""
    return {
        tool_name
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolAvailabilityDeltaPart)
        for tool_name in part.tools_added
    }


def _revealed_tool_schema_text(
    messages: Sequence[ModelMessage], model_request_parameters: ModelRequestParameters | None
) -> list[str]:
    """Return schemas added to the request by availability deltas in *messages*."""
    if model_request_parameters is None:
        return []
    return [
        _tool_schema_text(tool)
        for tool_name in _revealed_tool_names(messages)
        if (tool := model_request_parameters.tool_defs.get(tool_name)) is not None
    ]


def _tool_schema_text(tool: ToolDefinition) -> str:
    """Return the model-visible text fields of a tool definition for local estimation."""
    return ''.join((tool.name, tool.description or '', dumps(tool.parameters_json_schema, sort_keys=True)))


def record_compaction_reclaim(request_context: ModelRequestContext, before: int, after: int) -> None:
    """Record a conservative correction for a later usage reporter in this hook chain."""
    previous = _COMPACTION_RECLAIM.get()
    reclaimed = max(before - after, 0)
    if previous is not None and previous[0]() is request_context:
        reclaimed += previous[1]
    _COMPACTION_RECLAIM.set((ref(request_context), reclaimed))


def get_compaction_reclaim(request_context: ModelRequestContext) -> int:
    """Return the reclaim recorded for *request_context*, if it is still current."""
    previous = _COMPACTION_RECLAIM.get()
    if previous is None or previous[0]() is not request_context:
        return 0
    return previous[1]


def exceeds(
    messages: Sequence[ModelMessage],
    max_messages: int | None,
    max_tokens: int | None,
    tokenizer: Callable[[str], int] | None,
    *,
    model_request_parameters: ModelRequestParameters | None = None,
) -> bool:
    """Return True if *messages* exceeds either configured size threshold."""
    if max_messages is not None and len(messages) > max_messages:
        return True
    if (
        max_tokens is not None
        and estimate_context_tokens(messages, tokenizer, model_request_parameters=model_request_parameters) > max_tokens
    ):
        return True
    return False


# ---------------------------------------------------------------------------
# Token triggers
# ---------------------------------------------------------------------------


def validate_token_trigger(
    max_tokens: int | None,
    max_fraction: float | None,
    fallback_context_window: int = DEFAULT_CONTEXT_WINDOW,
    context_window: int | None = None,
    *,
    tokens_name: str = 'max_tokens',
    fraction_name: str = 'max_fraction',
) -> None:
    """Validate the absolute/relative token trigger a strategy was configured with.

    The two triggers are mutually exclusive: a strategy that took both would have to pick one
    and discard the other, and the caller could not tell which budget was in force. The names
    are parameterized so `TieredCompaction`'s `target_*` pair reports its own field names.
    """
    if max_tokens is not None and max_fraction is not None:
        raise ValueError(f'Set at most one of {tokens_name} or {fraction_name}.')
    if max_tokens is not None and max_tokens < 1:
        raise ValueError(f'{tokens_name} must be positive.')
    if max_fraction is not None and not 0 < max_fraction <= 1:
        raise ValueError(f'{fraction_name} must be greater than 0 and at most 1.')
    if fallback_context_window < 1:
        raise ValueError('fallback_context_window must be positive.')
    if context_window is not None and context_window < 1:
        raise ValueError('context_window must be positive.')


def context_for_request(
    ctx: RunContext[AgentDepsT],
    request_context: ModelRequestContext,
) -> RunContext[AgentDepsT]:
    """The run context as it applies to *this* request.

    A capability may replace `ModelRequestContext.model` before the request leaves, so a
    strategy driven from `before_model_request` has to see the model the request is going to
    rather than the one the run started with. Everything a strategy reads off the context
    follows: the window a fraction resolves against, the model a summarizing tier calls, and
    the same for a `TieredCompaction` nested inside another one.

    Returns `ctx` itself when no capability replaced the model, which is the common case.
    """
    if request_context.model is ctx.model:
        return ctx
    return replace(ctx, model=request_context.model)


def is_realtime_model(model: AbstractModel | str) -> bool:
    """Whether *model* is a realtime model: an `AbstractModel` that is not a request-response `Model`.

    A realtime model has no request-response context-window/token semantics, so the token
    triggers skip it. Written as a boolean check (rather than an inline `isinstance`) so callers
    keep the `AbstractModel | str` type -- narrowing against the un-parameterized generic `Model`
    would otherwise widen the value to `Model[Unknown]`. #585
    """
    return isinstance(model, AbstractModel) and not isinstance(model, Model)


def resolve_token_trigger(
    max_tokens: int | None,
    max_fraction: float | None,
    model: AbstractModel | str,
    fallback_context_window: int = DEFAULT_CONTEXT_WINDOW,
    context_window: int | None = None,
) -> int | None:
    """Absolute token trigger for this request, or `None` when no token trigger is configured.

    `max_fraction` is resolved against the model's real context window, so one configuration
    behaves correctly on a 128K model and on a 1M one. When the window cannot be resolved
    `fallback_context_window` stands in; it defaults to the conservative
    `DEFAULT_CONTEXT_WINDOW`, because compacting earlier than necessary costs a summary while
    overestimating costs the whole request.

    `context_window` overrides resolution entirely. The registry can be confidently wrong --
    it records the maximum a model can be made to accept, which for a beta-gated or
    tier-gated window is not what an ordinary request gets, and a self-hosted endpoint
    reports an id whose registry entry describes someone else's deployment. `fallback_*`
    cannot cover those: it only applies when resolution *fails*, and here it succeeds.

    Pass the model the request will actually be sent to. A capability may replace
    `ModelRequestContext.model` before the request leaves, so the run's model and the request's
    model are not always the same one.

    `RunContext.model` is typed as the wider `AbstractModel`, but a realtime session never
    compacts: its message history can't be modified mid-run, so the compaction hooks that call
    this never fire, and `model` is a request-response `Model` at runtime. The realtime guard
    below only keeps that wider type sound -- it returns `None`. #585
    """
    if is_realtime_model(model):
        # Unreachable at runtime (a realtime session doesn't compact, see above); the guard exists
        # only because `RunContext.model` is now the wider `AbstractModel`. #585
        return None
    if max_tokens is not None:
        return max_tokens
    if max_fraction is None:
        return None
    if context_window is None:
        context_window = resolve_context_window(model)
    return max(1, int((context_window if context_window is not None else fallback_context_window) * max_fraction))


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------

_SPAN_NAME = 'compact_messages'
"""Static, low-cardinality span name emitted whenever a strategy compacts. The strategy name
goes in the `compaction.strategy` attribute rather than the span name to keep cardinality low."""


def _history_changed(before: list[ModelMessage], after: list[ModelMessage]) -> bool:
    """Return True if *after* differs from *before*.

    The same list object, or an equal-length list that compares equal element-wise, counts as
    unchanged; anything else is a change.
    """
    # `!=` short-circuits on identity element-wise, so this also covers `before is after`; an
    # unequal length already implies an unequal list, so a separate length check is redundant.
    return before != after


async def compact_with_span(
    ctx: RunContext[AgentDepsT],
    *,
    strategy: str,
    messages: list[ModelMessage],
    compact: Callable[[], Awaitable[list[ModelMessage]]],
    tokenizer: Callable[[str], int] | None = None,
) -> list[ModelMessage]:
    """Run *compact* and emit a `compact_messages` span when it changes the history.

    *compact* runs before the span so a no-op compaction (a trigger fired but the history is
    returned unchanged) emits nothing. The span is started on `ctx.tracer`, which is a no-op
    tracer unless core's instrumentation is active, so this adds no overhead to a
    non-instrumented run; the before/after attributes are only computed when the span records.

    Args:
        ctx: Run context whose `tracer` the span is started on.
        strategy: Strategy name recorded in the `compaction.strategy` attribute.
        messages: The pre-compaction messages, measured for the `*_before` attributes.
        compact: Zero-argument async callable returning the compacted message list.
        tokenizer: Optional tokenizer for the `compaction.tokens_*` estimates. When `None`,
            uses the same ~4 characters-per-token heuristic as `estimate_token_count`.
    """
    token = open_receipt_scope()
    try:
        compacted = await compact()
        receipts = drain_receipts()
    finally:
        reset_receipt_scope(token)
    if not _history_changed(messages, compacted):
        return messages
    with ctx.tracer.start_as_current_span(_SPAN_NAME) as span:
        if span.is_recording():
            span.set_attributes(
                {
                    # GenAI semconv flag; the convention says set `true` only, never `false`.
                    'gen_ai.conversation.compacted': True,
                    'compaction.strategy': strategy,
                    'compaction.messages_before': len(messages),
                    'compaction.messages_after': len(compacted),
                    'compaction.tokens_before': estimate_token_count(messages, tokenizer),
                    'compaction.tokens_after': estimate_token_count(compacted, tokenizer),
                }
            )
            for receipt in receipts:
                attributes = {
                    'compaction.receipt.strategy': receipt.strategy,
                    'compaction.receipt.messages_dropped': receipt.dropped_messages,
                    'compaction.receipt.tokens_dropped': receipt.dropped_tokens,
                    'compaction.receipt.by': receipt.by,
                }
                if receipt.handle is not None:
                    attributes['compaction.receipt.handle'] = receipt.handle
                span.add_event(RECEIPT_EVENT_NAME, attributes)
    return compacted


# ---------------------------------------------------------------------------
# Compaction strategy protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SupportsFocus(Protocol):
    """A strategy whose output can be steered toward a topic.

    Only a strategy that *writes* something -- a summary -- can be focused. One that drops or
    blanks content by rule has nothing to steer, so `compact_now` passes a focus it cannot
    honour over rather than rejecting it. A composing strategy is focusable when any strategy
    it wraps is, so the hint reaches the tier that writes the prose.
    """

    def with_focus(self, focus: str) -> Self:
        """Return a copy of this strategy that prioritizes `focus`."""
        ...  # pragma: no cover


class CompactionStrategy(Protocol[AgentDepsT]):
    """A history transform that can be used standalone or as a `TieredCompaction` tier.

    `compact` applies the transform *unconditionally* (the trigger check lives in the
    capability's `before_model_request`).  A strategy that composes others may define its own
    stop condition instead -- `TieredCompaction` escalates only until the history fits its
    target.  Implementations must preserve tool-call / tool-return pairing.
    """

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
    ) -> list[ModelMessage]: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Safe cutoff logic -- preserves tool-call / tool-return pairs
# ---------------------------------------------------------------------------

_TOOL_PAIR_SEARCH_RANGE = 5
"""Number of messages to search around a cutoff point for tool-call pairs."""


def _is_safe_cutoff(
    messages: list[ModelMessage],
    cutoff: int,
    search_range: int = _TOOL_PAIR_SEARCH_RANGE,
) -> bool:
    """Return True if cutting at *cutoff* does not orphan any tool-call pair.

    A tool-call pair is a ``ToolCallPart`` in a ``ModelResponse`` together with
    the corresponding ``ToolReturnPart`` in a subsequent ``ModelRequest``.  Both
    sides must end up on the same side of the cut.
    """
    if cutoff >= len(messages):
        return True

    start = max(0, cutoff - search_range)
    end = min(len(messages), cutoff + search_range)

    for i in range(start, end):
        msg = messages[i]
        if not isinstance(msg, ModelResponse):
            continue

        call_ids: set[str] = set()
        for part in msg.parts:
            if isinstance(part, ToolCallPart) and part.tool_call_id:
                call_ids.add(part.tool_call_id)

        if not call_ids:
            continue

        for j in range(i + 1, len(messages)):
            later = messages[j]
            if not isinstance(later, ModelRequest):
                continue
            for rpart in later.parts:
                if isinstance(rpart, ToolReturnPart) and rpart.tool_call_id in call_ids:
                    call_before = i < cutoff
                    return_before = j < cutoff
                    if call_before != return_before:
                        return False

    return True


def find_safe_cutoff(messages: list[ModelMessage], keep: int) -> int:
    """Find a cutoff index that keeps *keep* tail messages without splitting tool pairs.

    Returns 0 if trimming is unnecessary (fewer messages than *keep*).
    """
    if keep == 0:
        return len(messages)
    if len(messages) <= keep:
        return 0

    target = len(messages) - keep
    for idx in range(target, -1, -1):
        if _is_safe_cutoff(messages, idx):
            return idx
    return 0  # pragma: no cover


def find_token_cutoff(
    messages: list[ModelMessage],
    target_tokens: int,
    tokenizer: Callable[[str], int] | None = None,
) -> int:
    """Binary-search for a cutoff such that ``messages[cutoff:]`` fits in *target_tokens*.

    Adjusts the result so that no tool-call pairs are orphaned.
    """
    if not messages or estimate_token_count(messages, tokenizer) <= target_tokens:
        return 0

    lo, hi = 0, len(messages)
    candidate = len(messages)

    while lo < hi:
        mid = (lo + hi) // 2
        if estimate_token_count(messages[mid:], tokenizer) <= target_tokens:
            candidate = mid
            hi = mid
        else:
            lo = mid + 1

    if candidate >= len(messages):
        candidate = max(0, len(messages) - 1)  # pragma: no cover

    # Walk backward to a safe point.
    for idx in range(candidate, -1, -1):
        if _is_safe_cutoff(messages, idx):
            return idx
    return 0  # pragma: no cover


# ---------------------------------------------------------------------------
# First user message preservation
# ---------------------------------------------------------------------------


def _is_harness_marker_part(part: ModelRequestPart) -> bool:
    """Return whether a user-role part is harness bookkeeping rather than a user turn."""
    return is_pinned(part) or is_receipt_part(part)


def find_first_user_message(messages: list[ModelMessage]) -> ModelRequest | None:
    """Return the first ``ModelRequest`` that contains a ``UserPromptPart``, or ``None``."""
    for msg in messages:
        if isinstance(msg, ModelRequest) and any(
            isinstance(part, UserPromptPart) and not _is_harness_marker_part(part) for part in msg.parts
        ):
            return msg
    return None


def prepend_first_user_message(
    original: list[ModelMessage],
    cutoff: int,
    trimmed: list[ModelMessage],
) -> list[ModelMessage]:
    """Ensure the first user message from *original* appears in *trimmed*.

    If the first ``ModelRequest`` containing a ``UserPromptPart`` in *original*
    was discarded (its index is before *cutoff*) and is not already in *trimmed*,
    prepend it.
    """
    first = find_first_user_message(original)
    if first is None:
        return trimmed
    idx = original.index(first)
    if idx < cutoff and first not in trimmed:
        return [first, *trimmed]
    return trimmed


# ---------------------------------------------------------------------------
# Tool-pair inspection and in-place clearing
# ---------------------------------------------------------------------------

_CLEARED_TOOL_ARGS = '{}'
"""Replacement for cleared tool-call arguments.

Kept JSON-valid: ``ToolCallPart.args_as_json_str()`` returns a ``str`` arg verbatim, so a
non-JSON placeholder would reach the provider as malformed function arguments.
"""


@dataclass(frozen=True)
class _ToolPair:
    """A matched tool call and its return, with the order the return appeared."""

    tool_call_id: str
    tool_name: str
    call_part: ToolCallPart
    order: int


def iter_tool_pairs(messages: Sequence[ModelMessage]) -> list[_ToolPair]:
    """Return matched tool-call / tool-return pairs in return-appearance order."""
    calls: dict[str, ToolCallPart] = {}
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart) and part.tool_call_id:
                    calls[part.tool_call_id] = part

    pairs: list[_ToolPair] = []
    order = 0
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart) and part.tool_call_id in calls:
                    call = calls[part.tool_call_id]
                    pairs.append(_ToolPair(part.tool_call_id, call.tool_name, call, order))
                    order += 1
    return pairs


def rebuild_with_cleared(
    messages: Sequence[ModelMessage],
    clear_return_ids: set[str],
    clear_input_ids: set[str],
    placeholder: str,
) -> list[ModelMessage]:
    """Return *messages* with selected tool results (and optionally inputs) blanked.

    The ``ToolReturnPart`` / ``ToolCallPart`` are kept in place with placeholder content,
    so tool-call pairing is never broken.  Already-blanked parts are left untouched.
    """
    out: list[ModelMessage] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            request_parts: list[ModelRequestPart] = []
            changed = False
            for part in msg.parts:
                # Exact type, not `isinstance`: typed `ToolReturnPart` subclasses such as
                # `ToolSearchReturnPart` carry structured `TypedDict` content that core re-parses
                # (and trusts to be valid) on every request -- see `parse_discovered_tools`. Blanking
                # it to a string both breaks that invariant (crash next request) and discards
                # discovery state, so only untyped tool results are cleared.
                if (
                    type(part) is ToolReturnPart
                    and part.tool_call_id in clear_return_ids
                    and str(part.content) != placeholder
                ):
                    request_parts.append(replace(part, content=placeholder))
                    changed = True
                else:
                    request_parts.append(part)
            out.append(replace(msg, parts=request_parts) if changed else msg)
        else:
            response_parts: list[ModelResponsePart] = []
            changed = False
            for part in msg.parts:
                if (
                    isinstance(part, ToolCallPart)
                    and part.tool_call_id in clear_input_ids
                    and part.args != _CLEARED_TOOL_ARGS
                ):
                    response_parts.append(replace(part, args=_CLEARED_TOOL_ARGS))
                    changed = True
                else:
                    response_parts.append(part)
            out.append(replace(msg, parts=response_parts) if changed else msg)
    return out
