"""AG-UI 1.0 lifecycle fields, gated for the `>=0.1.10` SDK floor."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ...messages import ModelMessage, ModelResponse
from ...usage import RequestUsage

if TYPE_CHECKING:
    from ag_ui.core import PROTOCOL_VERSION, RunFinishedCancelledOutcome, TokenUsage

    HAS_LIFECYCLE_1_0 = True
else:
    try:
        from ag_ui.core import PROTOCOL_VERSION, RunFinishedCancelledOutcome, TokenUsage

        HAS_LIFECYCLE_1_0 = True
    except ImportError:
        HAS_LIFECYCLE_1_0 = False
        PROTOCOL_VERSION = None

        class RunFinishedCancelledOutcome:
            """Stub for SDKs without the 1.0 cancelled outcome."""

        class TokenUsage:
            """Stub for SDKs without 1.0 token usage."""


__all__ = [
    'HAS_LIFECYCLE_1_0',
    'PROTOCOL_VERSION',
    'RunFinishedCancelledOutcome',
    'TokenUsage',
    'token_usage_from_messages',
]


_MAX_TOKEN_COUNT = 2**53 - 1
"""`TokenUsage` bounds every count to JavaScript's `Number.MAX_SAFE_INTEGER`, the ceiling every binding can encode."""


def _reported(count: int) -> int | None:
    """A count as the protocol reads it: absent when zero, which means "not reported", or outside the wire's range.

    `RequestUsage` doesn't bound its counts, and `TokenUsage` rejects one below zero or above the ceiling
    in its constructor, which would fail the run at its terminal event. Losing the count is strictly
    better than losing the run, as the SDK's own producers reason.
    """
    return count if 0 < count <= _MAX_TOKEN_COUNT else None


def token_usage_from_messages(messages: Sequence[ModelMessage]) -> list[TokenUsage]:
    """Return one `TokenUsage` per `(provider, model)` pair in the messages, in order of first appearance.

    Responses are summed before any count is bounded, so an overflow that only appears in the sum is
    caught; the trade is that a count one response misreports is folded into its pair's total rather
    than dropped alone, which is fine for a report whose job is to survive to the wire. `RequestUsage`
    defaults every count to `0`, and the protocol reads a zero as "reported zero" but an absent count
    as "not reported": a pair with no input or output tokens contributes nothing, and every other zero
    count is left absent rather than claimed as measured. The total is the sum of the input and output
    counts sent, bounded the same way.
    """
    assert HAS_LIFECYCLE_1_0, '`token_usage_from_messages` needs ag-ui-protocol >= 1.0'
    by_model: dict[tuple[str | None, str | None], RequestUsage] = {}
    for message in messages:
        if isinstance(message, ModelResponse):
            # usage-attribution: a fresh per-(provider, model) total for the RUN_FINISHED report, not a run's usage
            by_model.setdefault((message.provider_name, message.model_name), RequestUsage()).incr(message.usage)
    entries: list[TokenUsage] = []
    for (provider, model), usage in by_model.items():
        input_tokens, output_tokens = _reported(usage.input_tokens), _reported(usage.output_tokens)
        if input_tokens is None and output_tokens is None:
            continue
        entries.append(
            TokenUsage(
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=_reported((input_tokens or 0) + (output_tokens or 0)),
                cached_input_tokens=_reported(usage.cache_read_tokens),
                cache_write_input_tokens=_reported(usage.cache_write_tokens),
            )
        )
    return entries
