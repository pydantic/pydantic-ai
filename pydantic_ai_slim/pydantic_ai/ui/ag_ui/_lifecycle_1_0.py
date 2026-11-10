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


__all__ = [
    'HAS_LIFECYCLE_1_0',
    'PROTOCOL_VERSION',
    'RunFinishedCancelledOutcome',
    'token_usage_from_messages',
]


_MAX_TOKEN_COUNT = 2**53 - 1
"""`TokenUsage` bounds every count to JavaScript's `Number.MAX_SAFE_INTEGER`, the ceiling every binding can encode."""


def _in_range(count: int) -> bool:
    """`TokenUsage` rejects an out-of-range count in its constructor, which would fail the run at its last event."""
    return 0 <= count <= _MAX_TOKEN_COUNT


def _reported(count: int) -> int | None:
    """A count as `TokenUsage` takes it: absent when zero ("not reported") or out of range."""
    return count if count and _in_range(count) else None


def token_usage_from_messages(messages: Sequence[ModelMessage]) -> list[TokenUsage]:
    """One `TokenUsage` per `(provider, model)` pair in the messages, in order of first appearance.

    Counts are summed per pair before they are bounded, so an overflow that only shows in the sum is caught.
    A pair with neither input nor output tokens is left out.
    """
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
                # A total built from a dropped count would be wrong rather than missing.
                total_tokens=_reported(usage.input_tokens + usage.output_tokens)
                if _in_range(usage.input_tokens) and _in_range(usage.output_tokens)
                else None,
                reasoning_tokens=_reported(usage.output_reasoning_tokens),
                cached_input_tokens=_reported(usage.cache_read_tokens),
                cache_write_input_tokens=_reported(usage.cache_write_tokens),
            )
        )
    return entries
