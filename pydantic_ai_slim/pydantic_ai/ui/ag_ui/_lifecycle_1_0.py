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


def token_usage_from_messages(messages: Sequence[ModelMessage]) -> list[TokenUsage]:
    """Return one `TokenUsage` per `(provider, model)` pair in the messages.

    `RequestUsage` defaults every count to `0`, and the protocol reads a zero as "reported zero" but
    an absent count as "not reported". So a response with no input or output tokens contributes
    nothing, and every other zero count is left absent rather than claimed as measured.
    """
    grouped: dict[tuple[str | None, str | None], RequestUsage] = {}
    for message in messages:
        if not isinstance(message, ModelResponse):
            continue
        usage = message.usage
        if not (usage.input_tokens or usage.output_tokens):
            continue
        # usage-attribution: a fresh per-(provider, model) total for the RUN_FINISHED report, not a run's usage
        grouped.setdefault((message.provider_name, message.model_name), RequestUsage()).incr(usage)

    return [
        TokenUsage(
            provider=provider,
            model=model,
            input_tokens=usage.input_tokens or None,
            output_tokens=usage.output_tokens or None,
            total_tokens=usage.total_tokens,
            reasoning_tokens=usage.details.get('reasoning_tokens') or None,
            cached_input_tokens=usage.cache_read_tokens or None,
            cache_write_input_tokens=usage.cache_write_tokens or None,
        )
        for (provider, model), usage in grouped.items()
    ]
