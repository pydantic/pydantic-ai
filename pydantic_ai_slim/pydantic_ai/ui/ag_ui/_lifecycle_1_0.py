"""AG-UI 1.0 lifecycle fields, gated for the `>=0.1.10` SDK floor."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ...messages import ModelMessage, ModelResponse

if TYPE_CHECKING:
    from ag_ui.core import PROTOCOL_VERSION, RunFinishedCancelledOutcome, TokenUsage, aggregate_token_usage

    HAS_LIFECYCLE_1_0 = True
else:
    try:
        from ag_ui.core import PROTOCOL_VERSION, RunFinishedCancelledOutcome, TokenUsage, aggregate_token_usage

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
    nothing, and every other zero count is left absent rather than claimed as measured. Only called
    when `HAS_LIFECYCLE_1_0`.
    """
    entries: list[TokenUsage] = []
    for message in messages:
        if not isinstance(message, ModelResponse):
            continue
        usage = message.usage
        if not (usage.input_tokens or usage.output_tokens):
            continue
        entries.append(
            TokenUsage(
                provider=message.provider_name,
                model=message.model_name,
                input_tokens=usage.input_tokens or None,
                output_tokens=usage.output_tokens or None,
                total_tokens=usage.total_tokens,
                reasoning_tokens=usage.details.get('reasoning_tokens') or None,
                cached_input_tokens=usage.cache_read_tokens or None,
                cache_write_input_tokens=usage.cache_write_tokens or None,
            )
        )
    return aggregate_token_usage(entries)
