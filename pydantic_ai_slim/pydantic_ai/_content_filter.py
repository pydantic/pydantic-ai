"""The message of the `ContentFilterError` raised for a response the provider filtered."""

from __future__ import annotations as _annotations

from collections.abc import Mapping
from typing import Any


def content_filter_message(provider_details: Mapping[str, Any] | None) -> str:
    """The message of the `ContentFilterError` raised for a response with `finish_reason='content_filter'`.

    It names the provider's own finish or block reason, and then the refusal text a provider like Anthropic sends
    alongside its `'refusal'` stop reason, so neither is lost.
    """
    details = provider_details or {}
    reasons: list[str] = []
    if reason := details.get('finish_reason'):
        reasons.append(f"Finish reason: '{reason}'")
    elif reason := details.get('block_reason'):
        reasons.append(f"Block reason: '{reason}'")
    if refusal := details.get('refusal'):
        reasons.append(f'Refusal: {refusal!r}')
    message = 'Content filter triggered.'
    if reasons:
        message += ' ' + '. '.join(reasons)
    return message
