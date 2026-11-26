"""The digest of the history a stored response was created from, which decides whether a request continues from it."""

from __future__ import annotations as _annotations

from dataclasses import replace

from ...messages import ModelResponse
from .._history_digest import Hasher, continuation_digest


def with_history_digest(response: ModelResponse, history: Hasher | None) -> ModelResponse:
    """`response` with its digest recorded, when `history` says a later request may continue from it."""
    if history is None:
        return response
    return replace(response, history_digest=continuation_digest(history, response.parts))
