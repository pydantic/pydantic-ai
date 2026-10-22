"""The context size of a model response, which its usage overstates when a provider SDK merged it from several requests.

google-genai continues a long decode in further requests, each sending the output so far back as input, and sums their
usage. That sum is what the requests cost, but overstates the context the model was given. The model stamps the first
request's input tokens on the response, and context-size readers (`RunContext.context_window_used`, the per-request
input token limit, harness compaction) read them back here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import _utils
from .messages import (
    _PYDANTIC_AI_METADATA_KEY,  # pyright: ignore[reportPrivateUsage]
    ModelResponse,
)

if TYPE_CHECKING:
    from .models import StreamedResponse

__all__ = ('context_input_tokens', 'context_tokens', 'mark_initial_input_tokens')

# Unlike the continuation replace marker in `models/_continuation.py`, this one persists into history, where
# context-size estimates read it back.
_INITIAL_INPUT_TOKENS_KEY = 'initial_input_tokens'


def mark_initial_input_tokens(response: ModelResponse | StreamedResponse, input_tokens: int) -> None:
    """Record the input tokens of the first of the requests a provider SDK merged `response` from."""
    metadata = response.metadata or {}
    metadata.setdefault(_PYDANTIC_AI_METADATA_KEY, {})[_INITIAL_INPUT_TOKENS_KEY] = input_tokens
    response.metadata = metadata


def context_input_tokens(response: ModelResponse) -> int:
    """The input tokens of `response` as a measure of context size, for context-window checks rather than cost.

    That is `usage.input_tokens`, except for a response a provider SDK merged from several requests (see
    `mark_initial_input_tokens`), where it is the input tokens of the first request.
    """
    namespace = (response.metadata or {}).get(_PYDANTIC_AI_METADATA_KEY)
    if _utils.is_str_dict(namespace) and isinstance(input_tokens := namespace.get(_INITIAL_INPUT_TOKENS_KEY), int):
        return input_tokens
    return response.usage.input_tokens


def context_tokens(response: ModelResponse) -> int:
    """The context `response` leaves behind: its input tokens, as `context_input_tokens` reads them, plus its output."""
    return context_input_tokens(response) + response.usage.output_tokens
