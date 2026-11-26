"""The context size of a model response, which its usage overstates when it sums several model requests.

Some responses sum the usage of several requests to the model: google-genai continues a long decode in further
requests, each sending the output so far back as input, and Anthropic samples the model once per round of a
server-side tool loop and once more for a compaction. That sum is what the requests cost, but overstates the context
the model was given. The model stamps the context size on the response, and context-size readers
(`RunContext.context_window_used`, the per-request input token limit, harness compaction) read it back here.

A response from `AnthropicModel.compact_messages` is stamped with zero input tokens: its usage is the conversation it
summarized, but the context it leaves behind is the summary, its output.
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

__all__ = ('context_input_tokens', 'context_tokens', 'mark_context_input_tokens', 'marked_context_input_tokens')

# Unlike the continuation replace marker in `models/_continuation.py`, this one persists into history, where
# context-size estimates read it back.
_CONTEXT_INPUT_TOKENS_KEY = 'context_input_tokens'
# The key v2.56.0 stamped Google long decodes with, still read from the histories it saved.
_INITIAL_INPUT_TOKENS_KEY = 'initial_input_tokens'


def mark_context_input_tokens(response: ModelResponse | StreamedResponse, input_tokens: int) -> None:
    """Record the input tokens that measure the context of `response`, when its usage sums several model requests."""
    metadata = response.metadata or {}
    metadata.setdefault(_PYDANTIC_AI_METADATA_KEY, {})[_CONTEXT_INPUT_TOKENS_KEY] = input_tokens
    response.metadata = metadata


def marked_context_input_tokens(response: ModelResponse | StreamedResponse) -> int | None:
    """The input tokens `mark_context_input_tokens` recorded on `response`, or `None` if it has none."""
    namespace = (response.metadata or {}).get(_PYDANTIC_AI_METADATA_KEY)
    if not _utils.is_str_dict(namespace):
        return None
    for key in (_CONTEXT_INPUT_TOKENS_KEY, _INITIAL_INPUT_TOKENS_KEY):
        if isinstance(input_tokens := namespace.get(key), int):
            return input_tokens
    return None


def context_input_tokens(response: ModelResponse) -> int:
    """The input tokens of `response` as a measure of context size, for context-window checks rather than cost.

    That is `usage.input_tokens`, except for a response whose usage sums several model requests (see
    `mark_context_input_tokens`): the first request's input tokens for a Google long decode, the last sampling pass's
    for an Anthropic response that reported its passes.
    """
    marked = marked_context_input_tokens(response)
    return response.usage.input_tokens if marked is None else marked


def context_tokens(response: ModelResponse) -> int:
    """The context `response` leaves behind: its input tokens, as `context_input_tokens` reads them, plus its output."""
    return context_input_tokens(response) + response.usage.output_tokens
