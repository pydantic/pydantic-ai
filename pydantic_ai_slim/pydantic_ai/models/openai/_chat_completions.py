"""Chat Completions response types relaxed for OpenAI-compatible providers, finish reasons, and the vendor data the mapping keeps."""

from __future__ import annotations as _annotations

from typing import Annotated, Any, Literal, cast

from openai.types import chat
from openai.types.chat import ChatCompletionChunk, chat_completion, chat_completion_chunk
from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall
from pydantic import Field, field_validator

from ... import _utils
from ...messages import FinishReason
from ._response_fields import map_logprobs

CHAT_FINISH_REASON_MAP: dict[
    Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call'], FinishReason
] = {
    'stop': 'stop',
    'length': 'length',
    'tool_calls': 'tool_call',
    'content_filter': 'content_filter',
    'function_call': 'tool_call',
}


def split_content_parts(content: object) -> tuple[object, list[object]]:
    """Split a list of content parts into the string the Chat Completions API specifies, and the parts that aren't text.

    Some OpenAI-compatible servers (Mistral models served by the Albert API, for one) return `content` as a list of
    parts like `[{'type': 'text', 'text': '...'}]`, especially when the answer carries references. The text parts are
    joined; the others, like those references, are returned as they came, for `openai_include_unmapped_response_data`.
    Anything that isn't a list is returned unchanged; a list with no text becomes `None`.
    """
    if not isinstance(content, list):
        return content, []
    texts: list[str] = []
    others: list[object] = []
    for part in cast(list[object], content):
        if isinstance(part, str):
            texts.append(part)
        elif _utils.is_str_dict(part) and part.get('type') == 'text' and isinstance(text := part.get('text'), str):
            texts.append(text)
        else:
            others.append(part)
    return ''.join(texts) or None, others


_MAPPED_MESSAGE_EXTRA_FIELDS = frozenset({'reasoning', 'reasoning_content', 'annotations'})
"""Vendor fields on a message or delta that the Chat Completions mapping reads, besides the profile's
`openai_chat_thinking_field`. (`annotations` is declared on a message, but arrives as an extra on a delta.)

Thinking is mapped from the first of the thinking fields with a non-empty string, so the others are read but not
kept: in practice they carry the same text, as vLLM sends both `reasoning` and `reasoning_content`."""


def unmapped_extra_fields(obj: object, mapped: frozenset[str] = frozenset()) -> dict[str, Any]:
    """The fields of an OpenAI SDK object that its schema doesn't declare and the mapping doesn't read, as received."""
    extra: dict[str, Any] = getattr(obj, 'model_extra', None) or {}
    return {key: value for key, value in extra.items() if key not in mapped}


def raw_non_text_content_parts(response: chat.ChatCompletion) -> list[object]:
    """The non-text parts of a completion's list `content`, read before validation joins it into its text.

    The SDK doesn't validate the response, so a choice can lack a `message` here; validation reports that later.
    """
    message = getattr(response.choices[0], 'message', None) if response.choices else None
    return split_content_parts(getattr(message, 'content', None))[1]


def unmapped_choice_data(
    response: object,
    choice: object,
    message: object,
    content_parts: list[object],
    *,
    message_key: str,
    thinking_field: str | None,
) -> dict[str, Any]:
    """The data of one validated completion or chunk that the mapping doesn't read, keyed by where it came from.

    Read from the validated objects, so fields a subclass's response type declares (and maps) aren't extras.
    """
    mapped_message_fields = _MAPPED_MESSAGE_EXTRA_FIELDS | frozenset([thinking_field] if thinking_field else [])
    data = {
        'response': unmapped_extra_fields(response),
        'choice': unmapped_extra_fields(choice),
        message_key: unmapped_extra_fields(message, mapped_message_fields),
        'content_parts': content_parts,
    }
    return {key: value for key, value in data.items() if value}


class _ChatCompletionMessage(chat.ChatCompletionMessage):
    @field_validator('content', mode='before')
    @classmethod
    def _join_content_parts(cls, value: object) -> object:
        return split_content_parts(value)[0]


class _ChatCompletionChoice(chat_completion.Choice):
    finish_reason: Annotated[str, Field(min_length=1)]  # type: ignore[reportIncompatibleVariableOverride]
    """OpenAI-compatible providers can return arbitrary `finish_reason` values (e.g. `"guardrail_intervened"`)."""

    message: _ChatCompletionMessage  # type: ignore[reportIncompatibleVariableOverride]
    """OpenAI-compatible providers can return `content` as a list of parts rather than a string."""


class LenientChatCompletion(chat.ChatCompletion):
    """Relaxes validation on fields that OpenAI-compatible providers may return non-standard values for."""

    model_config = {'title': 'ChatCompletion'}

    choices: list[_ChatCompletionChoice]  # type: ignore[reportIncompatibleVariableOverride]

    service_tier: str | None = None  # type: ignore[reportIncompatibleVariableOverride]
    """OpenAI-compatible providers can return arbitrary `service_tier` values (e.g. `"standard"`, `"on_demand"`)."""

    metadata: dict[str, Any] | None = None
    """OpenAI-compatible providers can return non-string `metadata` values (e.g. Nebius returns a list under `weight_versions`)."""


class LenientChatCompletionChunk(ChatCompletionChunk):
    """Relaxes strict Literal validation on fields that OpenAI-compatible providers may return non-standard values for."""

    model_config = {'title': 'ChatCompletionChunk'}

    service_tier: str | None = None  # type: ignore[reportIncompatibleVariableOverride]
    """OpenAI-compatible providers can return arbitrary `service_tier` values (e.g. `"standard"`, `"on_demand"`)."""


def map_provider_details(
    choice: chat_completion_chunk.Choice | chat_completion.Choice,
) -> dict[str, Any] | None:
    provider_details: dict[str, Any] = {}

    # Add logprobs to provider_details if available
    if choice.logprobs is not None and choice.logprobs.content:
        provider_details['logprobs'] = map_logprobs(choice.logprobs.content)
    if raw_finish_reason := choice.finish_reason:
        provider_details['finish_reason'] = raw_finish_reason

    return provider_details or None


def map_tool_call_provider_details(
    tool_call: ChatCompletionMessageFunctionToolCall | chat_completion_chunk.ChoiceDeltaToolCall,
) -> dict[str, Any] | None:
    # Gemini 3 puts each tool call's thought signature in this vendor extension, and rejects the next request unless
    # it's sent back unchanged:
    # https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/thinking/thought-signatures
    if extra_content := (tool_call.model_extra or {}).get('extra_content'):
        return {'extra_content': extra_content}
    return None
