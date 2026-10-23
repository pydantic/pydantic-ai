"""Hypothesis strategies for text-only message histories.

`histories()` draws conversations in which every tool call is answered exactly once, in the request right
after its response. With `well_formed=False` the same conversations are then damaged the ways stored,
hand-built and UI-returned histories get damaged: orphan returns, dangling calls, a call id repeated within
one response, thinking-only responses, consecutive requests and empty-text parts.

The domain is deliberately narrower than what the message types accept:

- Text only, with no `BinaryContent` or URLs, so drawing and mapping a history does no I/O.
- Values are ones a JSON round trip preserves: no NaN or infinities, no tuples.
- Tool call ids follow the formats providers emit, and are never `''`: a provider that sends no id gets a
  generated `pyd_ai_` one, and an empty id is replaced by a random one when it is mapped, which no
  property about the mapped body can hold.
- Retries use the current parts: a `ToolReturnPart` with `outcome='retried'` for one that answers a
  call, a `RetryFeedbackPart` for one that doesn't. The deprecated `RetryPromptPart` would warn.
"""

from __future__ import annotations as _annotations

import base64
import string
from dataclasses import replace
from typing import Literal, assert_never

import pydantic_core
from hypothesis import strategies as st
from pydantic import JsonValue

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    RetryFeedbackCause,
    RetryFeedbackPart,
    RetryInfo,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolRetryCause,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

_ALPHANUMERIC = string.ascii_letters + string.digits

TOOL_CALL_IDS = st.one_of(
    st.text(_ALPHANUMERIC, min_size=1, max_size=24).map('call_{}'.format),
    st.text(_ALPHANUMERIC, min_size=24, max_size=24).map('toolu_{}'.format),
    st.text('0123456789abcdef', min_size=32, max_size=32).map('pyd_ai_{}'.format),
    st.text(_ALPHANUMERIC, min_size=9, max_size=9),
    st.uuids().map(str),
)
"""OpenAI, Anthropic, Pydantic AI's own (for providers that send none), Mistral, and a UUID."""

_TEXT = st.text(min_size=1)

_JSON_VALUES: st.SearchStrategy[JsonValue] = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False, allow_infinity=False) | st.text(),
    lambda children: st.lists(children, max_size=3) | st.dictionaries(st.text(), children, max_size=3),
    max_leaves=6,
)
_JSON_OBJECTS = st.dictionaries(st.text(), _JSON_VALUES, max_size=3)
_TOOL_ARGS = _JSON_OBJECTS | _JSON_OBJECTS.map(lambda args: pydantic_core.to_json(args).decode())

_TOOL_NAMES = st.from_regex(r'[a-z][a-z0-9_]{0,15}', fullmatch=True)

_PROVIDER_NAMES = st.sampled_from(
    ['openai', 'anthropic', 'google', 'bedrock', 'groq', 'mistral', 'cohere', 'huggingface']
)
"""The `system` of each mapped model, so signed thinking is sent back to the provider that made it."""

_TOOL_RETRY_CAUSES: list[ToolRetryCause] = ['validation_error', 'model_retry', 'unknown_tool', 'unavailable_tool']
_FEEDBACK_CAUSES: list[RetryFeedbackCause] = ['validation_error', 'no_output', 'model_retry']
_ANSWERED_OUTCOMES: list[Literal['success', 'failed', 'denied']] = ['success', 'failed', 'denied']
_SYSTEM_PROMPT_SCOPES: list[Literal['conversation', 'request']] = ['conversation', 'request']

_USER_PROMPTS = st.builds(UserPromptPart, _TEXT | st.lists(_TEXT, min_size=1, max_size=3))
_RETRY_FEEDBACK = st.builds(
    RetryFeedbackPart, _TEXT, retry=st.sampled_from(_FEEDBACK_CAUSES).map(RetryInfo[RetryFeedbackCause])
)
_MID_CONVERSATION_SYSTEM_PROMPTS = st.builds(SystemPromptPart, _TEXT, scope=st.sampled_from(_SYSTEM_PROMPT_SCOPES))
_METADATA = st.none() | _JSON_OBJECTS

_THINKING = st.builds(ThinkingPart, _TEXT) | st.builds(
    ThinkingPart,
    st.text(),
    signature=st.binary(min_size=1).map(lambda signature: base64.b64encode(signature).decode()),
    provider_name=_PROVIDER_NAMES,
)
"""Unsigned thinking, or signed thinking from one of the mapped providers. Signatures are base64, as every
provider sends them, and the content of signed thinking may be empty (an OpenAI reasoning item without a
summary)."""


def _tool_returns(call: ToolCallPart) -> st.SearchStrategy[ToolReturnPart]:
    content = _TEXT | _JSON_OBJECTS | st.lists(_JSON_VALUES, min_size=1, max_size=3)
    answered = st.builds(
        ToolReturnPart,
        st.just(call.tool_name),
        content,
        st.just(call.tool_call_id),
        outcome=st.sampled_from(_ANSWERED_OUTCOMES),
    )
    retried = st.builds(
        ToolReturnPart,
        st.just(call.tool_name),
        content,
        st.just(call.tool_call_id),
        outcome=st.just('retried'),
        retry=st.sampled_from(_TOOL_RETRY_CAUSES).map(RetryInfo[ToolRetryCause]),
    )
    return answered | retried


@st.composite
def _responses(draw: st.DrawFn, call_ids: list[str]) -> ModelResponse:
    """A response that calls a tool for each id, and says something when it calls none."""
    calls = [ToolCallPart(draw(_TOOL_NAMES), draw(_TOOL_ARGS), call_id) for call_id in call_ids]
    thinking = draw(st.lists(_THINKING, max_size=2))
    texts = draw(st.lists(st.builds(TextPart, _TEXT), min_size=0 if calls else 1, max_size=2))
    provider_name = draw(st.none() | _PROVIDER_NAMES)
    return ModelResponse(
        parts=[*thinking, *texts, *calls],
        usage=draw(st.builds(RequestUsage, input_tokens=st.integers(0, 10**6), output_tokens=st.integers(0, 10**6))),
        model_name=None if provider_name is None else draw(_TEXT),
        provider_name=provider_name,
    )


@st.composite
def _follow_up_requests(draw: st.DrawFn, calls: list[ToolCallPart], instructions: str | None) -> ModelRequest:
    """The request after a response: a return for each of its calls, in any order, then any new input."""
    returns = draw(st.permutations([draw(_tool_returns(call)) for call in calls]))
    new_input = draw(
        st.lists(
            _USER_PROMPTS | _RETRY_FEEDBACK | _MID_CONVERSATION_SYSTEM_PROMPTS,
            min_size=0 if calls else 1,
            max_size=2,
        )
    )
    return ModelRequest(parts=[*returns, *new_input], instructions=instructions, metadata=draw(_METADATA))


@st.composite
def _well_formed_histories(draw: st.DrawFn) -> list[ModelMessage]:
    calls_per_response = draw(st.lists(st.integers(0, 3), min_size=1, max_size=4))
    call_ids = iter(
        draw(st.lists(TOOL_CALL_IDS, min_size=sum(calls_per_response), max_size=sum(calls_per_response), unique=True))
    )
    request_count = len(calls_per_response) + 1
    shared_instructions = draw(st.none() | _TEXT)
    instructions = draw(
        st.just([shared_instructions] * request_count)
        | st.lists(st.none() | _TEXT, min_size=request_count, max_size=request_count)
    )
    system_prompts = draw(st.lists(st.builds(SystemPromptPart, _TEXT), max_size=2))
    history: list[ModelMessage] = [
        ModelRequest(
            parts=[*system_prompts, draw(_USER_PROMPTS)], instructions=instructions[0], metadata=draw(_METADATA)
        )
    ]
    for index, call_count in enumerate(calls_per_response):
        response = draw(_responses([next(call_ids) for _ in range(call_count)]))
        history.append(response)
        # Only a response that calls no tool can end the conversation.
        if call_count or index < len(calls_per_response) - 1 or draw(st.booleans()):
            calls = [part for part in response.parts if isinstance(part, ToolCallPart)]
            history.append(draw(_follow_up_requests(calls, instructions[index + 1])))
    return history


Damage = Literal[
    'orphan_return', 'dangling_call', 'repeated_call_id', 'thinking_only_response', 'consecutive_request', 'empty_text'
]
_DAMAGE: list[Damage] = [
    'orphan_return',
    'dangling_call',
    'repeated_call_id',
    'thinking_only_response',
    'consecutive_request',
    'empty_text',
]


@st.composite
def _damaged_histories(draw: st.DrawFn) -> list[ModelMessage]:
    """A well-formed history, damaged. Damage that adds a message puts it after a request, as the run would, so it
    never separates a call's response from the request answering it."""
    history = draw(_well_formed_histories())
    for damage in draw(st.lists(st.sampled_from(_DAMAGE), min_size=1, max_size=3)):
        used_ids = {
            part.tool_call_id
            for message in history
            for part in message.parts
            if isinstance(part, ToolCallPart | ToolReturnPart)
        }
        fresh_id = draw(TOOL_CALL_IDS.filter(lambda call_id: call_id not in used_ids))
        requests = {index: message for index, message in enumerate(history) if isinstance(message, ModelRequest)}
        responses = {index: message for index, message in enumerate(history) if isinstance(message, ModelResponse)}
        request_index, response_index = draw(st.sampled_from(list(requests))), draw(st.sampled_from(list(responses)))
        request, response = requests[request_index], responses[response_index]
        match damage:
            case 'orphan_return':
                # A return whose call is gone, as when context eviction drops the response that made it.
                position = draw(st.integers(0, len(request.parts)))
                orphan = ToolReturnPart(draw(_TOOL_NAMES), draw(_TEXT), fresh_id)
                history[request_index] = replace(
                    request, parts=[*request.parts[:position], orphan, *request.parts[position:]]
                )
            case 'dangling_call':
                # A call the run never answered, as when it was cancelled mid-tool.
                history[response_index] = replace(
                    response, parts=[*response.parts, ToolCallPart(draw(_TOOL_NAMES), {}, fresh_id)]
                )
            case 'repeated_call_id':
                # A model that numbers its calls per response (`call_0`, ...) can repeat an id in one response.
                existing_ids = [part.tool_call_id for part in response.parts if isinstance(part, ToolCallPart)]
                call_id = draw(st.sampled_from(existing_ids)) if existing_ids else fresh_id
                repeats = [ToolCallPart(draw(_TOOL_NAMES), {}, call_id) for _ in range(1 if existing_ids else 2)]
                history[response_index] = replace(response, parts=[*response.parts, *repeats])
                answered_by = history[response_index + 1] if response_index + 1 < len(history) else None
                if isinstance(answered_by, ModelRequest) and draw(st.booleans()):
                    # Both calls answered, as `end_strategy='graceful'` can record (#9815).
                    extra_return = ToolReturnPart(repeats[0].tool_name, draw(_TEXT), call_id)
                    history[response_index + 1] = replace(answered_by, parts=[extra_return, *answered_by.parts])
            case 'thinking_only_response':
                # A response with nothing but thinking, as an interrupted stream or a `no_output` retry leaves.
                history.insert(request_index + 1, ModelResponse(parts=[draw(st.builds(ThinkingPart, _TEXT))]))
            case 'consecutive_request':
                history.insert(request_index + 1, ModelRequest(parts=[draw(_USER_PROMPTS)]))
            case 'empty_text':
                empty: ModelRequestPart = draw(st.sampled_from([UserPromptPart(''), SystemPromptPart('')]))
                history[request_index] = replace(request, parts=[*request.parts, empty])
                history[response_index] = replace(response, parts=[*response.parts, TextPart('')])
            case _:
                assert_never(damage)
    return history


def histories(*, well_formed: bool = True) -> st.SearchStrategy[list[ModelMessage]]:
    """Text-only message histories, damaged as described in the module docstring unless `well_formed`."""
    return _well_formed_histories() if well_formed else _damaged_histories()
