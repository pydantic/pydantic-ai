"""Hypothesis strategies for text-only message histories.

`histories()` draws conversations in which every tool call is answered exactly once, in the request right
after its response, from a run's first request on its own to several tool rounds. With `well_formed=False`
the same conversations are then damaged the ways stored, hand-built and UI-returned histories get damaged,
each kind of damage named by a `Damage` so a test can leave it out with `without`.

The domain is narrower than what the message types accept. Each narrowing leaves out an input that providers
and users never produce, or one whose shape needs its own strategy:

- Text only, with no `BinaryContent` or URLs, so drawing and mapping a history does no I/O.
- Values are ones a JSON round trip preserves (no NaN or infinities, no tuples), as a stored history has to.
- Text in a well-formed history is never blank. Empty and whitespace-only text comes only from the `empty_*`
  damages, so a failure it causes is tied to a damage a test can name.
- Tool call ids follow the formats providers emit, and are never `''`. A provider that sends none gets a
  generated `pyd_ai_` id, and an empty id is replaced by a random one at mapping, which no property about
  the body can hold. No provider is known to send one.
- Tool names match `[a-zA-Z][a-zA-Z0-9_-]{0,63}`, what providers accept.
- Retries use the current parts: a `ToolReturnPart` with `outcome='retried'` for one that answers a call, a
  `RetryFeedbackPart` for one that doesn't. The deprecated `RetryPromptPart` warns, and doesn't round-trip.
- `RetryFeedbackPart.content` is a string. Its `list[ErrorDetails]` form carries pydantic-core's error shape
  (`loc` tuples, any `input`), which needs its own strategy.
- Thinking signatures are base64, as every provider sends them (Google decodes them), and signed thinking
  names one of the mapped models' providers, so it is sent back to the provider that made it.
- A response has a `model_name` only with a `provider_name`: it came from a provider, or was built by hand,
  which is what `repair_messages` tells apart when it merges responses.
- Parts have no `id` or `provider_details`, and responses no `provider_response_id`: their shapes are
  provider-specific, and drawing them needs a strategy per provider.
- Damage that adds a message puts it after a request, never between a call's response and the request
  answering it: the agent loop never puts one there, and repair pairs results with calls by order.
- `empty_parts` never leaves a history with no part and no instructions: the agent refuses to run on one
  (`No message history, user prompt, or instructions provided`). Nor does it empty a response that reuses an
  earlier response's call id, which leaves the shape `reanswered_call_id` draws on its own.
"""

from __future__ import annotations as _annotations

import base64
import string
from dataclasses import replace
from typing import Literal, assert_never, get_args

import pydantic_core
from hypothesis import strategies as st
from pydantic import JsonValue

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
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

_TEXT = st.text(min_size=1).filter(str.strip)
_BLANK = st.text(' \t\n', max_size=3)

_JSON_VALUES: st.SearchStrategy[JsonValue] = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False, allow_infinity=False) | st.text(),
    lambda children: st.lists(children, max_size=3) | st.dictionaries(st.text(), children, max_size=3),
    max_leaves=6,
)
_JSON_OBJECTS = st.dictionaries(st.text(), _JSON_VALUES, max_size=3)
_TOOL_ARGS = _JSON_OBJECTS | _JSON_OBJECTS.map(lambda args: pydantic_core.to_json(args).decode())

_TOOL_NAMES = st.from_regex(r'[a-zA-Z][a-zA-Z0-9_-]{0,63}', fullmatch=True)

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
    content = _TEXT | _JSON_OBJECTS | st.lists(_JSON_VALUES, max_size=3)
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
    # No response at all is the history a run's first request is mapped from.
    calls_per_response = draw(st.lists(st.integers(0, 3), max_size=4))
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
    'orphan_return',
    'dangling_call',
    'repeated_call_id',
    'repeated_call_id_across_responses',
    'reanswered_call_id',
    'thinking_only_response',
    'consecutive_request',
    'empty_prompt',
    'empty_system_prompt',
    'empty_response_text',
    'empty_parts',
]
_DAMAGE: list[Damage] = list(get_args(Damage))
_NEEDS_A_RESPONSE: frozenset[Damage] = frozenset({'dangling_call', 'repeated_call_id', 'empty_response_text'})


@st.composite
def _damaged_histories(draw: st.DrawFn, without: frozenset[Damage]) -> list[ModelMessage]:
    """A well-formed history, damaged. Damage that adds a message puts it after a request, as the run would, so it
    never separates a call's response from the request answering it."""
    history = draw(_well_formed_histories())
    for _ in range(draw(st.integers(1, 3))):
        used_ids = {
            part.tool_call_id
            for message in history
            for part in message.parts
            if isinstance(part, ToolCallPart | ToolReturnPart)
        }
        fresh_id = draw(TOOL_CALL_IDS.filter(lambda call_id: call_id not in used_ids))
        requests = {index: message for index, message in enumerate(history) if isinstance(message, ModelRequest)}
        responses = {index: message for index, message in enumerate(history) if isinstance(message, ModelResponse)}
        # Each response a request answers, with each call id of a response before it that it doesn't call itself,
        # so the id repeats only across responses.
        reusable_ids = [
            (later, part.tool_call_id)
            for later, later_response in responses.items()
            if later + 1 in requests
            for earlier, response in responses.items()
            if earlier < later
            for part in response.parts
            if isinstance(part, ToolCallPart)
            and not any(
                isinstance(call, ToolCallPart) and call.tool_call_id == part.tool_call_id
                for call in later_response.parts
            )
        ]
        # Each request right after a response, with each return of an earlier request whose call that response
        # doesn't make.
        reanswerable = [
            (index, part)
            for index in requests
            if index - 1 in responses
            for earlier in requests
            if earlier < index - 1
            for part in requests[earlier].parts
            if isinstance(part, ToolReturnPart)
            and not any(
                isinstance(call, ToolCallPart) and call.tool_call_id == part.tool_call_id
                for call in responses[index - 1].parts
            )
        ]
        earlier_call_ids = {
            index: {
                part.tool_call_id
                for earlier, response in responses.items()
                if earlier < index
                for part in response.parts
                if isinstance(part, ToolCallPart)
            }
            for index in responses
        }
        # Each message that can lose its parts and leave the run something to send, other than a response that
        # reuses an earlier response's call id.
        emptiable = [
            index
            for index in range(len(history))
            if (
                any(request.instructions for request in requests.values())
                or any(message.parts for other, message in enumerate(history) if other != index)
            )
            and not any(
                isinstance(part, ToolCallPart) and part.tool_call_id in earlier_call_ids.get(index, set())
                for part in history[index].parts
            )
        ]
        applicable: list[Damage] = [
            kind
            for kind in _DAMAGE
            if kind not in without
            and (responses or kind not in _NEEDS_A_RESPONSE)
            and (reusable_ids or kind != 'repeated_call_id_across_responses')
            and (reanswerable or kind != 'reanswered_call_id')
            and (emptiable or kind != 'empty_parts')
        ]
        damage = draw(st.sampled_from(applicable))
        request_index = draw(st.sampled_from(list(requests)))
        request = requests[request_index]
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
                response_index = draw(st.sampled_from(list(responses)))
                response = responses[response_index]
                history[response_index] = replace(
                    response, parts=[*response.parts, ToolCallPart(draw(_TOOL_NAMES), {}, fresh_id)]
                )
            case 'repeated_call_id':
                # An id repeated in one response, as `end_strategy='graceful'` can record (#9815).
                response_index = draw(st.sampled_from(list(responses)))
                response = responses[response_index]
                existing_ids = [part.tool_call_id for part in response.parts if isinstance(part, ToolCallPart)]
                call_id = draw(st.sampled_from(existing_ids)) if existing_ids else fresh_id
                repeats = [ToolCallPart(draw(_TOOL_NAMES), {}, call_id) for _ in range(1 if existing_ids else 2)]
                history[response_index] = replace(response, parts=[*response.parts, *repeats])
                answered_by = history[response_index + 1] if response_index + 1 < len(history) else None
                if isinstance(answered_by, ModelRequest) and draw(st.booleans()):
                    # Every call with the id answered.
                    extra_returns = [ToolReturnPart(repeat.tool_name, draw(_TEXT), call_id) for repeat in repeats]
                    history[response_index + 1] = replace(answered_by, parts=[*extra_returns, *answered_by.parts])
            case 'repeated_call_id_across_responses':
                # An id an earlier response used, answered again, as `TestModel` records when it calls an output tool
                # again after a retry: it names both calls `pyd_ai_tool_call_id__{name}`.
                response_index, call_id = draw(st.sampled_from(reusable_ids))
                response = responses[response_index]
                call = ToolCallPart(draw(_TOOL_NAMES), {}, call_id)
                history[response_index] = replace(response, parts=[*response.parts, call])
                answered_by = requests[response_index + 1]
                answer = ToolReturnPart(call.tool_name, draw(_TEXT), call_id)
                history[response_index + 1] = replace(answered_by, parts=[answer, *answered_by.parts])
            case 'reanswered_call_id':
                # A return for a call an earlier request already answered, after a response that doesn't make that
                # call, as dropping the later call of a reused id (`TestModel` reuses output tool ids) leaves.
                answering_index, answered = draw(st.sampled_from(reanswerable))
                answering = requests[answering_index]
                again = ToolReturnPart(answered.tool_name, draw(_TEXT), answered.tool_call_id)
                history[answering_index] = replace(answering, parts=[again, *answering.parts])
            case 'thinking_only_response':
                # A response with nothing but thinking, as an interrupted stream or a `no_output` retry leaves.
                history.insert(request_index + 1, ModelResponse(parts=[draw(st.builds(ThinkingPart, _TEXT))]))
            case 'consecutive_request':
                history.insert(request_index + 1, ModelRequest(parts=[draw(_USER_PROMPTS)]))
            case 'empty_prompt':
                # A prompt with no text, only whitespace, or no items, as `agent.run('')` or `agent.run([])` records.
                kept = [part for part in request.parts if not isinstance(part, UserPromptPart)]
                prompt = UserPromptPart(draw(_BLANK | st.lists(_TEXT, max_size=0)))
                history[request_index] = replace(request, parts=[*kept, prompt])
            case 'empty_system_prompt':
                # A system prompt that rendered to nothing, or to whitespace.
                history[request_index] = replace(request, parts=[*request.parts, SystemPromptPart(draw(_BLANK))])
            case 'empty_response_text':
                response_index = draw(st.sampled_from(list(responses)))
                response = responses[response_index]
                history[response_index] = replace(response, parts=[*response.parts, TextPart(draw(_BLANK))])
            case 'empty_parts':
                # A message left with no parts, as a history processor that filters parts out (thinking, say) leaves.
                message_index = draw(st.sampled_from(emptiable))
                history[message_index] = replace(history[message_index], parts=[])
            case _:
                assert_never(damage)
    return history


def histories(
    *, well_formed: bool = True, without: frozenset[Damage] = frozenset()
) -> st.SearchStrategy[list[ModelMessage]]:
    """Text-only message histories, damaged as described in the module docstring unless `well_formed`.

    `without` names the kinds of damage to leave out, such as the one that draws a known finding.
    """
    return _well_formed_histories() if well_formed else _damaged_histories(without)
