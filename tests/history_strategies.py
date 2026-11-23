"""Hypothesis strategies for message histories, with every message part kind and provider-shaped server references.

`histories()` draws conversations in which every tool call is answered exactly once, in the request right
after its response, from a run's first request on its own to several tool rounds. With `well_formed=False`
the same conversations are then damaged the ways stored, hand-built and UI-returned histories get damaged,
each kind of damage named by a `Damage` so a test can leave it out with `without`.

`PART_STRATEGIES` names, for every class in `ModelRequestPart` and `ModelResponsePart`, the strategy that draws it
into a history, or why none does. `test_every_part_kind_is_drawn_or_exempt` in `tests/test_history_properties.py` fails
until a new part kind gets an entry, and `test_every_drawn_part_kind_appears` until a `Drawn` kind shows up in a
drawn history.

Server references (response, item and call ids) are drawn in the shapes providers send them, from
`SERVER_REF_SHAPES`, each with the cassette it was read from. `test_server_ref_shapes_match_their_cassettes`
checks each shape still matches a value in its cassette, so a shape is recorded, not invented.

The domain is narrower than what the message types accept. Each narrowing leaves out an input that providers
and users never produce, or one whose shape needs its own strategy:

- Content is text, structured JSON, or a small inline image (`BinaryImage`), never a URL, so drawing and mapping
  a history does no I/O.
- Values are ones a JSON round trip preserves (no NaN or infinities, no tuples), as a stored history has to.
- Text in a well-formed history is never blank. Empty and whitespace-only text comes only from the `empty_*`
  damages, so a failure it causes is tied to a damage a test can name. That includes signed thinking with no
  content, as an OpenAI reasoning item without a summary stores it: `empty_signed_thinking` draws it.
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
- Ids, `provider_details` and `provider_response_id` are drawn only for the providers and fields in
  `SERVER_REF_SHAPES`; elsewhere they stay unset, as they are on parts built by hand.
- `history_digest` is never drawn: only the adapter that continues a response can compute it, and most stored
  histories carry none.
- Native tool parts come in the pairs recorded for OpenAI tool search, in the shape older histories store, and
  xAI web search. Another provider's native parts need a shape recorded from that provider.
- Instruction parts are an `InstructionBaselinePart` opening a request that has `instructions`, and
  `InstructionDeltaPart`s of one named agent block after it, as an agent run records them.
- A request is never made only of records of a change (the delta parts, an `InstructionDeltaPart`): the agent
  sends one with a prompt, a result or feedback.
- Damage that adds a message puts it after a request, never between a call's response and the request
  answering it: the agent loop never puts one there, and repair pairs results with calls by order.
- `empty_parts` never leaves a history with no part and no instructions: the agent refuses to run on one
  (`No message history, user prompt, or instructions provided`). Nor does it empty a response that reuses an
  earlier response's call id, which leaves the shape `reanswered_call_id` draws on its own.
"""

from __future__ import annotations as _annotations

import base64
import json
import os
import string
from dataclasses import dataclass, replace
from typing import Literal, assert_never, get_args

import pydantic_core
import pytest
from hypothesis import HealthCheck, assume, settings, strategies as st
from pydantic import JsonValue

from pydantic_ai._deferred_capabilities import LoadCapabilityCallPart, LoadCapabilityReturnPart
from pydantic_ai._tool_search import (
    NativeToolSearchCallPart,
    NativeToolSearchReturnPart,
    ToolSearchCallPart,
    ToolSearchReturnPart,
)
from pydantic_ai.messages import (
    AgentInstructionSource,
    BaseToolCallPart,
    BinaryImage,
    CapabilityLoadDeltaPart,
    CapabilityPart,
    CompactionPart,
    DataCapabilityPart,
    FilePart,
    InstructionBaselinePart,
    InstructionCapabilityPart,
    InstructionDeltaPart,
    InstructionId,
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    ModelSettingsDeltaPart,
    NativeToolCallPart,
    NativeToolReturnPart,
    OperatorSource,
    RetryFeedbackCause,
    RetryFeedbackPart,
    RetryInfo,
    RetryPromptPart,  # pyright: ignore[reportDeprecated]
    SpeechPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolRetryCause,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from .manifests import Exempt

_long_run_examples = os.environ.get('HISTORY_PROPERTY_EXAMPLES')
PROPERTY_SETTINGS = settings(
    max_examples=int(_long_run_examples) if _long_run_examples else 50,
    derandomize=_long_run_examples is None,
    database=None,
    deadline=None,
    print_blob=_long_run_examples is not None,
    suppress_health_check=[HealthCheck.too_slow],
)
"""A small, derandomized batch by default, so a run is reproducible, fits the per-test budget in `tests/cost_guards.py`, and
covers the same branches on every xdist worker. `HISTORY_PROPERTY_EXAMPLES` sets a long, randomized run to hunt for
new failures, as `REALTIME_SIMULATION_EXAMPLES` does for the realtime simulation. `cost_guards` bounds each test's
time, so Hypothesis's own check on slow drawing, which a loaded worker can trip, is off."""


IGNORE_RANDOM_STATE_NOTE = pytest.mark.filterwarnings(
    'ignore:Do not use the `random` module inside strategies:hypothesis.errors.HypothesisDeprecationWarning'
)
"""For a module that draws these strategies. Hypothesis compares the global `random` state before and after each
draw and notes this deprecation when it changed. Its own `control.py` documents the false positive: a different
thread touching `random` during a draw trips it, and the guard it keeps covers only the states it set itself. Under
xdist a property module shares a worker with tests that run threads, so the note fired on CI as a flaky error
(`filterwarnings = error`). No strategy here uses `random`."""


_ALPHANUMERIC = string.ascii_letters + string.digits
_HEX = '0123456789abcdef'

TOOL_CALL_IDS = st.one_of(
    st.text(_ALPHANUMERIC, min_size=1, max_size=24).map('call_{}'.format),
    st.text(_ALPHANUMERIC, min_size=24, max_size=24).map('toolu_{}'.format),
    st.text(_HEX, min_size=32, max_size=32).map('pyd_ai_{}'.format),
    st.text(_ALPHANUMERIC, min_size=9, max_size=9),
    st.uuids().map(str),
)
"""OpenAI, Anthropic, Pydantic AI's own (for providers that send none), Mistral, and a UUID."""


@dataclass(frozen=True)
class RecordedShape:
    """The shape a provider sends a server reference in, and a cassette that holds one."""

    pattern: str
    """A regular expression every value of the field matches in full."""
    cassette: str
    """A cassette, relative to `tests/`, that holds a value of the field."""


ServerRefApi = Literal['openai-responses', 'openai-chat', 'anthropic']
ServerRefField = Literal['response_id', 'message_id', 'reasoning_id', 'function_call_id', 'tool_search_call_id']

SERVER_REF_SHAPES: dict[tuple[ServerRefApi, ServerRefField], RecordedShape] = {
    ('openai-responses', 'response_id'): RecordedShape(
        r'resp_[0-9a-f]{50}',
        'models/cassettes/test_openai_responses/test_background_mode_streaming_starting_after_vcr.yaml',
    ),
    ('openai-responses', 'message_id'): RecordedShape(
        r'msg_[0-9a-f]{50}',
        'models/cassettes/test_openai_responses/test_openai_responses_always_reasoning_model_resolves_to_accepted_effort[o-series-minimal].yaml',
    ),
    ('openai-responses', 'reasoning_id'): RecordedShape(
        r'rs_[0-9a-f]{50}',
        'models/cassettes/test_openai_responses/test_openai_responses_always_reasoning_model_resolves_to_accepted_effort[o-series-minimal].yaml',
    ),
    ('openai-responses', 'function_call_id'): RecordedShape(
        r'fc_[0-9a-f]{50}',
        'models/cassettes/test_openai_responses/'
        'test_openai_conversation_id_continues_after_usage_limit_without_lookup.yaml',
    ),
    ('openai-responses', 'tool_search_call_id'): RecordedShape(
        r'tsc_[0-9a-f]{50}', 'models/cassettes/test_openai_codex/test_codex_deferred_tool_search.yaml'
    ),
    ('openai-chat', 'response_id'): RecordedShape(
        r'chatcmpl-[A-Za-z0-9]{29}', 'models/cassettes/test_openai/test_audio_as_binary_content_input.yaml'
    ),
    ('anthropic', 'response_id'): RecordedShape(
        r'msg_01[A-Za-z0-9]{22}', 'models/cassettes/test_anthropic/test_anthropic_advisor_tool_stream.yaml'
    ),
}
"""Each server reference the strategies draw, by the API that issues it and the field it fills."""


def server_refs(api: ServerRefApi, field: ServerRefField) -> st.SearchStrategy[str]:
    """Values of `field` in the shape `api` sends them."""
    return st.from_regex(SERVER_REF_SHAPES[api, field].pattern, fullmatch=True)


OPENAI_BASE_URL = 'https://api.openai.com/v1/'
"""Where the mapped OpenAI models send their requests, so a response stamped with it shares their server state."""

_TEXT = st.text(min_size=1).filter(str.strip)
_BLANK = st.text(' \t\n', max_size=3)

_JSON_VALUES: st.SearchStrategy[JsonValue] = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False, allow_infinity=False) | st.text(),
    lambda children: st.lists(children, max_size=3) | st.dictionaries(st.text(), children, max_size=3),
    max_leaves=6,
)
_JSON_OBJECTS = st.dictionaries(st.text(), _JSON_VALUES, max_size=3)
_TOOL_ARGS = (
    _JSON_OBJECTS | _JSON_OBJECTS.map(lambda args: pydantic_core.to_json(args).decode()) | _JSON_OBJECTS.map(json.dumps)
)
"""Arguments as a dict, or as the JSON string a model sent: compact, or spaced as `json.dumps` writes it
(https://github.com/pydantic/pydantic-ai/pull/10410)."""

_TOOL_NAMES = st.from_regex(r'[a-zA-Z][a-zA-Z0-9_-]{0,63}', fullmatch=True)

_PROVIDER_NAMES = st.sampled_from(
    ['openai', 'anthropic', 'google', 'bedrock', 'groq', 'mistral', 'cohere', 'huggingface', 'xai']
)
"""The `system` of each mapped model, so signed thinking is sent back to the provider that made it."""

_TOOL_RETRY_CAUSES: list[ToolRetryCause] = ['validation_error', 'model_retry', 'unknown_tool', 'unavailable_tool']
_FEEDBACK_CAUSES: list[RetryFeedbackCause] = ['validation_error', 'no_output', 'model_retry']
_ANSWERED_OUTCOMES: list[Literal['success', 'failed', 'denied']] = ['success', 'failed', 'denied']
_SYSTEM_PROMPT_SCOPES: list[Literal['conversation', 'request']] = ['conversation', 'request']

_IMAGES = st.builds(
    BinaryImage,
    data=st.binary(min_size=1, max_size=16).map(lambda data: b'\x89PNG\r\n\x1a\n' + data),
    media_type=st.just('image/png'),
)
"""A small PNG, inline, so drawing it does no I/O."""

_USER_PROMPTS = st.builds(UserPromptPart, _TEXT | st.lists(_TEXT, min_size=1, max_size=3))
_RETRY_FEEDBACK = st.builds(
    RetryFeedbackPart, _TEXT, retry=st.sampled_from(_FEEDBACK_CAUSES).map(RetryInfo[RetryFeedbackCause])
)
_MID_CONVERSATION_SYSTEM_PROMPTS = st.builds(SystemPromptPart, _TEXT, scope=st.sampled_from(_SYSTEM_PROMPT_SCOPES))
_METADATA = st.none() | _JSON_OBJECTS

_SIGNATURES = st.binary(min_size=1).map(lambda signature: base64.b64encode(signature).decode())

_THINKING = (
    st.builds(ThinkingPart, _TEXT)
    | st.builds(ThinkingPart, _TEXT, signature=_SIGNATURES, provider_name=_PROVIDER_NAMES)
    | st.builds(
        ThinkingPart,
        _TEXT,
        id=server_refs('openai-responses', 'reasoning_id'),
        signature=st.none() | _SIGNATURES,
        provider_name=st.just('openai'),
    )
)
"""Unsigned thinking, signed thinking from one of the mapped providers, or an OpenAI reasoning item with its id.
Signatures are base64, as every provider sends them."""

_TEXT_PARTS = st.builds(TextPart, _TEXT) | st.builds(
    TextPart, _TEXT, id=server_refs('openai-responses', 'message_id'), provider_name=st.just('openai')
)
"""Text, or an OpenAI Responses output message with its item id."""

_REQUEST_COMPACTIONS = st.builds(CompactionPart, content=_TEXT)
"""A summary opening a request in place of earlier turns, as an application or `SummarizingCompaction` writes it."""

_RESPONSE_COMPACTIONS = st.builds(CompactionPart, content=_TEXT, provider_name=_PROVIDER_NAMES)
"""A summary a provider's compaction returned, without the encrypted item an OpenAI compaction also carries."""

_SPEECH_INPUT = st.builds(SpeechPart, speaker=st.just('user'), transcript=_TEXT)
_SPEECH_OUTPUT = st.builds(SpeechPart, speaker=st.just('assistant'), transcript=_TEXT)
"""A realtime session's turns, as text: `prepare_messages` converts them into prompts and text."""

_FILES = st.builds(FilePart, _IMAGES)
"""An image a model generated."""


@dataclass(repr=False, kw_only=True)
class _BudgetPart(InstructionCapabilityPart, namespace='history_strategies', name='budget'):
    """An instruction a capability adds mid-conversation, in the application's voice."""


@dataclass(repr=False, kw_only=True)
class _ProgressPart(DataCapabilityPart[str], namespace='history_strategies', name='progress', tag='progress_report'):
    """Data a capability inserts, framed on the user channel."""


_CAPABILITY_PARTS = st.builds(_BudgetPart, content=_TEXT, source=st.just(OperatorSource())) | st.builds(
    _ProgressPart, content=_TEXT, source=st.just(OperatorSource())
)

_AGENT_BLOCK = InstructionId(AgentInstructionSource(), name='progress')
"""The named agent instruction block the drawn deltas edit."""

_INSTRUCTION_DELTAS = st.builds(InstructionDeltaPart, id=st.just(_AGENT_BLOCK), content=_TEXT)

_DELTAS = st.one_of(
    st.builds(ToolAvailabilityDeltaPart, tools_added=st.lists(_TOOL_NAMES, min_size=1, max_size=2, unique=True)),
    st.builds(CapabilityLoadDeltaPart, loaded=st.lists(_TOOL_NAMES, min_size=1, max_size=2, unique=True)),
    st.builds(
        ModelSettingsDeltaPart,
        settings=st.sampled_from(['low', 'medium', 'high']).map(lambda effort: {'thinking': effort}),
    ),
)
"""A mid-conversation change to the tools, the loaded capabilities, or the thinking effort."""

_RECORDS = (ToolAvailabilityDeltaPart, CapabilityLoadDeltaPart, ModelSettingsDeltaPart, InstructionDeltaPart)
"""Parts that record a change rather than say something: the agent never sends a request made of them alone."""


def _tool_returns(call: ToolCallPart) -> st.SearchStrategy[ToolReturnPart]:
    if isinstance(call, ToolSearchCallPart):
        return st.builds(
            ToolSearchReturnPart,
            content=st.lists(_TOOL_NAMES, max_size=2, unique=True).map(
                lambda names: {'discovered_tools': [{'name': name} for name in names]}
            ),
            tool_call_id=st.just(call.tool_call_id),
        )
    if isinstance(call, LoadCapabilityCallPart):
        return st.builds(
            LoadCapabilityReturnPart,
            content=_TEXT.map(lambda message: {'message': message}),
            tool_call_id=st.just(call.tool_call_id),
        )
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


def _tool_calls(call_id: str) -> st.SearchStrategy[ToolCallPart]:
    """A function call, an OpenAI function call item with its item id, a local tool search, or a capability load."""
    return st.one_of(
        st.builds(ToolCallPart, _TOOL_NAMES, _TOOL_ARGS, st.just(call_id)),
        st.builds(
            ToolCallPart,
            _TOOL_NAMES,
            _TOOL_ARGS,
            st.just(call_id),
            id=server_refs('openai-responses', 'function_call_id'),
            provider_name=st.just('openai'),
        ),
        st.builds(
            ToolSearchCallPart, args=_TEXT.map(lambda query: {'queries': [query]}), tool_call_id=st.just(call_id)
        ),
        st.builds(
            LoadCapabilityCallPart, args=_TOOL_NAMES.map(lambda name: {'id': name}), tool_call_id=st.just(call_id)
        ),
    )


@st.composite
def _native_pairs(draw: st.DrawFn) -> list[ModelResponsePart]:
    """A native tool call with its result, in a shape a provider recorded.

    OpenAI tool search, as older histories store it: the result carries the output item id, `tso_…`, where newer
    ones carry the call's (`tests/test_tool_search.py`, from https://github.com/pydantic/pydantic-ai/pull/13489).
    xAI web search, under the search function the model ran, which goes back by that name
    (`tests/models/xai/test_native_tool_results.py::test_xai_native_tool_result_replayed_paired`).
    """
    if draw(st.booleans()):
        call_id = draw(server_refs('openai-responses', 'tool_search_call_id'))
        details: dict[str, JsonValue] = {'call_id': None, 'execution': 'server', 'status': 'completed'}
        output_id = call_id.replace('tsc_', 'tso_', 1)
        return [
            NativeToolSearchCallPart(
                provider_name='openai',
                args={'queries': [draw(_TEXT)]},
                tool_call_id=call_id,
                id=call_id,
                provider_details=details,
            ),
            NativeToolSearchReturnPart(
                provider_name='openai',
                content={'discovered_tools': [{'name': draw(_TOOL_NAMES)}]},
                tool_call_id=output_id,
                provider_details={**details, 'id': output_id},
            ),
        ]
    call_id = draw(TOOL_CALL_IDS)
    return [
        NativeToolCallPart(
            'web_search',
            {'query': draw(_TEXT)},
            call_id,
            provider_name='xai',
            provider_details={'function_name': draw(st.sampled_from(['web_search', 'open_page']))},
        ),
        NativeToolReturnPart(
            'web_search',
            None,
            call_id,
            provider_name='xai',
            provider_details={'encrypted_content': draw(_SIGNATURES)},
        ),
    ]


@st.composite
def _responses(draw: st.DrawFn, call_ids: list[str]) -> ModelResponse:
    """A response that calls a tool for each id, and says something when it calls none."""
    calls = [draw(_tool_calls(call_id)) for call_id in call_ids]
    thinking = draw(st.lists(_THINKING, max_size=2))
    texts = draw(st.lists(_TEXT_PARTS, min_size=0 if calls else 1, max_size=2))
    extras = draw(
        st.lists(
            st.one_of(
                _native_pairs(),
                _RESPONSE_COMPACTIONS.map(lambda part: [part]),
                _SPEECH_OUTPUT.map(lambda part: [part]),
                _FILES.map(lambda part: [part]),
            ),
            max_size=1,
        )
    )
    provider_name = draw(st.none() | _PROVIDER_NAMES)
    response_id: str | None = None
    if provider_name == 'openai':
        response_id = draw(
            st.none() | server_refs('openai-responses', 'response_id') | server_refs('openai-chat', 'response_id')
        )
    elif provider_name == 'anthropic':
        response_id = draw(st.none() | server_refs('anthropic', 'response_id'))
    return ModelResponse(
        parts=[*thinking, *texts, *(part for extra in extras for part in extra), *calls],
        usage=draw(st.builds(RequestUsage, input_tokens=st.integers(0, 10**6), output_tokens=st.integers(0, 10**6))),
        model_name=None if provider_name is None else draw(_TEXT),
        provider_name=provider_name,
        provider_response_id=response_id,
        provider_url=draw(st.none() | st.just(OPENAI_BASE_URL)) if response_id is not None else None,
    )


@st.composite
def _follow_up_requests(draw: st.DrawFn, calls: list[ToolCallPart], instructions: str | None) -> ModelRequest:
    """The request after a response: a return for each of its calls, in any order, then any new input."""
    returns = draw(st.permutations([draw(_tool_returns(call)) for call in calls]))
    new_input = draw(
        st.lists(
            _USER_PROMPTS
            | _RETRY_FEEDBACK
            | _MID_CONVERSATION_SYSTEM_PROMPTS
            | _SPEECH_INPUT
            | _CAPABILITY_PARTS
            | _DELTAS
            | (_INSTRUCTION_DELTAS if instructions is not None else st.nothing()),
            min_size=0 if calls else 1,
            max_size=2,
        ).filter(lambda parts: calls or not all(isinstance(part, _RECORDS) for part in parts))
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
    opening: list[ModelRequestPart] = []
    if draw(st.booleans()):
        opening.append(draw(_REQUEST_COMPACTIONS))
    if instructions[0] is not None and draw(st.booleans()):
        opening.append(
            InstructionBaselinePart(
                instructions=[InstructionPart(instructions[0], on_change='append', id=_AGENT_BLOCK)]
            )
        )
    history: list[ModelMessage] = [
        ModelRequest(
            parts=[*opening, *system_prompts, draw(_USER_PROMPTS)],
            instructions=instructions[0],
            metadata=draw(_METADATA),
        )
    ]
    for index, call_count in enumerate(calls_per_response):
        response = draw(_responses([next(call_ids) for _ in range(call_count)]))
        history.append(response)
        # Only a response that calls no tool can end the conversation.
        if call_count or index < len(calls_per_response) - 1 or draw(st.booleans()):
            calls = [part for part in response.parts if isinstance(part, ToolCallPart)]
            history.append(draw(_follow_up_requests(calls, instructions[index + 1])))
    # Native calls draw their ids apart from the local ones, and no provider issues one id twice: the
    # `repeated_call_id` damages draw that.
    call_ids = [
        part.tool_call_id
        for message in history
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, BaseToolCallPart)
    ]
    assume(len(call_ids) == len(set(call_ids)))
    return history


@dataclass(frozen=True)
class Drawn:
    """A part kind the strategies draw into well-formed histories."""

    reason: str
    """Where in a history it is drawn."""


PART_STRATEGIES: dict[type, Drawn | Exempt] = {
    SystemPromptPart: Drawn('opening a run, or mid-conversation with either scope'),
    UserPromptPart: Drawn('opening a run, and as new input'),
    ToolReturnPart: Drawn('answering each function call, with every outcome'),
    ToolSearchReturnPart: Drawn('answering a local tool search'),
    LoadCapabilityReturnPart: Drawn('answering a capability load'),
    RetryFeedbackPart: Drawn('as new input, with every cause'),
    ToolAvailabilityDeltaPart: Drawn('as new input'),
    CapabilityLoadDeltaPart: Drawn('as new input'),
    ModelSettingsDeltaPart: Drawn('as new input, changing the thinking effort'),
    InstructionBaselinePart: Drawn('opening a run that has instructions'),
    InstructionDeltaPart: Drawn('as new input, editing one named agent block'),
    CompactionPart: Drawn('a summary opening a run, and a provider summary in a response'),
    CapabilityPart: Drawn('an instruction and a data capability part, as new input'),
    RetryPromptPart: Exempt(  # pyright: ignore[reportDeprecated]
        'deprecated for v3: building one warns, and the parts it is translated into, a retried `ToolReturnPart` and '
        'a `RetryFeedbackPart`, are drawn'
    ),
    SpeechPart: Drawn('as new input, and in a response'),
    TextPart: Drawn('in a response, with an OpenAI item id or none'),
    ThinkingPart: Drawn('in a response: unsigned, signed, or an OpenAI reasoning item'),
    ToolCallPart: Drawn('in a response: a function call, with an OpenAI item id or none'),
    ToolSearchCallPart: Drawn('in a response, as a local tool search'),
    LoadCapabilityCallPart: Drawn('in a response, as a capability load'),
    NativeToolCallPart: Drawn('in a response, an xAI web search with its result'),
    NativeToolReturnPart: Drawn('in a response, answering an xAI web search'),
    NativeToolSearchCallPart: Drawn('in a response, an OpenAI tool search with its result'),
    NativeToolSearchReturnPart: Drawn('in a response, answering an OpenAI tool search'),
    FilePart: Drawn('in a response, as a generated image'),
}
"""Each part kind in `ModelRequestPart` and `ModelResponsePart`, and how the strategies draw it, or why they don't."""


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
    'empty_signed_thinking',
    'empty_parts',
]
_DAMAGE: list[Damage] = list(get_args(Damage))
_NEEDS_A_RESPONSE: frozenset[Damage] = frozenset(
    {'dangling_call', 'repeated_call_id', 'empty_response_text', 'empty_signed_thinking'}
)


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
            case 'empty_signed_thinking':
                # An OpenAI reasoning item without a summary, or another provider's signed thinking with no text.
                response_index = draw(st.sampled_from(list(responses)))
                response = responses[response_index]
                thinking = ThinkingPart('', signature=draw(_SIGNATURES), provider_name=draw(_PROVIDER_NAMES))
                history[response_index] = replace(response, parts=[thinking, *response.parts])
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
    """Message histories, damaged as described in the module docstring unless `well_formed`.

    `without` names the kinds of damage to leave out, such as the one that draws a known finding.
    """
    return _well_formed_histories() if well_formed else _damaged_histories(without)
