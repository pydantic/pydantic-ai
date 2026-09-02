"""Tests for the Gemini Live realtime provider, all network-free."""

from __future__ import annotations as _annotations

import asyncio
import gc
import random
import re
import weakref
from collections.abc import AsyncIterator, MutableMapping, Sequence
from contextlib import AbstractAsyncContextManager
from contextvars import ContextVar
from types import SimpleNamespace
from typing import Any, Literal, cast

import anyio
import httpx
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UserError
from pydantic_ai.messages import (
    BinaryAudio,
    BinaryContent,
    BinaryImage,
    CachePoint,
    CompactionPart,
    FilePart,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    PartEndEvent,
    PartStartEvent,
    RealtimeSessionErrorEvent,
    RetryPromptPart,
    SpeechPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import CodeExecutionTool, ImageGenerationTool, WebFetchTool, WebSearchTool
from pydantic_ai.realtime import (
    RealtimeModelProfile,
    RealtimeModelSettings,
    RealtimeResponseInterruptedEvent,
    RealtimeSession,
    RealtimeSessionReconnectEvent,
    RealtimeTurnCompleteEvent,
)
from pydantic_ai.realtime.codec import (
    AudioDelta,
    InputTranscript,
    OutputTranscript,
    ResponseDone,
    SessionUsage,
    ToolCall,
    ToolCallCancelled,
    ToolResult,
)
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from ..conftest import IsDatetime, IsSameStr, IsStr, try_import
from .test_session import FakeRealtimeModel, make_tool_manager

with try_import() as imports_successful:
    from google.genai import Client, errors as genai_errors, types as genai_types
    from google.genai.live import AsyncSession, ConnectionClosed
    from websockets.exceptions import WebSocketException

    from pydantic_ai.models import google as model_google
    from pydantic_ai.providers.gateway import gateway_provider
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.realtime import google as rt_google
    from pydantic_ai.realtime.google import (
        GoogleRealtimeConnection,
        GoogleRealtimeModel,
        GoogleRealtimeModelSettings,
    )


pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
]


def test_google_public_exports_are_curated() -> None:
    assert rt_google.__all__ == (
        'GoogleRealtimeModel',
        'GoogleRealtimeModelSettings',
        'GoogleRealtimeConnection',
        'AutomaticVAD',
        'MultiSpeaker',
        'ContextCompression',
    )


_GOOGLE_API_URL = 'https://generativelanguage.googleapis.com/'


def _connect(
    model: GoogleRealtimeModel,
    instructions: str,
    *,
    messages: Sequence[ModelMessage] | None = None,
    model_settings: RealtimeModelSettings | None = None,
) -> AbstractAsyncContextManager[GoogleRealtimeConnection]:
    return model.connect(
        messages=[*(messages or ()), ModelRequest(parts=[], instructions=instructions)],
        model_settings=model_settings,
        model_request_parameters=ModelRequestParameters(),
    )


class _RecordingSession:
    """A fake `AsyncSession` that records sends and replays messages turn-by-turn.

    `receive()` mirrors the real SDK: each call yields one turn's messages and then returns; once
    the scripted turns run out it raises (defaulting to `ConnectionClosed`, as the live session does
    when the server closes the socket), so a `while`-loop over `receive()` terminates.
    """

    def __init__(self, turns: list[list[Any]] | None = None, *, close_exc: Exception | None = None) -> None:
        self._turns = list(turns or [])
        self._turn = 0
        self._close_exc = close_exc or ConnectionClosed(None, None)
        self.realtime: list[dict[str, Any]] = []
        self.tool_responses: list[Any] = []
        self.client_content: list[dict[str, Any]] = []

    async def send_realtime_input(self, **kwargs: Any) -> None:
        self.realtime.append(kwargs)

    async def send_client_content(self, *, turns: Any = None, turn_complete: bool = True) -> None:
        self.client_content.append({'turns': turns, 'turn_complete': turn_complete})

    async def send_tool_response(self, *, function_responses: Any) -> None:
        self.tool_responses.append(function_responses)

    async def receive(self) -> AsyncIterator[Any]:
        if self._turn >= len(self._turns):
            raise self._close_exc
        turn = self._turns[self._turn]
        self._turn += 1
        for message in turn:
            yield message


def _conn(session: _RecordingSession) -> GoogleRealtimeConnection:
    return GoogleRealtimeConnection(cast('AsyncSession', session))


def test_google_connection_restores_in_flight_state_on_reconnect() -> None:
    # Gemini settles the cut turn in the connection itself and resumes conversation state on re-dial, so
    # the session does not settle again — it keeps the base connection's default.
    assert _conn(_RecordingSession()).reconnect_restores_in_flight_state is True


# --- helpers -----------------------------------------------------------------


def test_automatic_vad_from_turn_detection_mapping() -> None:
    # All three cross-provider knobs map through; `'medium'` leaves Gemini's own default in charge.
    assert rt_google._automatic_vad_from_turn_detection(  # pyright: ignore[reportPrivateUsage]
        {'sensitivity': 'low', 'prefix_padding_ms': 100, 'silence_duration_ms': 300}
    ) == {
        'start_sensitivity': 'low',
        'end_sensitivity': 'low',
        'prefix_padding_ms': 100,
        'silence_duration_ms': 300,
    }
    assert rt_google._automatic_vad_from_turn_detection({'sensitivity': 'medium'}) == {}  # pyright: ignore[reportPrivateUsage]


def test_tool_def_to_genai_with_and_without_description() -> None:
    with_desc = rt_google._tool_def_to_genai(  # pyright: ignore[reportPrivateUsage]
        ToolDefinition(
            name='record_reading',
            description='Record a reading',
            parameters_json_schema={
                '$defs': {
                    'Measurement': {
                        'exclusiveMinimum': 0,
                        'title': 'Measurement',
                        'type': 'integer',
                    }
                },
                'additionalProperties': False,
                'properties': {
                    'zqx_measurement': {'$ref': '#/$defs/Measurement'},
                    'kind': {'const': 'sensor', 'title': 'Kind', 'type': 'string'},
                    'observed_at': {
                        'anyOf': [{'format': 'date-time', 'type': 'string'}, {'type': 'null'}],
                        'title': 'Observed At',
                    },
                },
                'required': ['zqx_measurement', 'kind'],
                'title': 'Reading',
                'type': 'object',
            },
            return_schema={'format': 'date-time', 'title': 'Result', 'type': 'string'},
        )
    )
    assert with_desc == genai_types.FunctionDeclaration(
        name='record_reading',
        description='Record a reading',
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                'zqx_measurement': genai_types.Schema(type=genai_types.Type.INTEGER),
                'kind': genai_types.Schema(type=genai_types.Type.STRING, enum=['sensor']),
                'observed_at': genai_types.Schema(
                    type=genai_types.Type.STRING, nullable=True, description='Format: date-time'
                ),
            },
            required=['zqx_measurement', 'kind'],
        ),
        response=genai_types.Schema(type=genai_types.Type.STRING, description='Format: date-time'),
    )

    without_desc = rt_google._tool_def_to_genai(  # pyright: ignore[reportPrivateUsage]
        ToolDefinition(name='ping', parameters_json_schema={'type': 'object'})
    )
    assert without_desc.description == ''
    assert without_desc.parameters == genai_types.Schema(type=genai_types.Type.OBJECT)
    assert without_desc.response is None


def test_tool_def_narrows_schema_to_the_openapi_subset() -> None:
    """Every JSON Schema construct Gemini's `Schema` can't express has to survive in *some* form.

    Live only reads a declaration's `parameters`, which is an OpenAPI v3.0.3 subset, so the schema
    can't just be pruned to the fields `Schema` happens to have: a `oneOf` union would collapse to
    an empty schema, an int enum would go on the wire with a type `Schema.enum` can't hold, and a
    tuple would leave an array with no `items` — which Gemini rejects outright (live-verified).
    """
    tool = rt_google._tool_def_to_genai(  # pyright: ignore[reportPrivateUsage]
        ToolDefinition(
            name='record_reading',
            parameters_json_schema={
                'type': 'object',
                'properties': {
                    'zqx_measurement': {'type': 'integer', 'multipleOf': 3, 'description': 'A multiple of three.'},
                    'tags': {'type': 'array', 'items': {'type': 'string'}, 'uniqueItems': True},
                    'span': {'type': 'array', 'prefixItems': [{'type': 'integer'}, {'type': 'string'}]},
                    'size': {'type': 'integer', 'enum': [1, 2]},
                    'counts': {'type': 'object', 'additionalProperties': {'type': 'integer'}},
                    'pet': {'oneOf': [{'type': 'object'}, {'type': 'string'}]},
                },
                'required': ['zqx_measurement'],
            },
        )
    )
    assert tool.parameters == genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            # A constraint with nowhere to go simply goes unenforced; the argument itself survives.
            'zqx_measurement': genai_types.Schema(type=genai_types.Type.INTEGER, description='A multiple of three.'),
            'tags': genai_types.Schema(
                type=genai_types.Type.ARRAY, items=genai_types.Schema(type=genai_types.Type.STRING)
            ),
            # A tuple loses its positions but keeps its element types and its length.
            'span': genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(
                    any_of=[
                        genai_types.Schema(type=genai_types.Type.INTEGER),
                        genai_types.Schema(type=genai_types.Type.STRING),
                    ]
                ),
                min_items=2,
                max_items=2,
            ),
            # `Schema.enum` is a list of strings, so an int enum can't be enforced. Stringifying it
            # would make the model answer `'1'` and then fail our own validation (Pydantic won't
            # coerce a string into an int literal), so the choices move to the description instead.
            'size': genai_types.Schema(type=genai_types.Type.INTEGER, description='Allowed values: 1, 2'),
            # `additionalProperties` is dropped because Gemini mishandles it, so a `dict` field
            # always arrives empty — the rest of the tool still works.
            'counts': genai_types.Schema(type=genai_types.Type.OBJECT),
            'pet': genai_types.Schema(
                any_of=[
                    genai_types.Schema(type=genai_types.Type.OBJECT),
                    genai_types.Schema(type=genai_types.Type.STRING),
                ]
            ),
        },
        required=['zqx_measurement'],
    )


def test_tool_def_narrows_a_uniform_tuple_to_one_item_type() -> None:
    """A `tuple[int, int]` widens to a single element type, not a one-member `anyOf`.

    The sibling test covers a mixed tuple; this pins the collapse when every position agrees, which
    is the shape `Schema.items` can express directly.
    """
    tool = rt_google._tool_def_to_genai(  # pyright: ignore[reportPrivateUsage]
        ToolDefinition(
            name='record_span',
            parameters_json_schema={
                'type': 'object',
                'properties': {'span': {'type': 'array', 'prefixItems': [{'type': 'integer'}, {'type': 'integer'}]}},
            },
        )
    )
    assert tool.parameters == genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            'span': genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.INTEGER),
                min_items=2,
                max_items=2,
            )
        },
    )


def test_schema_drops_false_any_of_member() -> None:
    schema = rt_google._schema_from_json_schema(  # pyright: ignore[reportPrivateUsage]
        {'type': 'object', 'properties': {'value': {'anyOf': [False, {'type': 'string'}]}}}
    )

    assert schema.properties['value'].any_of == [genai_types.Schema(type=genai_types.Type.STRING)]  # type: ignore[index]


def test_schema_handles_boolean_property_schemas() -> None:
    # JSON Schema allows a property's schema to be a boolean, which `Schema` can't express: `True`
    # accepts anything (the unconstrained schema) and `False` accepts nothing (the property is
    # dropped). Walking into one used to raise `AttributeError` while preparing the declaration.
    schema = rt_google._schema_from_json_schema(  # pyright: ignore[reportPrivateUsage]
        {'type': 'object', 'properties': {'anything': True, 'nothing': False, 'named': {'type': 'string'}}}
    )

    assert schema.properties == {
        'anything': genai_types.Schema(),
        'named': genai_types.Schema(type=genai_types.Type.STRING),
    }


def test_schema_flattens_all_of_instead_of_erasing_it() -> None:
    """`Schema` can't express an intersection; its members merge rather than vanish.

    Dropping `allOf` like any other unsupported keyword would leave `{}` — an unconstrained
    parameter — where the schema had a type and constraints.
    """
    schema = rt_google._schema_from_json_schema(  # pyright: ignore[reportPrivateUsage]
        {
            'type': 'object',
            'properties': {
                'value': {
                    'allOf': [
                        {'type': 'string', 'minLength': 2},
                        {'maxLength': 5},
                        True,
                    ],
                }
            },
            'required': ['value'],
        }
    )

    value = schema.properties['value']  # type: ignore[index]
    assert value.type == genai_types.Type.STRING  # pyright: ignore[reportUnknownMemberType]
    assert (value.min_length, value.max_length) == (2, 5)  # pyright: ignore[reportUnknownMemberType]


def test_schema_flattens_all_of_object_members() -> None:
    """`allOf` members contributing `properties` and `required` merge into one object schema.

    An intersection of object shapes is the common `allOf` use (e.g. a base model plus a mixin);
    merging keeps every field and its requiredness where dropping the keyword would erase them all.
    """
    schema = rt_google._schema_from_json_schema(  # pyright: ignore[reportPrivateUsage]
        {
            'allOf': [
                {'type': 'object', 'properties': {'a': {'type': 'string'}}, 'required': ['a']},
                {'properties': {'b': {'type': 'integer'}}, 'required': ['a', 'b']},
            ],
        }
    )

    assert schema.type == genai_types.Type.OBJECT
    properties = schema.properties or {}
    assert properties['a'].type == genai_types.Type.STRING
    assert properties['b'].type == genai_types.Type.INTEGER
    assert schema.required == ['a', 'b']


def test_tool_def_rejects_a_recursive_schema() -> None:
    """A recursive schema has no OpenAPI-subset form at all, so it fails with an explanation.

    Left alone it would reach the SDK as an unresolved `$ref` and raise `RecursionError`.
    """
    with pytest.raises(UserError, match='Recursive `\\$ref`s in JSON Schema are not supported by Gemini'):
        rt_google._tool_def_to_genai(  # pyright: ignore[reportPrivateUsage]
            ToolDefinition(
                name='walk_tree',
                parameters_json_schema={
                    '$defs': {
                        'Node': {
                            'type': 'object',
                            'properties': {'children': {'type': 'array', 'items': {'$ref': '#/$defs/Node'}}},
                        }
                    },
                    'type': 'object',
                    'properties': {'root': {'$ref': '#/$defs/Node'}},
                },
            )
        )


@pytest.mark.parametrize('async_tool_calls', [False, True])
def test_tool_def_async_behavior(async_tool_calls: bool) -> None:
    # The expected enum is resolved in the body, not the `parametrize` decorator: decorators are
    # evaluated at collection time, before `pytestmark` can skip the module, so naming `genai_types`
    # there breaks collection wherever the `google` extra isn't installed.
    tool = rt_google._tool_def_to_genai(  # pyright: ignore[reportPrivateUsage]
        ToolDefinition(name='get_weather', parameters_json_schema={'type': 'object'}),
        async_tool_calls=async_tool_calls,
    )
    assert tool.behavior == (genai_types.Behavior.NON_BLOCKING if async_tool_calls else None)


def test_native_tool_web_search_maps_to_google_search() -> None:
    tool = rt_google._native_tool_to_genai(WebSearchTool())  # pyright: ignore[reportPrivateUsage]
    assert tool.google_search is not None


def test_native_tool_web_fetch_maps_to_url_context() -> None:
    tool = rt_google._native_tool_to_genai(WebFetchTool())  # pyright: ignore[reportPrivateUsage]
    assert tool.url_context is not None


def test_native_tool_code_execution_maps_to_code_execution() -> None:
    tool = rt_google._native_tool_to_genai(CodeExecutionTool())  # pyright: ignore[reportPrivateUsage]
    assert tool.code_execution is not None


def test_native_tool_mapping_rejects_unsupported_tool() -> None:
    with pytest.raises(UserError, match="Google realtime does not support the native tool 'ImageGenerationTool'"):
        rt_google._native_tool_to_genai(ImageGenerationTool())  # pyright: ignore[reportPrivateUsage]


async def test_agent_realtime_session_rejects_unsupported_native_tool() -> None:
    # A native tool outside Gemini's `supported_native_tools`, with no local fallback, fails up front
    # before the Live session connects — via the same native ↔ local-tool swap the classic agent-run
    # path applies, so the error points at `local=`.
    agent: Agent[None, str] = Agent()
    with pytest.raises(
        UserError,
        match=r"'ImageGenerationTool'\] not supported by this model.*ImageGeneration\(local=my_func\)",
    ):
        async with agent.realtime(
            GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest'),
            capabilities=[NativeTool(ImageGenerationTool())],
        ).session():
            pass  # pragma: no cover


def test_config_combines_function_and_native_tools() -> None:
    tools = [ToolDefinition(name='f', parameters_json_schema={'type': 'object'})]
    config = GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest')._config(  # pyright: ignore[reportPrivateUsage]
        'hi', tools, model_settings=None, native_tools=[WebSearchTool()]
    )
    assert config.tools[0].function_declarations[0].name == 'f'  # type: ignore[index,union-attr]
    assert config.tools[1].google_search is not None  # type: ignore[index,union-attr]


def test_map_usage_matches_standard_google_typed_fields() -> None:
    modality_counts = [
        genai_types.ModalityTokenCount(modality=genai_types.MediaModality.TEXT, token_count=11),
        genai_types.ModalityTokenCount(modality=genai_types.MediaModality.AUDIO, token_count=12),
    ]
    tool_counts = [
        genai_types.ModalityTokenCount(modality=genai_types.MediaModality.TEXT, token_count=13),
        genai_types.ModalityTokenCount(modality=genai_types.MediaModality.AUDIO, token_count=14),
    ]
    realtime = rt_google._map_usage(  # pyright: ignore[reportPrivateUsage]
        genai_types.UsageMetadata(
            prompt_token_count=100,
            response_token_count=20,
            cached_content_token_count=30,
            thoughts_token_count=40,
            tool_use_prompt_token_count=50,
            prompt_tokens_details=modality_counts,
            cache_tokens_details=modality_counts,
            response_tokens_details=modality_counts,
            tool_use_prompt_tokens_details=tool_counts,
        ),
        provider_name='google',
        provider_url=_GOOGLE_API_URL,
    )
    standard = model_google._metadata_as_usage(  # pyright: ignore[reportPrivateUsage]
        genai_types.GenerateContentResponse(
            usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                prompt_token_count=100,
                candidates_token_count=20,
                cached_content_token_count=30,
                thoughts_token_count=40,
                tool_use_prompt_token_count=50,
                prompt_tokens_details=modality_counts,
                cache_tokens_details=modality_counts,
                candidates_tokens_details=modality_counts,
                tool_use_prompt_tokens_details=tool_counts,
            )
        ),
        provider='google',
        provider_url=_GOOGLE_API_URL,
    )
    assert {key: value for key, value in realtime.__dict__.items() if key != 'details'} == {
        key: value for key, value in standard.__dict__.items() if key != 'details'
    }
    assert realtime == RequestUsage(
        input_tokens=150,
        output_tokens=60,
        cache_read_tokens=30,
        input_audio_tokens=26,
        cache_audio_read_tokens=12,
        cache_text_read_tokens=11,
        output_audio_tokens=12,
        output_text_tokens=11,
        input_text_tokens=24,
        input_tool_tokens=50,
        input_text_tool_tokens=13,
        input_audio_tool_tokens=14,
        output_reasoning_tokens=40,
        details={
            'cached_content_tokens': 30,
            'thoughts_tokens': 40,
            'tool_use_prompt_tokens': 50,
            'text_prompt_tokens': 11,
            'audio_prompt_tokens': 12,
            'text_cache_tokens': 11,
            'audio_cache_tokens': 12,
            'text_response_tokens': 11,
            'audio_response_tokens': 12,
            'text_tool_use_prompt_tokens': 13,
            'audio_tool_use_prompt_tokens': 14,
        },
    )
    assert standard.details == {
        'cached_content_tokens': 30,
        'thoughts_tokens': 40,
        'tool_use_prompt_tokens': 50,
        'text_prompt_tokens': 11,
        'audio_prompt_tokens': 12,
        'text_cache_tokens': 11,
        'audio_cache_tokens': 12,
        'text_candidates_tokens': 11,
        'audio_candidates_tokens': 12,
        'text_tool_use_prompt_tokens': 13,
        'audio_tool_use_prompt_tokens': 14,
    }
    empty = rt_google._map_usage(  # pyright: ignore[reportPrivateUsage]
        genai_types.UsageMetadata(), provider_name='google', provider_url=_GOOGLE_API_URL
    )
    assert empty == RequestUsage()


def test_single_ws_user_agent_noop_without_duplicate() -> None:
    # A client whose headers hold fewer than two `user-agent` entries needs no reconciliation: the
    # context manager yields without touching them. A real `GoogleProvider` always adds a capitalized
    # duplicate, so this defensive branch can't be reached through `connect` — hence a direct unit test.
    from types import SimpleNamespace

    headers = {'user-agent': 'solo'}
    client = SimpleNamespace(_api_client=SimpleNamespace(_http_options=SimpleNamespace(headers=headers)))
    with rt_google._single_ws_user_agent(cast('Any', client)):  # pyright: ignore[reportPrivateUsage]
        assert headers == {'user-agent': 'solo'}
    assert headers == {'user-agent': 'solo'}


def test_ws_trace_context_injects_and_restores_headers() -> None:
    # `google-genai` forwards the client's HTTP headers as the Live handshake headers, so trace context
    # is injected into them for the connect only, then removed so the shared client's later HTTP
    # requests don't carry a stale `traceparent`. The header dict is the SDK's private one, so this is a
    # direct unit test. (The no-op-without-a-span case is covered by the OpenAI/xAI handshake tests.)
    pytest.importorskip('opentelemetry.sdk')
    from types import SimpleNamespace

    from opentelemetry.sdk.trace import TracerProvider

    headers = {'user-agent': 'solo'}
    client = SimpleNamespace(_api_client=SimpleNamespace(_http_options=SimpleNamespace(headers=headers)))
    tracer = TracerProvider().get_tracer('test')
    with tracer.start_as_current_span('root'):
        with rt_google._ws_trace_context(cast('Any', client)):  # pyright: ignore[reportPrivateUsage]
            assert 'traceparent' in headers
        # Injected keys are removed after the handshake; the original headers are untouched.
        assert headers == {'user-agent': 'solo'}


def test_ws_trace_context_does_not_duplicate_a_differently_cased_header() -> None:
    # Header names are case-insensitive and `websockets` stores them that way, so a client already
    # carrying `Traceparent` must not gain a second, lowercase one — the handshake can be rejected for
    # the duplicate (the same hazard `_single_ws_user_agent` reconciles for `User-Agent`).
    pytest.importorskip('opentelemetry.sdk')
    from types import SimpleNamespace

    from opentelemetry.sdk.trace import TracerProvider

    headers = {'Traceparent': 'preset'}
    client = SimpleNamespace(_api_client=SimpleNamespace(_http_options=SimpleNamespace(headers=headers)))
    tracer = TracerProvider().get_tracer('test')
    with tracer.start_as_current_span('root'):
        with rt_google._ws_trace_context(cast('Any', client)):  # pyright: ignore[reportPrivateUsage]
            assert headers == {'Traceparent': 'preset'}
    assert headers == {'Traceparent': 'preset'}


def test_ws_connect_lock_is_per_event_loop() -> None:
    # The lock is process-wide by intent (it guards a replacement of the `google.genai.live.ws_connect`
    # module global), but an `anyio.Lock` binds to the loop it is first used on, so one shared instance
    # would break an app that opens sessions from more than one runtime. Deliberately a sync test: it
    # needs to own the loops. Within one loop the same lock still serializes every handshake, and the
    # `RunVar` holding them is weak-keyed on the loop, so a torn-down loop's lock isn't retained.
    refs: list[weakref.ReferenceType[Any]] = []

    async def take_lock() -> Any:
        lock = rt_google._ws_connect_lock()  # pyright: ignore[reportPrivateUsage]
        assert rt_google._ws_connect_lock() is lock  # pyright: ignore[reportPrivateUsage]
        refs.append(weakref.ref(lock))
        return lock

    first = asyncio.run(take_lock())
    second = asyncio.run(take_lock())
    assert second is not first

    del first, second
    gc.collect()
    assert [ref() for ref in refs] == [None, None]


def test_google_genai_private_http_options_contract() -> None:
    """Pin the private header chain used with the minimum supported `google-genai` SDK."""
    client = Client(api_key='test')
    headers = client._api_client._http_options.headers  # pyright: ignore[reportPrivateUsage]
    assert isinstance(headers, MutableMapping)


async def test_connect_serializes_shared_client_header_mutations(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = GoogleProvider(api_key='test-key')
    client = provider.client
    headers = client._api_client._http_options.headers  # pyright: ignore[reportPrivateUsage]
    assert headers is not None
    original_headers = headers.copy()
    handshake_headers: list[dict[str, str]] = []

    class _ConcurrentConnect:
        async def __aenter__(self) -> _RecordingSession:
            # Let the other task reach the handshake. Without per-client serialization its header
            # contexts overlap this suspension and one handshake observes the other's mutations.
            await anyio.sleep(0)
            handshake_headers.append(headers.copy())
            return _RecordingSession()

        async def __aexit__(self, *exc: object) -> bool:
            return False

    def connect(*, model: str, config: genai_types.LiveConnectConfig) -> _ConcurrentConnect:
        return _ConcurrentConnect()

    traceparent: ContextVar[str] = ContextVar('traceparent')

    def inject(carrier: dict[str, str]) -> None:
        carrier['traceparent'] = traceparent.get()

    monkeypatch.setattr(client.aio.live, 'connect', connect)
    monkeypatch.setattr(rt_google, 'inject_trace_context', inject)
    model = GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest', provider=provider)

    async def open_connection(value: str) -> None:
        token = traceparent.set(value)
        try:
            async with _connect(model, ''):
                pass
        finally:
            traceparent.reset(token)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(open_connection, 'trace-1')
        task_group.start_soon(open_connection, 'trace-2')

    assert {captured['traceparent'] for captured in handshake_headers} == {'trace-1', 'trace-2'}
    assert all(sum(key.lower() == 'user-agent' for key in captured) == 1 for captured in handshake_headers)
    assert headers == original_headers


class _StopDial(Exception):
    """Raised from the fake handshake to short-circuit `connect` once headers are captured."""


class _FakeWSConnect:
    async def __aenter__(self) -> None:
        raise _StopDial()

    async def __aexit__(self, *exc: object) -> bool:  # pragma: no cover
        return False


def _capture_ws_connect(captured: dict[str, Any]) -> Any:
    """A stand-in for `google.genai.live.ws_connect` that records the dialed URI and headers."""

    def ws_connect(uri: str, *, additional_headers: dict[str, str] | None = None, **kwargs: Any) -> _FakeWSConnect:
        captured['uri'] = uri
        captured['headers'] = dict(additional_headers or {})
        return _FakeWSConnect()

    return ws_connect


async def test_gateway_handshake_carries_bearer_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    # A gateway provider dials the Live WebSocket through the SDK, which forwards the client's HTTP
    # headers as the handshake's `additional_headers`. The gateway authenticates on `Authorization:
    # Bearer <key>` — added to REST calls by its httpx hook, which can't reach this `websockets` dial.
    # So `gateway_provider` sets the bearer as a static header on the client at build time (see
    # `_set_google_ws_gateway_auth`), and it rides along to the handshake automatically. Driven end-to-end
    # through `connect` (patching the SDK's `ws_connect`, as the cassette engine does) so the real URL
    # derivation and header stack are exercised, not a hand-built dict.
    provider = gateway_provider('google', api_key='gw-key', base_url='https://gateway.pydantic.dev/proxy')
    model = GoogleRealtimeModel('gemini-live-2.5-flash', provider=provider)

    captured: dict[str, Any] = {}
    monkeypatch.setattr('google.genai.live.ws_connect', _capture_ws_connect(captured))
    with pytest.raises(_StopDial):
        async with _connect(model, 'hi'):
            pass  # pragma: no cover

    # The SDK swaps https→wss and appends the Vertex BidiGenerateContent path onto the gateway base
    # URL; the gateway's realtime relay routes this native Bidi path directly, so the dialed URL is
    # exactly what the SDK built — no client-side reshaping.
    assert captured['uri'] == snapshot(
        'wss://gateway.pydantic.dev/proxy/google-vertex/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent'
    )
    assert captured['headers'].get('Authorization') == 'Bearer gw-key'
    # `_single_ws_user_agent` still runs, so the handshake carries exactly one user-agent header.
    assert sum(key.lower() == 'user-agent' for key in captured['headers']) == 1
    # The bearer lives permanently on the client's static http options (that's what carries it onto the
    # WebSocket), so REST requests carry it too. That's redundant with the gateway's httpx request hook
    # but harmless — the same value — and the hook leaves a pre-existing `Authorization` header untouched.
    rest_headers = provider.client._api_client._http_options.headers  # pyright: ignore[reportPrivateUsage]
    assert rest_headers is not None and rest_headers['Authorization'] == 'Bearer gw-key'


async def test_non_gateway_handshake_has_no_bearer_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    # A plain `GoogleProvider` is not a gateway provider, so `connect` leaves the handshake auth to the
    # SDK (the API key travels as `x-goog-api-key`) and adds no `Authorization` header.
    model = GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest', provider=GoogleProvider(api_key='k'))

    captured: dict[str, Any] = {}
    monkeypatch.setattr('google.genai.live.ws_connect', _capture_ws_connect(captured))
    with pytest.raises(_StopDial):
        async with _connect(model, 'hi'):
            pass  # pragma: no cover

    assert 'Authorization' not in captured['headers']


# --- provider resolution & capabilities --------------------------------------


def test_default_provider_is_google() -> None:
    # The default `'google'` provider reads GOOGLE_API_KEY (set to a placeholder by the autouse fixture).
    model = GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest')
    assert isinstance(model.client, Client)


def test_provider_instance_is_reused() -> None:
    provider = GoogleProvider(api_key='k')
    model = GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest', provider=provider)
    assert model.client is provider.client


def test_profile() -> None:
    profile = GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest').profile
    # Gemini Live has no manual turn control or server-side interruption (automatic VAD only).
    assert (
        profile.get('supports_image_input'),
        profile.get('supports_manual_turn_control'),
        profile.get('supports_interruption'),
        profile.get('supports_session_seeding'),
        profile.get('supports_seeding_images'),
        profile.get('supports_seeding_audio'),
    ) == (
        True,
        False,
        False,
        True,
        True,
        False,
    )
    # Search grounding only, on every Live model: code execution is rejected outright by the
    # native-audio models (verified live: `1007 Code Execution tool is not supported for this model`)
    # and URL context is accepted but never actually grounds, so neither is advertised and a
    # `local=` fallback is used instead.
    assert profile.get('supported_native_tools') == frozenset({WebSearchTool})
    assert GoogleRealtimeModel('gemini-3.1-flash-live-preview').profile.get('supported_native_tools') == frozenset(
        {WebSearchTool}
    )
    # The default model is native-audio, the only Gemini family that honors `NON_BLOCKING`.
    # Supported is not the same as enabled: it gates the opt-in `google_async_tool_calls` setting.
    assert profile.get('supports_async_tool_calls') is True
    # Gemini Live renders an opted-in return schema natively (the declaration's `response`).
    assert profile.get('supports_tool_return_schema') is True
    assert profile.get('audio_input_sample_rate') == 16000
    assert profile.get('audio_output_sample_rate') == 24000


# --- config ------------------------------------------------------------------


def test_config_full() -> None:
    settings = GoogleRealtimeModelSettings(
        max_tokens=256,
        temperature=0.5,
        top_p=0.9,
        google_voice='Puck',
        google_vad={'prefix_padding_ms': 200, 'silence_duration_ms': 400},
    )
    model = GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest', settings=settings)
    assert model.settings == settings
    tools = [ToolDefinition(name='get_weather', description='Weather', parameters_json_schema={'type': 'object'})]
    config = model._config('Be nice', tools, model_settings=settings)  # pyright: ignore[reportPrivateUsage]

    assert model.model_name == 'gemini-2.5-flash-native-audio-latest'
    assert config.response_modalities == [genai_types.Modality.AUDIO]
    assert config.system_instruction == 'Be nice'
    assert config.speech_config.voice_config.prebuilt_voice_config.voice_name == 'Puck'  # type: ignore[union-attr]
    assert config.input_audio_transcription is not None
    assert config.output_audio_transcription is not None
    detection = config.realtime_input_config.automatic_activity_detection  # type: ignore[union-attr]
    assert detection.prefix_padding_ms == 200 and detection.silence_duration_ms == 400  # type: ignore[union-attr]
    assert config.tools[0].function_declarations[0].name == 'get_weather'  # type: ignore[index,union-attr]
    assert config.max_output_tokens == 256
    assert config.temperature == 0.5
    assert config.top_p == 0.9


def test_config_thinking_maps_to_thinking_level() -> None:
    # The default native-audio model supports thinking (verified live); `thinking` maps to a level,
    # and `False` disables it via a zero budget.
    def thinking_config(thinking: object) -> genai_types.ThinkingConfig | None:
        settings = GoogleRealtimeModelSettings(thinking=thinking)  # type: ignore[typeddict-item]
        return (
            GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest')
            ._config('hi', None, model_settings=settings)  # pyright: ignore[reportPrivateUsage]
            .thinking_config
        )

    assert thinking_config('high') == genai_types.ThinkingConfig(thinking_level=genai_types.ThinkingLevel.HIGH)
    assert thinking_config('xhigh') == genai_types.ThinkingConfig(thinking_level=genai_types.ThinkingLevel.HIGH)
    assert thinking_config('minimal') == genai_types.ThinkingConfig(thinking_level=genai_types.ThinkingLevel.MINIMAL)
    assert thinking_config(True) == genai_types.ThinkingConfig(thinking_level=genai_types.ThinkingLevel.MEDIUM)
    assert thinking_config(False) == genai_types.ThinkingConfig(thinking_budget=0)


def test_config_tool_choice_restricts_advertised_tools() -> None:
    tools = [ToolDefinition(name=name, parameters_json_schema={'type': 'object'}) for name in ('allowed', 'unsafe')]
    allowed = GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest')._config(  # pyright: ignore[reportPrivateUsage]
        'hi', tools, model_settings=GoogleRealtimeModelSettings(tool_choice=['allowed'])
    )
    assert [tool.name for tool in allowed.tools[0].function_declarations] == ['allowed']  # type: ignore[index,union-attr]

    none = GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest')._config(  # pyright: ignore[reportPrivateUsage]
        'hi', tools, model_settings=GoogleRealtimeModelSettings(tool_choice='none')
    )
    assert none.tools is None


def test_config_google_thinking_config_wins_over_unified_thinking() -> None:
    settings = GoogleRealtimeModelSettings(thinking='low', google_thinking_config={'thinking_budget': 512})
    config = GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest')._config('hi', None, model_settings=settings)  # pyright: ignore[reportPrivateUsage]
    assert config.thinking_config == genai_types.ThinkingConfig(thinking_budget=512)


def test_config_thinking_on_non_thinking_model_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every current Gemini Live model takes a thinking config (verified live), but the setting stays
    # profile-gated so a future model that can't reason silently falls back to its default rather than
    # failing the handshake on a config it rejects.
    model = GoogleRealtimeModel('gemini-live-2.5-flash-preview', settings=GoogleRealtimeModelSettings(thinking='high'))

    def no_thinking_profile(model_name: str) -> RealtimeModelProfile:
        return RealtimeModelProfile()

    monkeypatch.setattr(
        type(model._provider),  # pyright: ignore[reportPrivateUsage]
        'realtime_model_profile',
        staticmethod(no_thinking_profile),
    )
    config = model._config('hi', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]
    assert config.thinking_config is None


def test_async_tool_calls_opt_in_resolution() -> None:
    # Opt-in and capability-gated: off unless asked for, and on only where the model honors it.
    # A Live model that doesn't (verified live: it accepts `NON_BLOCKING` and blocks anyway) warns
    # rather than quietly promising speech that never arrives.
    on = GoogleRealtimeModelSettings(google_async_tool_calls=True)
    native_audio = GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest')
    assert native_audio._async_tool_calls(None) is False  # pyright: ignore[reportPrivateUsage]
    assert native_audio._async_tool_calls(GoogleRealtimeModelSettings()) is False  # pyright: ignore[reportPrivateUsage]
    assert native_audio._async_tool_calls(on) is True  # pyright: ignore[reportPrivateUsage]

    half_cascade = GoogleRealtimeModel('gemini-live-2.5-flash-preview')
    assert half_cascade._async_tool_calls(on) is False  # pyright: ignore[reportPrivateUsage]


def test_config_minimal_text_no_transcription_no_vad() -> None:
    model = GoogleRealtimeModel(
        'gemini-2.5-flash-native-audio-latest',
        settings=GoogleRealtimeModelSettings(
            output_modality='text', google_input_transcription=False, google_output_transcription=False
        ),
    )
    config = model._config('', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]
    assert config.response_modalities == [genai_types.Modality.TEXT]
    assert config.system_instruction is None  # empty instructions → not set
    assert config.speech_config is None
    assert config.input_audio_transcription is None
    assert config.output_audio_transcription is None
    assert config.realtime_input_config is None
    assert config.tools is None
    assert config.max_output_tokens is None


def test_shared_input_transcription_none_turns_gemini_transcription_off() -> None:
    """`input_transcription_model=None` means "don't transcribe" on Gemini too.

    Gemini has no separate transcription model, so a pinned id can't be honored and is ignored — but the
    `None` that asks for transcription *off* is the whole point of the setting for anyone keeping the
    user's words out of history, so Gemini must honor it rather than transcribe anyway. Kept a unit test
    because it's the request payload that has to change, which a cassette match isn't sensitive to.
    """
    off = GoogleRealtimeModel(
        'gemini-2.5-flash-native-audio-latest', settings=GoogleRealtimeModelSettings(input_transcription_model=None)
    )
    assert off._config('', None, model_settings=None).input_audio_transcription is None  # pyright: ignore[reportPrivateUsage]

    # A pinned id can't be pointed at anything, so transcription stays on, as documented.
    pinned = GoogleRealtimeModel(
        'gemini-2.5-flash-native-audio-latest',
        settings=GoogleRealtimeModelSettings(input_transcription_model='gpt-4o-transcribe'),
    )
    assert pinned._config('', None, model_settings=None).input_audio_transcription is not None  # pyright: ignore[reportPrivateUsage]

    # The provider-specific setting wins where both are given, in either direction.
    both_on = GoogleRealtimeModel(
        'gemini-2.5-flash-native-audio-latest',
        settings=GoogleRealtimeModelSettings(input_transcription_model=None, google_input_transcription=True),
    )
    assert both_on._config('', None, model_settings=None).input_audio_transcription is not None  # pyright: ignore[reportPrivateUsage]


def test_config_forwards_only_present_model_settings() -> None:
    # `model_settings` is non-empty but carries none of the forwarded fields → all stay unset
    # (`presence_penalty` has no Gemini Live equivalent and is ignored).
    config = GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest')._config(  # pyright: ignore[reportPrivateUsage]
        'hi', None, model_settings=GoogleRealtimeModelSettings()
    )
    assert config.max_output_tokens is None
    assert config.temperature is None
    assert config.top_p is None
    assert config.top_k is None
    assert config.seed is None
    assert config.thinking_config is None
    assert config.media_resolution is None


# --- send --------------------------------------------------------------------


async def test_send_audio() -> None:
    session = _RecordingSession()
    await _conn(session).send(BinaryAudio(data=b'\x01\x02', media_type='audio/pcm'))
    blob = session.realtime[0]['audio']
    assert blob.data == b'\x01\x02'
    assert blob.mime_type == 'audio/pcm;rate=16000'


async def test_send_audio_rejects_non_pcm_media_type() -> None:
    session = _RecordingSession()
    with pytest.raises(UserError, match='require raw PCM audio'):
        await _conn(session).send(BinaryAudio(data=b'RIFF', media_type='audio/wav'))
    assert session.realtime == []


async def test_send_text() -> None:
    # A typed turn is committed with `send_client_content(turn_complete=True)` so the model replies.
    session = _RecordingSession()
    await _conn(session).send('hello')
    sent = session.client_content[0]
    assert sent['turn_complete'] is True
    assert sent['turns'].role == 'user'
    assert sent['turns'].parts[0].text == 'hello'


async def test_send_image_as_video_frame() -> None:
    session = _RecordingSession()
    await _conn(session).send(BinaryImage(data=b'\xff\xd8', media_type='image/jpeg'))
    blob = session.realtime[0]['video']
    assert blob.data == b'\xff\xd8'
    assert blob.mime_type == 'image/jpeg'


async def test_send_tool_result_echoes_name() -> None:
    session = _RecordingSession()
    conn = _conn(session)
    # a prior ToolCall populates the call_id -> name map.
    conn._map_message(  # pyright: ignore[reportPrivateUsage]
        genai_types.LiveServerMessage(
            tool_call=genai_types.LiveServerToolCall(
                function_calls=[genai_types.FunctionCall(id='c1', name='get_weather', args={})]
            )
        )
    )
    await conn.send(ToolResult(tool_call_id='c1', output='Sunny'))
    response = session.tool_responses[0]
    assert response.id == 'c1'
    assert response.name == 'get_weather'
    assert response.response == {'output': 'Sunny'}


@pytest.mark.parametrize('async_tool_calls', [False, True])
async def test_send_tool_result_async_scheduling(async_tool_calls: bool) -> None:
    # As in `test_tool_def_async_behavior`, the expected enum is resolved in the body so collection
    # doesn't need the `google` extra.
    session = _RecordingSession()
    conn = GoogleRealtimeConnection(cast('AsyncSession', session), async_tool_calls=async_tool_calls)
    conn._map_message(  # pyright: ignore[reportPrivateUsage]
        genai_types.LiveServerMessage(
            tool_call=genai_types.LiveServerToolCall(
                function_calls=[genai_types.FunctionCall(id='c1', name='get_weather', args={})]
            )
        )
    )

    await conn.send(ToolResult(tool_call_id='c1', output='Sunny'))

    # `INTERRUPT`, so the result lands in the reply the model is already speaking rather than being
    # queued until after it has answered from its own knowledge.
    assert session.tool_responses[0].scheduling == (
        genai_types.FunctionResponseScheduling.INTERRUPT if async_tool_calls else None
    )


def _register_call(conn: GoogleRealtimeConnection, tool_call_id: str = 'c1', name: str = 'inspect') -> None:
    conn._map_message(  # pyright: ignore[reportPrivateUsage]
        genai_types.LiveServerMessage(
            tool_call=genai_types.LiveServerToolCall(
                function_calls=[genai_types.FunctionCall(id=tool_call_id, name=name, args={})]
            )
        )
    )


async def test_send_tool_result_text_content_folds_into_output() -> None:
    """`FunctionResponse.response` is JSON-only, so text attachments are folded into the output."""
    session = _RecordingSession()
    conn = _conn(session)
    _register_call(conn)
    await conn.send(
        ToolResult(
            tool_call_id='c1',
            output='done',
            content=['plain context', TextContent('extra context'), CachePoint()],
        )
    )
    assert session.tool_responses[0].response == {'output': 'done\n\nplain context\n\nextra context'}


async def test_send_tool_result_binary_content_raises_with_nothing_sent() -> None:
    """Media attached to a tool return raises with the tool result unsent — never a silent
    placeholder. Gemini Live has no channel that delivers it correctly today (probed live; see
    https://github.com/pydantic/pydantic-ai/issues/7362), so the loud error is the honest behavior,
    matching the never-silent rule the OpenAI-protocol codec applies to its unsupported media."""
    session = _RecordingSession()
    conn = _conn(session)
    _register_call(conn)
    with pytest.raises(UserError, match='tool results are JSON-only, so `BinaryContent` content'):
        await conn.send(
            ToolResult(
                tool_call_id='c1',
                output='done',
                content=[BinaryContent(data=b'png', media_type='image/png', identifier='result.png')],
            )
        )
    assert session.tool_responses == []
    assert session.client_content == []


async def test_parallel_id_less_calls_do_not_collide() -> None:
    # Gemini may emit multiple function calls without ids; each must get a distinct internal id so
    # results echo the right name back (Gemini gets `id=None`, which is what it sent).
    session = _RecordingSession()
    conn = _conn(session)
    events = conn._map_message(  # pyright: ignore[reportPrivateUsage]
        genai_types.LiveServerMessage(
            tool_call=genai_types.LiveServerToolCall(
                function_calls=[
                    genai_types.FunctionCall(name='get_weather', args={}),
                    genai_types.FunctionCall(name='get_time', args={}),
                ]
            )
        )
    )
    call_ids = [e.tool_call_id for e in events if isinstance(e, ToolCall)]
    assert len(set(call_ids)) == 2  # distinct internal ids, no collision

    await conn.send(ToolResult(tool_call_id=call_ids[0], output='Sunny'))
    await conn.send(ToolResult(tool_call_id=call_ids[1], output='Noon'))
    assert [(r.id, r.name, r.response) for r in session.tool_responses] == [
        (None, 'get_weather', {'output': 'Sunny'}),
        (None, 'get_time', {'output': 'Noon'}),
    ]


async def test_send_unsupported_raises() -> None:
    session = _RecordingSession()
    with pytest.raises(UserError, match='Gemini Live does not support object input'):
        await _conn(session).send(object())  # type: ignore[arg-type]


# --- message mapping ---------------------------------------------------------


def test_map_audio_and_text_parts() -> None:
    conn = _conn(_RecordingSession())
    message = genai_types.LiveServerMessage(
        server_content=genai_types.LiveServerContent(
            model_turn=genai_types.Content(
                parts=[
                    genai_types.Part(inline_data=genai_types.Blob(data=b'\x01', mime_type='audio/pcm')),
                    genai_types.Part(text='partial'),
                    genai_types.Part(),  # neither audio nor text → produces no event
                ]
            )
        )
    )
    assert conn._map_message(message) == [  # pyright: ignore[reportPrivateUsage]
        AudioDelta(data=b'\x01'),
        OutputTranscript(text='partial', is_final=False, output_text=True),
    ]


def test_map_skips_thought_parts() -> None:
    # Native-audio models stream their reasoning as `thought` text next to the spoken answer; it must
    # not leak into the transcript (only the real spoken text becomes a `OutputTranscript`). Kept as a unit
    # test because a cassette can't reliably force a model to think.
    conn = _conn(_RecordingSession())
    message = genai_types.LiveServerMessage(
        server_content=genai_types.LiveServerContent(
            model_turn=genai_types.Content(
                parts=[
                    genai_types.Part(text='**Planning the greeting**', thought=True),
                    genai_types.Part(text='Hello there.'),
                ]
            )
        )
    )
    assert conn._map_message(message) == [  # pyright: ignore[reportPrivateUsage]
        OutputTranscript(text='Hello there.', is_final=False, output_text=True)
    ]


def test_map_transcriptions_interrupt_and_turn_complete() -> None:
    conn = _conn(_RecordingSession())
    message = genai_types.LiveServerMessage(
        server_content=genai_types.LiveServerContent(
            input_transcription=genai_types.Transcription(text='weather?', finished=True),
            output_transcription=genai_types.Transcription(text='Sunny', finished=False),
            interrupted=True,
            turn_complete=True,
        )
    )
    assert conn._map_message(message) == [  # pyright: ignore[reportPrivateUsage]
        InputTranscript(text='weather?', is_final=True),
        OutputTranscript(text='Sunny', is_final=False),
        RealtimeResponseInterruptedEvent(),
        ResponseDone(interrupted=True),
    ]


def test_map_interruption_latches_until_turn_complete() -> None:
    conn = _conn(_RecordingSession())
    interrupted = genai_types.LiveServerMessage(server_content=genai_types.LiveServerContent(interrupted=True))
    completed = genai_types.LiveServerMessage(server_content=genai_types.LiveServerContent(turn_complete=True))
    assert conn._map_message(interrupted) == [  # pyright: ignore[reportPrivateUsage]
        RealtimeResponseInterruptedEvent()
    ]
    assert conn._map_message(completed) == [ResponseDone(interrupted=True)]  # pyright: ignore[reportPrivateUsage]
    assert conn._map_message(completed) == [ResponseDone(interrupted=False)]  # pyright: ignore[reportPrivateUsage]


async def test_interruption_finalizes_session_response_as_interrupted() -> None:
    provider_session = _RecordingSession(
        [
            [
                genai_types.LiveServerMessage(
                    server_content=genai_types.LiveServerContent(
                        output_transcription=genai_types.Transcription(text='Cut off', finished=False),
                        interrupted=True,
                    )
                ),
                genai_types.LiveServerMessage(server_content=genai_types.LiveServerContent(turn_complete=True)),
            ]
        ]
    )
    session = RealtimeSession(
        _conn(provider_session),
        model=FakeRealtimeModel(_conn(provider_session), model_name='gemini-live', system='google'),
        tool_manager=make_tool_manager(),
    )
    events: list[Any] = []
    async with session:
        async for event in session:
            events.append(event)
            if isinstance(event, RealtimeTurnCompleteEvent):
                break

    assert RealtimeResponseInterruptedEvent() in events
    assert not any(event.event_kind == 'input_speech_start' for event in events)
    response = next(message for message in session.new_messages() if isinstance(message, ModelResponse))
    assert response.state == 'interrupted'
    assert response.finish_reason is None


def test_map_tool_call_and_usage() -> None:
    conn = _conn(_RecordingSession())
    message = genai_types.LiveServerMessage(
        tool_call=genai_types.LiveServerToolCall(
            function_calls=[genai_types.FunctionCall(id='c1', name='calc', args={'x': 1})]
        ),
        usage_metadata=genai_types.UsageMetadata(prompt_token_count=7, response_token_count=2),
    )
    assert conn._map_message(message) == [  # pyright: ignore[reportPrivateUsage]
        ToolCall(tool_call_id='c1', tool_name='calc', args='{"x":1}'),
        SessionUsage(usage=RequestUsage(input_tokens=7, output_tokens=2)),
    ]


def test_map_tool_call_cancellation() -> None:
    # Gemini's `toolCallCancellation` (sent when the model abandons in-flight calls, e.g. on barge-in)
    # maps to a `ToolCallCancelled` carrying the cancelled call ids for the session to act on.
    conn = _conn(_RecordingSession())
    conn._map_message(  # pyright: ignore[reportPrivateUsage]
        genai_types.LiveServerMessage(
            tool_call=genai_types.LiveServerToolCall(
                function_calls=[
                    genai_types.FunctionCall(id='c1', name='first', args={}),
                    genai_types.FunctionCall(id='c2', name='second', args={}),
                    genai_types.FunctionCall(id='active', name='active', args={}),
                ]
            )
        )
    )
    message = genai_types.LiveServerMessage(
        tool_call_cancellation=genai_types.LiveServerToolCallCancellation(ids=['c1', 'c2'])
    )
    assert conn._map_message(message) == [ToolCallCancelled(tool_call_ids=['c1', 'c2'])]  # pyright: ignore[reportPrivateUsage]
    assert conn._tool_calls == {'active': ('active', 'active')}  # pyright: ignore[reportPrivateUsage]


def test_map_grounding_and_url_context_to_native_tool_part_events() -> None:
    # Grounding streams native tool parts matching the classic `GoogleModel` shapes exactly (web_search +
    # web_fetch, including a source's `domain` and a fetch's retrieval status). Kept as a unit test because a
    # cassette can't reliably force the model to ground and the recording key only exposes audio-out.
    conn = _conn(_RecordingSession())
    message = genai_types.LiveServerMessage(
        server_content=genai_types.LiveServerContent(
            grounding_metadata=genai_types.GroundingMetadata(
                web_search_queries=['weather rome'],
                grounding_chunks=[
                    genai_types.GroundingChunk(
                        web=genai_types.GroundingChunkWeb(
                            uri='https://example.com', title='Example', domain='example.com'
                        )
                    ),
                    genai_types.GroundingChunk(web=None),  # ignored by `SourcesEvent`: no web chunk
                    genai_types.GroundingChunk(web=genai_types.GroundingChunkWeb(uri=None)),  # ignored: no uri
                ],
            ),
            url_context_metadata=genai_types.UrlContextMetadata(
                url_metadata=[
                    genai_types.UrlMetadata(
                        retrieved_url='https://fetched.example',
                        url_retrieval_status=genai_types.UrlRetrievalStatus.URL_RETRIEVAL_STATUS_SUCCESS,
                    ),
                    genai_types.UrlMetadata(retrieved_url=None),  # ignored by `SourcesEvent`: no url
                ]
            ),
        )
    )
    parts = [
        NativeToolCallPart(
            tool_name='web_search',
            args={'queries': ['weather rome']},
            tool_call_id=IsStr(),
            provider_name='google',
        ),
        NativeToolReturnPart(
            tool_name='web_search',
            content=[
                {'domain': 'example.com', 'title': 'Example', 'uri': 'https://example.com'},
                # The `web=None` chunk is dropped; the uri-less one round-trips, matching classic.
                {'domain': None, 'title': None, 'uri': None},
            ],
            tool_call_id=IsStr(),
            timestamp=IsDatetime(),
            provider_name='google',
        ),
        NativeToolCallPart(
            tool_name='web_fetch',
            args={'urls': ['https://fetched.example']},
            tool_call_id=IsStr(),
            provider_name='google',
        ),
        NativeToolReturnPart(
            tool_name='web_fetch',
            content=[
                {
                    'retrieved_url': 'https://fetched.example',
                    'url_retrieval_status': 'URL_RETRIEVAL_STATUS_SUCCESS',
                },
                {'retrieved_url': None, 'url_retrieval_status': None},
            ],
            tool_call_id=IsStr(),
            timestamp=IsDatetime(),
            provider_name='google',
        ),
    ]
    assert conn._map_message(message) == [  # pyright: ignore[reportPrivateUsage]
        event
        for index, part in enumerate(parts)
        for event in (PartStartEvent(index=index, part=part), PartEndEvent(index=index, part=part))
    ]


def test_map_code_execution_to_native_tool_parts() -> None:
    # When Gemini Live runs code, the executed code and its result arrive as `executable_code` /
    # `code_execution_result` parts on the model turn. They map to a `NativeToolCallPart` /
    # `NativeToolReturnPart` pair byte-identical to the classic `GoogleModel`'s (tool_name
    # `code_execution`, `args`/`content` from the SDK models' JSON dump), sharing a single `tool_call_id`
    # so the return pairs with its call, and stream as part start/end events. The spoken transcript still
    # comes through as its own `OutputTranscript`. Kept as a unit test because
    # a cassette can't reliably force the model to run code and the recording key only exposes audio-out.
    conn = _conn(_RecordingSession())
    message = genai_types.LiveServerMessage(
        server_content=genai_types.LiveServerContent(
            model_turn=genai_types.Content(
                parts=[
                    genai_types.Part(
                        executable_code=genai_types.ExecutableCode(
                            code='print(1 + 1)', language=genai_types.Language.PYTHON
                        )
                    ),
                    genai_types.Part(
                        code_execution_result=genai_types.CodeExecutionResult(
                            outcome=genai_types.Outcome.OUTCOME_OK, output='2\n'
                        )
                    ),
                    genai_types.Part(text='The answer is 2.'),
                ]
            )
        )
    )
    parts = [
        NativeToolCallPart(
            tool_name='code_execution',
            args={'code': 'print(1 + 1)', 'language': 'PYTHON'},
            tool_call_id=(code_id := IsSameStr()),
            provider_name='google',
        ),
        NativeToolReturnPart(
            tool_name='code_execution',
            content={'outcome': 'OUTCOME_OK', 'output': '2\n'},
            tool_call_id=code_id,
            timestamp=IsDatetime(),
            provider_name='google',
        ),
    ]
    assert conn._map_message(message) == [  # pyright: ignore[reportPrivateUsage]
        OutputTranscript(text='The answer is 2.', is_final=False, output_text=True),
        *[
            event
            for index, part in enumerate(parts)
            for event in (PartStartEvent(index=index, part=part), PartEndEvent(index=index, part=part))
        ],
    ]


def test_native_tool_part_indexes_increase_across_messages_and_reset_each_turn() -> None:
    conn = _conn(_RecordingSession())

    def message(code: str, *, turn_complete: bool = False) -> genai_types.LiveServerMessage:
        return genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                model_turn=genai_types.Content(
                    parts=[
                        genai_types.Part(
                            executable_code=genai_types.ExecutableCode(code=code, language=genai_types.Language.PYTHON)
                        )
                    ]
                ),
                turn_complete=turn_complete,
            )
        )

    first = conn._map_message(message('print(1)'))  # pyright: ignore[reportPrivateUsage]
    second = conn._map_message(message('print(2)', turn_complete=True))  # pyright: ignore[reportPrivateUsage]
    next_turn = conn._map_message(message('print(3)'))  # pyright: ignore[reportPrivateUsage]

    assert [event.index for event in first + second if isinstance(event, PartStartEvent)] == [0, 1]
    assert [event.index for event in next_turn if isinstance(event, PartStartEvent)] == [0]


def test_map_grounding_absent_yields_no_sources() -> None:
    conn = _conn(_RecordingSession())
    message = genai_types.LiveServerMessage(
        server_content=genai_types.LiveServerContent(
            grounding_metadata=genai_types.GroundingMetadata(grounding_chunks=[]),
        )
    )
    assert conn._map_message(message) == []  # pyright: ignore[reportPrivateUsage]


def test_map_empty_message_yields_nothing() -> None:
    conn = _conn(_RecordingSession())
    assert conn._map_message(genai_types.LiveServerMessage()) == []  # pyright: ignore[reportPrivateUsage]


# --- connect -----------------------------------------------------------------


def _turn(text: str) -> genai_types.LiveServerMessage:
    return genai_types.LiveServerMessage(
        server_content=genai_types.LiveServerContent(
            output_transcription=genai_types.Transcription(text=text, finished=True), turn_complete=True
        )
    )


class _ApiClient:
    """The private client attribute `GoogleProvider.base_url` reads."""

    def __init__(self) -> None:
        self._http_options = SimpleNamespace(base_url='https://generativelanguage.googleapis.com/', headers={})


def _fake_client(session: _RecordingSession, captured: dict[str, Any] | None = None) -> Client:
    """A fake `google-genai` client whose `.aio.live.connect(...)` yields `session` (recording `model`/`config`)."""

    class _FakeConnect:
        async def __aenter__(self) -> _RecordingSession:
            return session

        async def __aexit__(self, *exc: object) -> bool:
            return False

    class _Live:
        def connect(self, *, model: str, config: Any) -> _FakeConnect:
            if captured is not None:
                captured['model'] = model
                captured['config'] = config
            return _FakeConnect()

    class _Aio:
        def __init__(self) -> None:
            self.live = _Live()

    class _Client:
        def __init__(self) -> None:
            self.aio = _Aio()
            # `GoogleProvider.base_url` reads this, and the connection reports it as the provider URL
            # that prices a session's usage.
            self._api_client = _ApiClient()
            self.vertexai = False

    return cast('Client', _Client())


def _model(session: _RecordingSession, captured: dict[str, Any] | None = None, **kwargs: Any) -> GoogleRealtimeModel:
    """A `GoogleRealtimeModel` whose provider reuses a fake client backed by `session`."""
    return GoogleRealtimeModel(
        'gemini-2.5-flash-native-audio-latest',
        provider=GoogleProvider(client=_fake_client(session, captured)),
        **kwargs,
    )


async def test_connect_streams_events() -> None:
    # Two turns: `receive()` yields one turn per call, so the connection must loop to serve both
    # (a single `receive()` would stop the session after the first reply).
    session = _RecordingSession([[_turn('hi')], [_turn('bye')]])
    captured: dict[str, Any] = {}
    model = _model(session, captured)
    async with _connect(model, 'x') as conn:
        events = [e async for e in conn]
    assert captured['model'] == 'gemini-2.5-flash-native-audio-latest'
    # Both turns stream, then the server closes the socket; without a reconnect policy that surfaces a
    # non-recoverable `RealtimeSessionErrorEvent` before the stream ends (see `test_iter_ends_on_api_error_close`).
    assert events[:4] == [
        OutputTranscript(text='hi', is_final=True),
        ResponseDone(interrupted=False),
        OutputTranscript(text='bye', is_final=True),
        ResponseDone(interrupted=False),
    ]
    assert isinstance(events[-1], RealtimeSessionErrorEvent) and events[-1].recoverable is False
    assert events[-1].message.startswith('Gemini Live connection closed: ')


async def test_connect_maps_rejected_config_to_model_http_error() -> None:
    # A rejected session config (here an unsupported voice) closes the WebSocket, which the SDK raises as
    # an `APIError` carrying the close code and reason. `connect` maps it to `ModelHTTPError` like a
    # regular `GoogleModel` request, rather than leaking the raw SDK error, so users can handle realtime
    # and non-realtime failures uniformly.
    reason = 'No matching speaker voice found for name: alloy'
    response = httpx.Response(429, headers={'Retry-After': '5', 'X-Request-ID': 'request-123'})

    class _RejectingConnect:
        async def __aenter__(self) -> Any:
            raise genai_errors.APIError(1007, reason, response)

        async def __aexit__(self, *exc: object) -> bool:  # pragma: no cover
            return False

    class _Live:
        def connect(self, *, model: str, config: Any) -> _RejectingConnect:
            return _RejectingConnect()

    client = cast('Client', type('_C', (), {'aio': type('_A', (), {'live': _Live()})(), '_api_client': _ApiClient()})())
    model = GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest', provider=GoogleProvider(client=client))
    with pytest.raises(ModelHTTPError) as exc_info:
        async with _connect(model, 'x'):
            pass  # pragma: no cover
    assert exc_info.value.status_code == 1007
    assert exc_info.value.model_name == 'gemini-2.5-flash-native-audio-latest'
    assert exc_info.value.body == reason
    assert exc_info.value.headers == {'retry-after': '5', 'x-request-id': 'request-123'}


async def test_connect_maps_websocket_invalid_status_to_model_http_error() -> None:
    # A rejected WebSocket upgrade (e.g. a bad key → 401) surfaces from `google-genai` as a raw
    # `websockets.InvalidStatus`, not an `APIError`. The WebSocket is the API here, so its HTTP status
    # maps to `ModelHTTPError` rather than escaping untyped.
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus
    from websockets.http11 import Response

    class _RejectingConnect:
        async def __aenter__(self) -> Any:
            raise InvalidStatus(Response(401, 'Unauthorized', Headers({'Retry-After': '5'}), body=b'bad key'))

        async def __aexit__(self, *exc: object) -> bool:  # pragma: no cover
            return False

    class _Live:
        def connect(self, *, model: str, config: Any) -> _RejectingConnect:
            return _RejectingConnect()

    client = cast('Client', type('_C', (), {'aio': type('_A', (), {'live': _Live()})(), '_api_client': _ApiClient()})())
    model = GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest', provider=GoogleProvider(client=client))
    with pytest.raises(ModelHTTPError) as exc_info:
        async with _connect(model, 'x'):
            pass  # pragma: no cover
    assert exc_info.value.status_code == 401
    assert exc_info.value.body == 'bad key'
    assert exc_info.value.headers == {'retry-after': '5'}


async def test_connect_maps_other_websocket_errors_to_model_api_error() -> None:
    # A handshake failure with no HTTP status (DNS, TLS, protocol) reaches us as a bare
    # `websockets.WebSocketException`. There's no status to report, so it becomes a `ModelAPIError`
    # rather than escaping untyped — the sibling of the `InvalidStatus` → `ModelHTTPError` mapping.
    class _FailingConnect:
        async def __aenter__(self) -> Any:
            raise WebSocketException('handshake went sideways')

        async def __aexit__(self, *exc: object) -> bool:  # pragma: no cover
            return False

    class _Live:
        def connect(self, *, model: str, config: Any) -> _FailingConnect:
            return _FailingConnect()

    client = cast('Client', type('_C', (), {'aio': type('_A', (), {'live': _Live()})(), '_api_client': _ApiClient()})())
    model = GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest', provider=GoogleProvider(client=client))
    with pytest.raises(ModelAPIError) as exc_info:
        async with _connect(model, 'x'):
            pass  # pragma: no cover
    assert exc_info.value.message == snapshot('WebSocket error during connect: handshake went sideways')


async def test_connect_maps_unreachable_api_to_model_api_error() -> None:
    # The connection never came up at all (DNS, refused, reset, dial timeout). The SDK doesn't wrap
    # these, so without mapping the caller would get a bare `OSError` from what looks like an ordinary
    # model call; there is no HTTP status, so it becomes a `ModelAPIError`.
    class _UnreachableConnect:
        async def __aenter__(self) -> Any:
            raise ConnectionRefusedError('connection refused')

        async def __aexit__(self, *exc: object) -> bool:  # pragma: no cover
            return False

    class _Live:
        def connect(self, *, model: str, config: Any) -> _UnreachableConnect:
            return _UnreachableConnect()

    client = cast('Client', type('_C', (), {'aio': type('_A', (), {'live': _Live()})(), '_api_client': _ApiClient()})())
    model = GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest', provider=GoogleProvider(client=client))
    with pytest.raises(ModelAPIError) as exc_info:
        async with _connect(model, 'x'):
            pass  # pragma: no cover
    assert exc_info.value.message == snapshot('Could not reach the realtime API: connection refused')


async def test_connect_continues_after_empty_server_turn() -> None:
    session = _RecordingSession([[], [_turn('hi')]])

    events = [event async for event in _conn(session)]

    assert events[:2] == [OutputTranscript(text='hi', is_final=True), ResponseDone(interrupted=False)]
    assert isinstance(events[-1], RealtimeSessionErrorEvent)


async def test_connect_seeds_message_history(monkeypatch: pytest.MonkeyPatch) -> None:
    async def download_image(*args: Any, **kwargs: Any) -> Any:
        return {'data': b'url-image', 'data_type': 'image/png'}

    session = _RecordingSession([[_turn('hi')]])

    history = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='sys'),
                UserPromptPart(content=['earlier question', TextContent(' with context'), CachePoint()]),
                UserPromptPart(content=[CachePoint(), '']),
                SpeechPart(speaker='user', transcript=''),
            ]
        ),
        ModelResponse(
            parts=[
                ThinkingPart(
                    content='reasoning',
                    signature='session-bound',
                    provider_name='google',
                    provider_details={'thought_signature': 'secret'},
                ),
                ThinkingPart(content='', signature='signature-only', provider_name='google'),
                TextPart(content=''),
                TextPart(content='earlier answer'),
                SpeechPart(speaker='assistant', transcript=''),
                NativeToolCallPart(tool_name='web_search', args={}, tool_call_id='native-call'),
                NativeToolReturnPart(tool_name='web_search', content='native metadata', tool_call_id='native-call'),
                ToolCallPart(tool_name='weather', args={'city': 'Paris'}, tool_call_id='call-1'),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='weather',
                    content=[
                        'sunny',
                        BinaryContent(data=b'tool-image', media_type='image/png', identifier='weather.png'),
                    ],
                    tool_call_id='call-1',
                ),
                ToolReturnPart(tool_name='plain', content='ok', tool_call_id='plain-call'),
                RetryPromptPart(tool_name='weather', content='invalid city', tool_call_id='call-1'),
                RetryPromptPart(content='answer in prose'),
                UserPromptPart(
                    content=[
                        ImageUrl(url='https://example.com/a.png'),
                        BinaryContent(data=b'inline-image', media_type='image/png'),
                    ]
                ),
                SpeechPart(speaker='user', transcript='spoken question'),
            ]
        ),
        ModelResponse(parts=[SpeechPart(speaker='assistant', transcript='spoken answer')]),
    ]
    monkeypatch.setattr('pydantic_ai.realtime._utils.download_item', download_image)
    model = _model(session)
    async with _connect(model, 'x', messages=history) as conn:
        _ = [e async for e in conn]

    seeded = session.client_content[0]
    assert seeded['turn_complete'] is False
    turns = seeded['turns']
    assert [turn.model_dump(exclude_none=True) for turn in turns] == snapshot(
        [
            {
                'parts': [{'text': 'earlier question'}, {'text': ' with context'}],
                'role': 'user',
            },
            {
                'parts': [
                    {'text': '<think>\nreasoning\n</think>'},
                    {'text': 'earlier answer'},
                    {'text': '[Tool call-1: weather({"city":"Paris"})]'},
                ],
                'role': 'model',
            },
            {
                'parts': [
                    {'text': '[Tool call-1: weather returned: ["sunny","See file weather.png."]]'},
                    {'text': 'This is file weather.png:'},
                    {'inline_data': {'data': b'tool-image', 'mime_type': 'image/png'}},
                    {'text': '[Tool plain-call: plain returned: ok]'},
                    {'text': '[Tool call-1: weather error: invalid city\n\nFix the errors and try again.]'},
                    {'text': 'Validation feedback:\nanswer in prose\n\nFix the errors and try again.'},
                    {'inline_data': {'data': b'url-image', 'mime_type': 'image/png'}},
                    {'inline_data': {'data': b'inline-image', 'mime_type': 'image/png'}},
                    {'text': 'spoken question'},
                ],
                'role': 'user',
            },
            {'parts': [{'text': 'spoken answer'}], 'role': 'model'},
        ]
    )
    assert 'session-bound' not in repr(turns)
    assert 'thought_signature' not in repr(turns)


async def test_connect_seed_projects_tool_calls_as_text() -> None:
    session = _RecordingSession([[_turn('hi')]])
    history = [ModelResponse(parts=[ToolCallPart(tool_name='t', args='{}', tool_call_id='call-1')])]
    model = _model(session)
    async with _connect(model, 'x', messages=history) as conn:
        _ = [e async for e in conn]

    turns = session.client_content[0]['turns']
    assert [(t.role, [p.text for p in t.parts]) for t in turns] == [('model', ['[Tool call-1: t({})]'])]


async def test_connect_rejects_audio_only_user_turn() -> None:
    session = _RecordingSession()
    history = [
        ModelRequest(parts=[SpeechPart(speaker='user', audio=BinaryContent(data=b'pcm-audio', media_type='audio/pcm'))])
    ]

    with pytest.raises(UserError, match='google realtime history seeding does not support retained user audio'):
        async with _connect(_model(session), 'x', messages=history):
            pass  # pragma: no cover


async def test_connect_rejects_unseedable_response_parts() -> None:
    session = _RecordingSession()
    async with _connect(
        _model(session),
        'x',
        messages=[
            ModelRequest(parts=[SpeechPart(speaker='user')]),
            ModelResponse(parts=[SpeechPart(speaker='assistant')]),
        ],
    ):
        pass
    assert session.client_content == []

    history = [ModelResponse(parts=[FilePart(content=BinaryContent(data=b'file', media_type='application/pdf'))])]
    with pytest.raises(UserError, match=re.escape('`FilePart`')):
        async with _connect(_model(_RecordingSession()), 'x', messages=history):
            pass  # pragma: no cover


async def test_connect_seed_skips_compaction_parts() -> None:
    # Provider-session-bound compaction state can't round-trip into another session; like the classic
    # model adapters crossing APIs, seeding skips it silently rather than erroring.
    session = _RecordingSession()
    history = [ModelResponse(parts=[CompactionPart(content='summary'), TextPart(content='the answer')])]
    async with _connect(_model(session), 'x', messages=history):
        pass
    turns = session.client_content[0]['turns']
    assert [part.text for turn in turns for part in turn.parts] == ['the answer']


async def test_connect_reconnect_auto_enables_session_resumption() -> None:
    # A `reconnect` policy alone (here a model-level default via `settings=`) is enough: session
    # resumption is requested automatically, so the server restores state when the connection re-dials.
    captured: dict[str, Any] = {}
    on = _model(
        _RecordingSession([[_turn('hi')]]),
        captured,
        settings=GoogleRealtimeModelSettings(reconnect={}),
    )
    async with _connect(on, 'x') as conn:
        assert conn._dial is not None and conn._reconnect is not None  # pyright: ignore[reportPrivateUsage]
    assert captured['config'].session_resumption == genai_types.SessionResumptionConfig(handle=None)

    # An explicit `google_enable_session_resumption=True` without a policy still just requests
    # handles; nothing re-dials.
    captured = {}
    handles_only = _model(
        _RecordingSession([[_turn('hi')]]),
        captured,
        settings=GoogleRealtimeModelSettings(google_enable_session_resumption=True),
    )
    async with _connect(handles_only, 'x') as conn:
        assert conn._dial is None and conn._reconnect is None  # pyright: ignore[reportPrivateUsage]
    assert captured['config'].session_resumption == genai_types.SessionResumptionConfig(handle=None)


async def test_connect_reconnect_from_session_model_settings() -> None:
    # A per-session policy (via `model_settings=`) enables reconnect + resumption on a model with no
    # defaults, following the standard model-settings layering.
    captured: dict[str, Any] = {}
    model = _model(_RecordingSession([[_turn('hi')]]), captured)
    async with _connect(model, 'x', model_settings={'reconnect': {}}) as conn:
        assert conn._dial is not None and conn._reconnect is not None  # pyright: ignore[reportPrivateUsage]
    assert captured['config'].session_resumption == genai_types.SessionResumptionConfig(handle=None)


async def test_connect_rejects_reconnect_with_resumption_disabled() -> None:
    # An explicit `google_enable_session_resumption=False` can't be combined with a `reconnect`
    # policy: a re-dial without resumption would lose the conversation, so `connect` fails loudly
    # before dialing rather than silently reconnecting into a model that remembers nothing.
    captured: dict[str, Any] = {}
    model = _model(
        _RecordingSession(),
        captured,
        settings=GoogleRealtimeModelSettings(reconnect={}, google_enable_session_resumption=False),
    )
    with pytest.raises(UserError, match='requires Gemini session resumption'):
        async with _connect(model, 'x'):
            pass  # pragma: no cover
    assert 'config' not in captured  # no socket was dialed


async def test_iter_ends_on_api_error_close() -> None:
    # The SDK surfaces a server-closed socket as an `APIError`; without a reconnect policy iteration
    # should end (not raise) but first surface a non-recoverable `RealtimeSessionErrorEvent` so callers can tell a
    # dropped connection from a completed turn (mirroring the OpenAI provider).
    session = _RecordingSession([[_turn('hi')]], close_exc=genai_errors.APIError(1011, {'message': 'go away'}))
    events = [e async for e in _conn(session)]
    assert events[:2] == [OutputTranscript(text='hi', is_final=True), ResponseDone(interrupted=False)]
    assert isinstance(events[-1], RealtimeSessionErrorEvent) and events[-1].recoverable is False


async def test_iter_ends_on_oserror() -> None:
    session = _RecordingSession(close_exc=ConnectionResetError('connection reset'))

    events = [event async for event in _conn(session)]

    assert events == [
        RealtimeSessionErrorEvent(message='Gemini Live connection closed: connection reset', recoverable=False)
    ]


# --- config: voice / tone / turn-taking knobs --------------------------------


def test_speech_config_voice_and_language() -> None:
    speech = (
        GoogleRealtimeModel(
            'gemini-2.5-flash-native-audio-latest',
            settings=GoogleRealtimeModelSettings(google_voice='Puck', google_language_code='pl-PL'),
        )
        ._config('hi', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]
        .speech_config
    )
    assert speech is not None
    assert speech.language_code == 'pl-PL'
    assert speech.voice_config.prebuilt_voice_config.voice_name == 'Puck'  # type: ignore[union-attr]
    assert speech.multi_speaker_voice_config is None


def test_speech_config_multi_speaker_overrides_voice() -> None:
    # Multi-speaker and single-voice configs are mutually exclusive in the API, so multi-speaker wins.
    model = GoogleRealtimeModel(
        'gemini-2.5-flash-native-audio-latest',
        settings=GoogleRealtimeModelSettings(
            google_voice='Puck', google_multi_speaker={'voices': {'Joe': 'Puck', 'Jane': 'Kore'}}
        ),
    )
    speech = model._config('hi', None, model_settings=None).speech_config  # pyright: ignore[reportPrivateUsage]
    assert speech is not None
    assert speech.voice_config is None
    speakers = speech.multi_speaker_voice_config.speaker_voice_configs  # type: ignore[union-attr]
    assert [s.speaker for s in speakers] == ['Joe', 'Jane']  # type: ignore[union-attr]
    assert speakers[1].voice_config.prebuilt_voice_config.voice_name == 'Kore'  # type: ignore[union-attr,index]


def test_speech_config_absent_when_unset() -> None:
    assert (
        GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest')
        ._config('hi', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]
        .speech_config
        is None
    )


def test_realtime_input_full() -> None:
    model = GoogleRealtimeModel(
        'gemini-2.5-flash-native-audio-latest',
        settings=GoogleRealtimeModelSettings(
            google_vad={'start_sensitivity': 'high', 'end_sensitivity': 'low', 'silence_duration_ms': 300},
            google_activity_handling='no_interruption',
            google_turn_coverage='all_video',
        ),
    )
    rt = model._config('hi', None, model_settings=None).realtime_input_config  # pyright: ignore[reportPrivateUsage]
    assert rt is not None
    detection = rt.automatic_activity_detection
    assert detection.start_of_speech_sensitivity == genai_types.StartSensitivity.START_SENSITIVITY_HIGH  # type: ignore[union-attr]
    assert detection.end_of_speech_sensitivity == genai_types.EndSensitivity.END_SENSITIVITY_LOW  # type: ignore[union-attr]
    assert detection.silence_duration_ms == 300  # type: ignore[union-attr]
    assert rt.activity_handling == genai_types.ActivityHandling.NO_INTERRUPTION
    assert rt.turn_coverage == genai_types.TurnCoverage.TURN_INCLUDES_AUDIO_ACTIVITY_AND_ALL_VIDEO


@pytest.mark.parametrize('sensitivity', ['low', 'medium', 'high'])
def test_cross_provider_turn_detection_sensitivity(sensitivity: Literal['low', 'medium', 'high']) -> None:
    # Resolve the expected `genai_types` enums inside the test (not in the `parametrize` decorator, which
    # is evaluated at collection time before the module-level skip can apply when `google-genai` is absent).
    expected_start, expected_end = {
        'low': (genai_types.StartSensitivity.START_SENSITIVITY_LOW, genai_types.EndSensitivity.END_SENSITIVITY_LOW),
        'medium': (None, None),
        'high': (genai_types.StartSensitivity.START_SENSITIVITY_HIGH, genai_types.EndSensitivity.END_SENSITIVITY_HIGH),
    }[sensitivity]
    config = GoogleRealtimeModel(
        'gemini-2.5-flash-native-audio-latest',
        settings=GoogleRealtimeModelSettings(turn_detection={'sensitivity': sensitivity}),
    )._config('hi', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]
    realtime_input_config = config.realtime_input_config
    assert realtime_input_config is not None
    detection = realtime_input_config.automatic_activity_detection
    assert detection is not None
    assert detection.start_of_speech_sensitivity == expected_start
    assert detection.end_of_speech_sensitivity == expected_end


def test_google_vad_overrides_cross_provider_turn_detection() -> None:
    config = GoogleRealtimeModel(
        'gemini-2.5-flash-native-audio-latest',
        settings=GoogleRealtimeModelSettings(
            turn_detection={'sensitivity': 'high'},
            google_vad={'start_sensitivity': 'low', 'end_sensitivity': 'low'},
        ),
    )._config('hi', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]
    realtime_input_config = config.realtime_input_config
    assert realtime_input_config is not None
    detection = realtime_input_config.automatic_activity_detection
    assert detection is not None
    assert detection.start_of_speech_sensitivity == genai_types.StartSensitivity.START_SENSITIVITY_LOW
    assert detection.end_of_speech_sensitivity == genai_types.EndSensitivity.END_SENSITIVITY_LOW


def test_cross_provider_turn_detection_false_is_rejected() -> None:
    """Gemini has no manual turn controls, so disabling VAD (push-to-talk) fails loudly rather than
    producing an unusable session."""
    model = GoogleRealtimeModel(
        'gemini-2.5-flash-native-audio-latest', settings=GoogleRealtimeModelSettings(turn_detection=False)
    )
    with pytest.raises(UserError, match='does not support disabling automatic turn detection'):
        model._config('hi', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]


def test_google_vad_disabled_is_rejected() -> None:
    model = GoogleRealtimeModel(
        'gemini-2.5-flash-native-audio-latest', settings=GoogleRealtimeModelSettings(google_vad={'disabled': True})
    )

    with pytest.raises(UserError, match='does not support disabling automatic turn detection'):
        model._config('hi', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]


def test_realtime_input_absent_when_unset() -> None:
    # no vad, no activity handling, no turn coverage → no realtime input config at all.
    assert (
        GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest')
        ._config('hi', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]
        .realtime_input_config
        is None
    )


def test_vad_without_sensitivities() -> None:
    # a bare `{}` sets a detection block but leaves sensitivities/disabled unset.
    rt = (
        GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest', settings=GoogleRealtimeModelSettings(google_vad={}))
        ._config(  # pyright: ignore[reportPrivateUsage]
            'hi', None, model_settings=None
        )
        .realtime_input_config
    )
    detection = rt.automatic_activity_detection  # type: ignore[union-attr]
    assert detection.disabled is None  # type: ignore[union-attr]
    assert detection.start_of_speech_sensitivity is None  # type: ignore[union-attr]
    assert detection.end_of_speech_sensitivity is None  # type: ignore[union-attr]


def test_affective_and_proactive_audio() -> None:
    config = GoogleRealtimeModel(
        'gemini-2.5-flash-native-audio-latest',
        settings=GoogleRealtimeModelSettings(google_affective_dialog=True, google_proactive_audio=True),
    )._config('hi', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]
    assert config.enable_affective_dialog is True
    assert config.proactivity.proactive_audio is True  # type: ignore[union-attr]


def test_affective_and_proactive_default_off() -> None:
    config = GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest')._config('hi', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]
    assert config.enable_affective_dialog is None
    assert config.proactivity is None


def test_transcription_language_codes() -> None:
    config = GoogleRealtimeModel(
        'gemini-2.5-flash-native-audio-latest',
        settings=GoogleRealtimeModelSettings(google_transcription_language_codes=['pl-PL']),
    )._config('hi', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]
    assert config.input_audio_transcription.language_codes == ['pl-PL']  # type: ignore[union-attr]
    assert config.output_audio_transcription.language_codes == ['pl-PL']  # type: ignore[union-attr]


def test_context_compression_and_session_resumption() -> None:
    model = GoogleRealtimeModel(
        'gemini-2.5-flash-native-audio-latest',
        settings=GoogleRealtimeModelSettings(
            google_context_compression={'trigger_tokens': 8000, 'target_tokens': 4000},
            google_enable_session_resumption=True,
        ),
    )
    config = model._config('hi', None, model_settings=None)  # pyright: ignore[reportPrivateUsage]
    cwc = config.context_window_compression
    assert cwc.trigger_tokens == 8000  # type: ignore[union-attr]
    assert cwc.sliding_window.target_tokens == 4000  # type: ignore[union-attr]
    # resumption requested with no handle on first connect.
    assert config.session_resumption is not None and config.session_resumption.handle is None


def test_session_resumption_passes_handle() -> None:
    config = GoogleRealtimeModel(
        'gemini-2.5-flash-native-audio-latest',
        settings=GoogleRealtimeModelSettings(google_enable_session_resumption=True),
    )._config(  # pyright: ignore[reportPrivateUsage]
        'hi', None, model_settings=None, resumption_handle='h9'
    )
    assert config.session_resumption.handle == 'h9'  # type: ignore[union-attr]


def test_generation_params_from_model_settings() -> None:
    settings = GoogleRealtimeModelSettings(
        temperature=0.3,
        top_p=0.8,
        top_k=20,
        max_tokens=128,
        seed=7,
        google_thinking_config={'thinking_budget': 100},
        google_video_resolution=genai_types.MediaResolution.MEDIA_RESOLUTION_LOW,
    )
    config = GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest')._config('hi', None, model_settings=settings)  # pyright: ignore[reportPrivateUsage]
    assert config.temperature == 0.3
    assert config.top_p == 0.8
    assert config.top_k == 20
    assert config.max_output_tokens == 128
    assert config.seed == 7
    assert config.thinking_config.thinking_budget == 100  # type: ignore[union-attr]
    assert config.media_resolution == genai_types.MediaResolution.MEDIA_RESOLUTION_LOW


def test_config_overrides_escape_hatch() -> None:
    model = GoogleRealtimeModel(
        'gemini-2.5-flash-native-audio-latest',
        settings=GoogleRealtimeModelSettings(google_config_overrides={'explicit_vad_signal': True}),
    )
    assert model._config('hi', None, model_settings=None).explicit_vad_signal is True  # pyright: ignore[reportPrivateUsage]


# --- reconnect via session resumption ----------------------------------------


def test_map_message_captures_resumption_handle() -> None:
    conn = _conn(_RecordingSession())
    message = genai_types.LiveServerMessage(
        session_resumption_update=genai_types.LiveServerSessionResumptionUpdate(new_handle='h-123', resumable=True)
    )
    assert conn._map_message(message) == []  # pyright: ignore[reportPrivateUsage] # internal state, not an event
    assert conn._resumption_handle == 'h-123'  # pyright: ignore[reportPrivateUsage]


def _dialer(*sessions: _RecordingSession) -> tuple[Any, list[str | None]]:
    """A `dial` that hands out `sessions` in order, then fails — records the handles it was called with."""
    handles: list[str | None] = []
    pending = iter(sessions)

    async def dial(handle: str | None) -> AsyncSession:
        handles.append(handle)
        try:
            return cast('AsyncSession', next(pending))
        except StopIteration:
            raise ConnectionClosed(None, None)

    return dial, handles


async def test_reconnect_resumes_then_gives_up() -> None:
    # s1 drops at once; reconnect resumes into s2 (one turn, then drops); reconnect then runs out.
    s1 = _RecordingSession([])
    s2 = _RecordingSession([[_turn('back')]])
    dial, handles = _dialer(s2)
    conn = GoogleRealtimeConnection(
        cast('AsyncSession', s1), dial=dial, reconnect={'base_delay': 0.0, 'max_attempts': 2, 'jitter': False}
    )
    conn._resumption_handle = 'h1'  # pyright: ignore[reportPrivateUsage]
    events = [e async for e in conn]
    assert events[:3] == [
        RealtimeSessionReconnectEvent(state_restored=True),
        OutputTranscript(text='back', is_final=True),
        ResponseDone(interrupted=False),
    ]
    assert isinstance(events[-1], RealtimeSessionErrorEvent) and events[-1].recoverable is False
    # reconnect resumed from the stored handle; one success + two failed attempts.
    assert handles == ['h1', 'h1', 'h1']


@pytest.mark.parametrize('handle', [None, 'resume-me'])
async def test_reconnect_reports_whether_state_was_actually_restored(handle: str | None) -> None:
    # `state_restored` tells the consumer whether to treat the reconnect as a fresh turn, so it has to
    # follow the resumption handle. Gemini only sends one once the session is under way: a socket that
    # drops before then reconnects into a genuinely empty session, however resumption was configured.
    messages = (
        []
        if handle is None
        else [
            genai_types.LiveServerMessage(
                session_resumption_update=genai_types.LiveServerSessionResumptionUpdate(new_handle=handle)
            )
        ]
    )
    s1 = _RecordingSession([messages] if messages else [])
    dial, _ = _dialer(_RecordingSession([[_turn('back')]]))
    conn = GoogleRealtimeConnection(
        cast('AsyncSession', s1), dial=dial, reconnect={'base_delay': 0.0, 'max_attempts': 1, 'jitter': False}
    )

    events = [e async for e in conn]

    reconnects = [e for e in events if isinstance(e, RealtimeSessionReconnectEvent)]
    assert reconnects == [RealtimeSessionReconnectEvent(state_restored=handle is not None)]


async def test_reconnect_closes_orphaned_turn_with_interrupted_boundary() -> None:
    # s1 completes one turn, then drops mid-way through a second (output streamed, no `turn_complete`).
    # The re-dialed connection never continues an in-flight generation — session resumption restores
    # conversation state, not the generation (verified live: a resumed session stays silent) — so the
    # orphaned turn's boundary would never arrive. The connection closes it with an interrupted
    # `ResponseDone` ahead of the reconnect event; without one the session keeps the partial response
    # open forever, never ending the turn or delivering messages queued behind it. The completed first
    # turn doesn't arm this: only output since the last boundary marks a turn open.
    partial = genai_types.LiveServerMessage(
        server_content=genai_types.LiveServerContent(
            output_transcription=genai_types.Transcription(text='cut off', finished=False)
        )
    )
    s1 = _RecordingSession([[_turn('done')], [partial]])
    dial, _ = _dialer(_RecordingSession([[_turn('back')]]))
    conn = GoogleRealtimeConnection(
        cast('AsyncSession', s1), dial=dial, reconnect={'base_delay': 0.0, 'max_attempts': 1, 'jitter': False}
    )
    conn._resumption_handle = 'h1'  # pyright: ignore[reportPrivateUsage]

    events = [e async for e in conn]

    assert events[:6] == [
        OutputTranscript(text='done', is_final=True),
        ResponseDone(interrupted=False),
        OutputTranscript(text='cut off', is_final=False),
        ResponseDone(interrupted=True),
        RealtimeSessionReconnectEvent(state_restored=True),
        OutputTranscript(text='back', is_final=True),
    ]


async def test_reconnect_closes_orphaned_turn_opened_by_a_tool_call() -> None:
    # A tool call opens the turn like audio output does: the session holds a partial response for
    # it, so a socket that drops between the `toolCall` and `turn_complete` needs the same synthetic
    # interrupted boundary — otherwise the turn (and every message queued behind it) stalls forever.
    tool_call = genai_types.LiveServerMessage(
        tool_call=genai_types.LiveServerToolCall(
            function_calls=[genai_types.FunctionCall(id='c1', name='get_weather', args={})]
        )
    )
    s1 = _RecordingSession([[tool_call]])
    dial, _ = _dialer(_RecordingSession([[_turn('back')]]))
    conn = GoogleRealtimeConnection(
        cast('AsyncSession', s1), dial=dial, reconnect={'base_delay': 0.0, 'max_attempts': 1, 'jitter': False}
    )
    conn._resumption_handle = 'h1'  # pyright: ignore[reportPrivateUsage]

    events = [e async for e in conn]

    assert events[:4] == [
        ToolCall(tool_call_id='c1', tool_name='get_weather', args='{}'),
        ResponseDone(interrupted=True),
        RealtimeSessionReconnectEvent(state_restored=True),
        OutputTranscript(text='back', is_final=True),
    ]


async def test_reconnect_without_state_abandons_outstanding_tool_calls() -> None:
    # With no resumption handle the re-dialed session is a brand new one that never issued the calls
    # the lost session did, so a tool task still running against one would send its result back
    # against an id Gemini doesn't know. They are abandoned the way Gemini's own
    # `tool_call_cancellation` abandons a call, so each still gets a matching return in history.
    tool_call = genai_types.LiveServerMessage(
        tool_call=genai_types.LiveServerToolCall(
            function_calls=[genai_types.FunctionCall(id='c1', name='get_weather', args={})]
        )
    )
    s1 = _RecordingSession([[tool_call]])
    dial, _ = _dialer(_RecordingSession([[_turn('back')]]))
    conn = GoogleRealtimeConnection(
        cast('AsyncSession', s1), dial=dial, reconnect={'base_delay': 0.0, 'max_attempts': 1, 'jitter': False}
    )

    events = [e async for e in conn]

    assert events[:5] == [
        ToolCall(tool_call_id='c1', tool_name='get_weather', args='{}'),
        ToolCallCancelled(tool_call_ids=['c1']),
        ResponseDone(interrupted=True),
        RealtimeSessionReconnectEvent(state_restored=False),
        OutputTranscript(text='back', is_final=True),
    ]
    assert conn._tool_calls == {}  # pyright: ignore[reportPrivateUsage]


async def test_reconnect_with_restored_state_keeps_outstanding_tool_calls() -> None:
    # A resumption handle means the same server-side session continues, so it still knows the call:
    # the running tool task's result is deliverable and the call must not be abandoned.
    tool_call = genai_types.LiveServerMessage(
        tool_call=genai_types.LiveServerToolCall(
            function_calls=[genai_types.FunctionCall(id='c1', name='get_weather', args={})]
        )
    )
    s1 = _RecordingSession([[tool_call]])
    dial, _ = _dialer(_RecordingSession([[_turn('back')]]))
    conn = GoogleRealtimeConnection(
        cast('AsyncSession', s1), dial=dial, reconnect={'base_delay': 0.0, 'max_attempts': 1, 'jitter': False}
    )
    conn._resumption_handle = 'h1'  # pyright: ignore[reportPrivateUsage]

    events = [e async for e in conn]

    assert not any(isinstance(event, ToolCallCancelled) for event in events)
    assert conn._tool_calls == {'c1': ('get_weather', 'c1')}  # pyright: ignore[reportPrivateUsage]


async def test_reconnect_applies_jitter(monkeypatch: pytest.MonkeyPatch) -> None:
    # With `jitter=True` the backoff delay is scaled by `0.5 + random()*0.5`, so a fixed `random()`
    # of 0.4 turns the first attempt's 0.5s base delay into 0.5 * 0.7 = 0.35s. Capturing the actual
    # slept delay proves jitter is applied (and with a real value, not the un-jittered 0.5s) — a
    # non-zero `base_delay` is required, otherwise the multiply is a no-op and tests nothing.
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    # `reconnect_with_backoff` calls `random.random()` and `asyncio.sleep()` from these module
    # singletons, so patching them here controls the jitter factor and captures the resulting delay.
    monkeypatch.setattr(random, 'random', lambda: 0.4)
    monkeypatch.setattr(asyncio, 'sleep', record_sleep)

    s1 = _RecordingSession([])
    dial, _ = _dialer(_RecordingSession([[_turn('hi')]]))
    conn = GoogleRealtimeConnection(
        cast('AsyncSession', s1), dial=dial, reconnect={'base_delay': 0.5, 'max_attempts': 1, 'jitter': True}
    )
    conn._resumption_handle = 'h1'  # pyright: ignore[reportPrivateUsage]
    events = [e async for e in conn]
    assert events[0] == RealtimeSessionReconnectEvent(state_restored=True)
    # Every backoff delay is the jittered 0.35s, never the un-jittered 0.5s base delay.
    assert delays
    assert all(delay == pytest.approx(0.35) for delay in delays)


async def test_connect_reconnect_closes_previous_session() -> None:
    # End-to-end through `connect()`'s own dial: a reconnect must close the previous connection's
    # context manager before opening the next, so they don't accumulate.
    sessions = iter([_RecordingSession([]), _RecordingSession([[_turn('back')]])])
    closed: list[int] = []

    class _SeqConnect:
        def __init__(self, idx: int, session: _RecordingSession) -> None:
            self._idx, self._session = idx, session

        async def __aenter__(self) -> _RecordingSession:
            return self._session

        async def __aexit__(self, *exc: object) -> bool:
            closed.append(self._idx)
            return False

    class _Live:
        def __init__(self) -> None:
            self.n = 0

        def connect(self, *, model: str, config: Any) -> _SeqConnect:
            try:
                session = next(sessions)
            except StopIteration:
                raise ConnectionClosed(None, None)  # out of sessions → reconnect ultimately fails
            cm = _SeqConnect(self.n, session)
            self.n += 1
            return cm

    class _Aio:
        def __init__(self) -> None:
            self.live = _Live()

    class _Client:
        def __init__(self) -> None:
            self.aio = _Aio()
            self._api_client = _ApiClient()
            self.vertexai = False

    model = GoogleRealtimeModel(
        'gemini-2.5-flash-native-audio-latest',
        provider=GoogleProvider(client=cast('Client', _Client())),
        settings=GoogleRealtimeModelSettings(reconnect={'base_delay': 0.0, 'max_attempts': 1, 'jitter': False}),
    )
    async with _connect(model, 'x') as conn:
        events = [e async for e in conn]
    # `state_restored` is covered by its own test; this one is about closing the previous session's CM.
    assert isinstance(events[0], RealtimeSessionReconnectEvent)
    assert events[1:3] == [OutputTranscript(text='back', is_final=True), ResponseDone(interrupted=False)]
    assert isinstance(events[-1], RealtimeSessionErrorEvent)
    # cm0 closed when reconnecting into cm1; cm1 closed when the next reconnect runs out of sessions.
    assert closed == [0, 1]
