"""Every recorded streamed Responses API response builds the same `ModelResponse` as the complete response it ends with.

`OpenAIResponsesStreamedResponse` must build the same response as `OpenAIResponsesModel._process_response` does for the
complete `Response` that the stream's terminal event (`response.completed`, `.incomplete` or `.failed`) carries. This
replays each recorded stream, from OpenAI and the other providers recorded through the Responses API, through both,
and replays it again re-chunked, as `tests/models/stream_parity.py` describes.
"""

from __future__ import annotations as _annotations

import dataclasses
import itertools
from collections.abc import Sequence
from functools import cache
from random import Random
from typing import Any, cast

import httpx2
import pytest

from pydantic_ai.messages import (
    CompactionPart,
    ModelResponse,
    ModelResponsePart,
    NativeToolCallPart,
    NativeToolReturnPart,
    TextPart,
    ThinkingPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition

from ..cassette_utils import recorded_interactions
from ..conftest import try_import
from .stream_parity import (
    JsonObject,
    SseEvent,
    Strategy,
    StreamParityDriver,
    assert_rechunked_streams_match,
    comparable_response,
    cut,
    parse_sse,
    rechunk_sse,
    render_sse,
    with_field,
)

with try_import() as imports_successful:
    from openai import AsyncAzureOpenAI, AsyncOpenAI
    from openai.types import responses

    from pydantic_ai.models.bedrock_mantle import BedrockMantleResponsesModel
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.models.openai_codex import OpenAICodexModel
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.bedrock_mantle import BedrockMantleProvider
    from pydantic_ai.providers.deepseek import DeepSeekProvider
    from pydantic_ai.providers.meta import MetaProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openai_codex import OpenAICodexProvider
    from pydantic_ai.toolsets._tool_search import TOOL_SEARCH_FUNCTION_TOOL_NAME

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def _response_body(interaction: dict[str, Any]) -> str | None:
    body = interaction['response'].get('body', {})
    content = body.get('string', body.get('content'))
    return content if isinstance(content, str) else None


# Both tests read the streams, so the cassettes are scanned once.
@cache
def _recorded_streams() -> dict[str, dict[str, Any]]:
    streams: dict[str, dict[str, Any]] = {}
    for name, interaction in recorded_interactions('"type":"response.'):
        content = _response_body(interaction)
        # Some providers send each event's `data:` line before its `event:` line.
        if content is not None and content.startswith(('event:', 'data:')) and '"type":"response.' in content:
            streams[name] = interaction
    return streams


def _request_body(interaction: dict[str, Any]) -> dict[str, Any]:
    return interaction['request'].get('parsed_body') or {}


def _model(interaction: dict[str, Any], client: AsyncOpenAI) -> OpenAIResponsesModel:
    # A resumed background stream is a `GET` without a body; its model comes from the stream itself.
    model_name: str = _request_body(interaction).get('model', 'gpt-5')
    url = httpx2.URL(interaction['request']['uri'])
    host = url.host
    if host == 'chatgpt.com':
        return OpenAICodexModel(model_name, provider=OpenAICodexProvider(openai_client=client))
    if host.startswith('bedrock-mantle.'):
        return BedrockMantleResponsesModel(model_name, provider=BedrockMantleProvider(openai_client=client))
    if host == 'api.deepseek.com':
        return OpenAIResponsesModel(model_name, provider=DeepSeekProvider(openai_client=client))
    if host == 'api.meta.ai':
        return OpenAIResponsesModel(model_name, provider=MetaProvider(openai_client=client))
    if host.endswith('.openai.azure.com') or (host == 'gateway.pydantic.info' and url.path.startswith('/proxy/azure/')):
        # The replay client is a plain `AsyncOpenAI`; `AzureProvider` only reads its base URL and API key from it.
        return OpenAIResponsesModel(model_name, provider=AzureProvider(openai_client=cast(AsyncAzureOpenAI, client)))
    assert host == 'api.openai.com', host
    return OpenAIResponsesModel(model_name, provider=OpenAIProvider(openai_client=client))


def _model_request_parameters(interaction: dict[str, Any]) -> ModelRequestParameters:
    # The streamed parts manager types a client-executed tool search call from the `search_tools` definition the
    # request was built from, where the complete response is typed from the output item itself.
    tools: list[dict[str, Any]] = _request_body(interaction).get('tools') or []
    if any(tool['type'] == 'tool_search' and tool.get('execution') == 'client' for tool in tools):
        return ModelRequestParameters(
            function_tools=[ToolDefinition(name=TOOL_SEARCH_FUNCTION_TOOL_NAME, tool_kind='tool-search')]
        )
    return ModelRequestParameters()


def _comparable(response: ModelResponse) -> dict[str, object]:
    return comparable_response(
        dataclasses.replace(response, parts=_mcp_list_tools_results_after_calls(response.parts)), _comparable_details
    )


def _mcp_list_tools_results_after_calls(parts: Sequence[ModelResponsePart]) -> list[ModelResponsePart]:
    # A stream sends each MCP server's tool discovery call as soon as it starts, but the API sends `output_item.done`
    # with the discovered tools for only the last server, so the others' results only arrive with the terminal
    # response, after the text. A complete response lists each result right after its call.
    results = {
        part.tool_call_id: part
        for part in parts
        if isinstance(part, NativeToolReturnPart) and part.tool_call_id.startswith('mcpl_')
    }
    ordered: list[ModelResponsePart] = []
    for part in parts:
        if isinstance(part, NativeToolReturnPart) and part.tool_call_id in results:
            continue
        ordered.append(part)
        if isinstance(part, NativeToolCallPart) and (result := results.get(part.tool_call_id)):
            ordered.append(result)
    return ordered


def _comparable_details(part: ModelResponsePart) -> dict[str, Any]:
    # Reasoning and compaction `encrypted_content` is encrypted anew for every event that carries the item
    # (`output_item.added`, `output_item.done` and the terminal response), so only its presence compares.
    if isinstance(part, ThinkingPart):
        return {'signature': part.signature is not None}
    if isinstance(part, CompactionPart) and part.provider_details:
        return {
            'provider_details': part.provider_details
            | {'encrypted_content': 'encrypted_content' in part.provider_details}
        }
    if isinstance(part, TextPart) and part.provider_details and 'logprobs' in part.provider_details:
        # The `output_text.done` event keeps the logprobs of the deltas where the terminal response rounds them
        # (`-1.9e-07` vs `-0.0`).
        return {
            'provider_details': part.provider_details
            | {
                'logprobs': [
                    logprob
                    | {
                        'logprob': round(logprob['logprob'], 6),
                        'top_logprobs': [
                            top | {'logprob': round(top['logprob'], 6)} for top in logprob['top_logprobs']
                        ],
                    }
                    for logprob in part.provider_details['logprobs']
                ]
            }
        }
    return {}


def _events(interaction: dict[str, Any]) -> str:
    events = _response_body(interaction)
    assert events is not None
    return events


def _client(interaction: dict[str, Any], events: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key='test',
        base_url=str(interaction['request']['uri']).split('/responses')[0],
        http_client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(
                lambda _: httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=events.encode())
            )
        ),
    )


async def _streamed_and_complete(interaction: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    client = _client(interaction, _events(interaction))
    model = _model(interaction, client)
    model_request_parameters = _model_request_parameters(interaction)

    async def open_stream():
        return await client.responses.create(model=model.model_name, input=[], stream=True)

    complete: responses.Response | None = None
    done_items: dict[int, responses.ResponseOutputItem] = {}
    async for event in await open_stream():
        if isinstance(event, responses.ResponseOutputItemDoneEvent):
            done_items[event.output_index] = event.item
        elif isinstance(
            event, (responses.ResponseCompletedEvent, responses.ResponseIncompleteEvent, responses.ResponseFailedEvent)
        ):
            complete = event.response
    assert complete is not None
    if not complete.output:
        # Codex's terminal response leaves `output` empty, so it's built from the `output_item.done` items, as the
        # SDK's own stream accumulator does.
        complete = complete.model_copy(update={'output': [done_items[index] for index in sorted(done_items)]})

    streamed = await model._process_streamed_response(  # pyright: ignore[reportPrivateUsage]
        await open_stream(), {}, model_request_parameters
    )
    async for _ in streamed:
        pass
    return (
        _comparable(streamed.get()),
        _comparable(model._process_response(complete, {}, model_request_parameters)),  # pyright: ignore[reportPrivateUsage]
    )


async def test_recorded_streams_match_the_complete_response() -> None:
    streams = _recorded_streams()
    # Guards against a cassette format change silently leaving nothing to compare.
    assert len(streams) > 40

    streamed: dict[str, dict[str, Any]] = {}
    complete: dict[str, dict[str, Any]] = {}
    for name, interaction in streams.items():
        streamed[name], complete[name] = await _streamed_and_complete(interaction)
    assert streamed == complete


async def _replay(interaction: dict[str, Any], events: str) -> ModelResponse:
    client = _client(interaction, events)
    model = _model(interaction, client)
    streamed = await model._process_streamed_response(  # pyright: ignore[reportPrivateUsage]
        await client.responses.create(model=model.model_name, input=[], stream=True),
        {},
        _model_request_parameters(interaction),
    )
    async for _ in streamed:
        pass
    return streamed.get()


# The events whose `delta` carries a piece of text, thinking, tool input or code, which an event may cut anywhere.
_DELTA_EVENTS = frozenset(
    {
        'response.output_text.delta',
        'response.refusal.delta',
        'response.reasoning_text.delta',
        'response.reasoning_summary_text.delta',
        'response.function_call_arguments.delta',
        'response.custom_tool_call_input.delta',
        'response.mcp_call_arguments.delta',
        'response.code_interpreter_call_code.delta',
    }
)
# The fields that say which block a delta event belongs to.
_BLOCK_FIELDS = ('type', 'item_id', 'output_index', 'content_index', 'summary_index')


def _split(event: JsonObject, rng: Random) -> tuple[JsonObject, JsonObject] | None:
    delta = event.get('delta')
    if event.get('type') not in _DELTA_EVENTS or not isinstance(delta, str) or (pieces := cut(delta, rng)) is None:
        return None
    head = with_field(event, 'delta', pieces[0])
    if 'logprobs' in event:
        head['logprobs'] = []
    return head, with_field(event, 'delta', pieces[1])


def _merge(first: JsonObject, second: JsonObject) -> JsonObject | None:
    first_delta, second_delta = first.get('delta'), second.get('delta')
    if (
        first.get('type') not in _DELTA_EVENTS
        or any(first.get(field) != second.get(field) for field in _BLOCK_FIELDS)
        or not isinstance(first_delta, str)
        or not isinstance(second_delta, str)
    ):
        return None
    merged = with_field(first, 'delta', first_delta + second_delta)
    first_logprobs, second_logprobs = first.get('logprobs'), second.get('logprobs')
    if isinstance(first_logprobs, list) and isinstance(second_logprobs, list):
        merged['logprobs'] = first_logprobs + second_logprobs
    return merged


def _drop_argument_deltas(events: list[SseEvent], rng: Random) -> list[SseEvent] | None:
    """`events` without the argument deltas of a seeded half of the function calls whose `done` event repeats them."""
    streamed: dict[str, str] = {}
    done: dict[str, str] = {}
    for event in events:
        data = event.data or {}
        item_id, delta, arguments = data.get('item_id'), data.get('delta'), data.get('arguments')
        if not isinstance(item_id, str):
            continue
        if data.get('type') == 'response.function_call_arguments.delta' and isinstance(delta, str):
            streamed[item_id] = streamed.get(item_id, '') + delta
        elif data.get('type') == 'response.function_call_arguments.done' and isinstance(arguments, str):
            done[item_id] = arguments
    candidates = sorted(
        item_id for item_id, arguments in streamed.items() if arguments and done.get(item_id) == arguments
    )
    if not candidates:
        return None
    chosen = {item_id for item_id in candidates if rng.random() < 0.5} or {rng.choice(candidates)}
    return [
        event
        for event in events
        if not (
            (data := event.data or {}).get('type') == 'response.function_call_arguments.delta'
            and data.get('item_id') in chosen
        )
    ]


def _rechunk(events: str, strategy: Strategy, rng: Random) -> str | None:
    if strategy != 'drop-argument-deltas':
        return rechunk_sse(events, strategy, rng, split=_split, merge=_merge)
    parsed, newline = parse_sse(events)
    kept = _drop_argument_deltas(parsed, rng)
    return None if kept is None else render_sse(kept, newline)


DRIVER = StreamParityDriver(
    wire=_events, replay=_replay, rechunk=_rechunk, strategies=('split-delta', 'merge-deltas', 'drop-argument-deltas')
)


@pytest.fixture(scope='module')
def recorded_streams() -> dict[str, dict[str, Any]]:
    return _recorded_streams()


# Splits the replays across tests so that no single test carries all of them against the per-test budget.
_STRIDE = 4


# A full sweep over every recording under many seeds stays out of the suite on purpose; see `stream_parity.py`.
@pytest.mark.parametrize('offset', range(_STRIDE))
async def test_rechunked_streams_match_the_recorded_stream(
    offset: int, recorded_streams: dict[str, dict[str, Any]]
) -> None:
    await assert_rechunked_streams_match(DRIVER, itertools.islice(recorded_streams.items(), offset, None, _STRIDE))
