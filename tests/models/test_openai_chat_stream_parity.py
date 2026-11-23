"""Every recorded streamed Chat Completions response builds the same `ModelResponse` as the completion the SDK accumulates from it.

This replays each recorded stream, from OpenAI and the OpenAI-compatible providers whose model classes build on
`OpenAIChatModel`, through both `OpenAIStreamedResponse` (or the model's subclass of it) and
`OpenAIChatModel._process_response` for the complete `ChatCompletion` that the SDK's `ChatCompletionStreamState`
accumulates from the same chunks, and replays it again re-chunked, as `tests/models/stream_parity.py` describes.

`STREAM_PARITY` says which model class recorded each stream: the class whose claim names the stream's host, else
`OpenAIChatModel`. A stream from a host another driver's class claims, like Groq's, is left to that driver.
"""

from __future__ import annotations as _annotations

import itertools
from collections.abc import Callable
from functools import cache
from random import Random
from typing import Any, cast

import httpx2
import pytest

from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import ModelRequestParameters

from ..cassette_utils import recorded_interactions
from ..conftest import try_import
from .stream_parity import (
    STREAM_PARITY,
    ReplayedBy,
    Strategy,
    StreamParityDriver,
    assert_rechunked_streams_match,
    claim_of,
    claimant,
    comparable_response,
    merge_chat_chunks,
    parse_sse,
    rechunk_sse,
    split_chat_chunk,
)

with try_import() as imports_successful:
    from openai import APIError, AsyncOpenAI, AsyncStream
    from openai._models import construct_type
    from openai.lib.streaming.chat import ChatCompletionStreamState
    from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage

    from pydantic_ai.models.crusoe import CrusoeModel
    from pydantic_ai.models.github_copilot import GitHubCopilotModel
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.models.snowflake import SnowflakeModel
    from pydantic_ai.models.zai import ZaiModel
    from pydantic_ai.providers.crusoe import CrusoeProvider
    from pydantic_ai.providers.deepseek import DeepSeekProvider
    from pydantic_ai.providers.github_copilot import GitHubCopilotProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider
    from pydantic_ai.providers.snowflake import SnowflakeProvider
    from pydantic_ai.providers.zai import ZaiProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


_CLAIM = claim_of('OpenAIChatModel')
_MODEL_CLASSES = frozenset(
    name for name, entry in STREAM_PARITY.items() if isinstance(entry, ReplayedBy) and entry.module == __name__
)


def _model(interaction: dict[str, Any], client: AsyncOpenAI) -> OpenAIChatModel:
    model_name: str = interaction['request']['parsed_body']['model']
    uri: str = interaction['request']['uri']
    factories: dict[str, Callable[[], OpenAIChatModel]] = {
        'GitHubCopilotModel': lambda: GitHubCopilotModel(
            model_name, provider=GitHubCopilotProvider(openai_client=client)
        ),
        'CrusoeModel': lambda: CrusoeModel(model_name, provider=CrusoeProvider(openai_client=client)),
        'ZaiModel': lambda: ZaiModel(model_name, provider=ZaiProvider(openai_client=client)),
        'SnowflakeModel': lambda: SnowflakeModel(model_name, provider=SnowflakeProvider(openai_client=client)),
        'OpenRouterModel': lambda: OpenRouterModel(model_name, provider=OpenRouterProvider(openai_client=client)),
    }
    if factory := factories.get(claimant(uri, _events(interaction)) or ''):
        return factory()
    if 'api.deepseek.com' in uri:
        return OpenAIChatModel(model_name, provider=DeepSeekProvider(openai_client=client))
    return OpenAIChatModel(model_name, provider=OpenAIProvider(openai_client=client))


def _events(interaction: dict[str, Any]) -> str:
    return interaction['response']['body']['string']


# Both tests read the streams, so the cassettes are scanned once.
@cache
def _recorded_streams() -> dict[str, dict[str, Any]]:
    streams: dict[str, dict[str, Any]] = {}
    for name, interaction in recorded_interactions(_CLAIM.endpoint, 'data: {'):
        uri: str = interaction['request']['uri']
        body = interaction['response'].get('body', {}).get('string')
        if (
            uri.endswith(_CLAIM.endpoint)
            and isinstance(body, str)
            # A stream may start with an SSE comment, like OpenRouter's `: OPENROUTER PROCESSING`.
            and body.lstrip().startswith(('data:', ':'))
            and claimant(uri, body) in _MODEL_CLASSES
        ):
            streams[name] = interaction
    return streams


class _CompletionAccumulator:
    """Feeds chunks to the SDK's `ChatCompletionStreamState`, undoing the OpenAI-compatible wire habits it can't take.

    None of these change what a stream means; they only keep the SDK's delta accumulation from corrupting it.
    """

    def __init__(self) -> None:
        self._state = ChatCompletionStreamState()
        self._roles: set[int] = set()
        self._identities: dict[tuple[int, str, int], dict[str, Any]] = {}
        # Values of top-level (`None`) and choice-level (choice index) fields, by the last chunk that sent them.
        self._fields: dict[int | None, dict[str, Any]] = {}
        self._unindexed_lists: dict[tuple[int, str], list[Any]] = {}

    def handle_chunk(self, chunk: ChatCompletionChunk) -> None:
        data = chunk.to_dict()
        # The SDK keeps the first chunk's top-level and choice-level fields (and `usage` from the last chunk, even when
        # that one's is null), where a complete response has the value the stream ended on, e.g. OpenRouter's
        # `native_finish_reason` or a `service_tier` that only a later chunk carries.
        self._fields.setdefault(None, {}).update(
            {key: value for key, value in data.items() if key not in ('choices', 'object') and value not in (None, '')}
        )
        choices: list[dict[str, Any]] = cast(list[dict[str, Any]], data.get('choices') or [])
        for choice in choices:
            index: int = choice.get('index', 0)
            self._fields.setdefault(index, {}).update(
                {
                    key: value
                    for key, value in choice.items()
                    if key not in ('delta', 'index', 'logprobs') and value is not None
                }
            )
            delta: dict[str, Any] = choice.get('delta') or {}
            # OpenAI-compatible providers repeat `role` in every delta, which the SDK would concatenate.
            if 'role' in delta:
                if index in self._roles:
                    del delta['role']
                self._roles.add(index)
            for key, value in list(delta.items()):
                if key == 'tool_calls' or not isinstance(value, list) or not value:
                    continue
                entries = cast(list[dict[str, Any]], value)
                indexed_entries = [entry for entry in entries if 'index' in entry]
                if not indexed_entries:
                    # The SDK rejects lists of objects without an `index` (e.g. OpenRouter's `annotations`), so
                    # they're collected in order and set on the complete message.
                    self._unindexed_lists.setdefault((index, key), []).extend(entries)
                    del delta[key]
                    continue
                for entry in indexed_entries:
                    # Identity fields repeated on every fragment of an indexed entry (e.g. OpenRouter's
                    # `reasoning_details[].id` and `.format`) would be concatenated by the SDK.
                    seen = self._identities.setdefault((index, key, entry['index']), {})
                    for identity in ('id', 'format'):
                        if identity in entry:
                            if seen.get(identity) == entry[identity]:
                                del entry[identity]
                            else:
                                seen[identity] = entry[identity]
        self._state.handle_chunk(cast(ChatCompletionChunk, construct_type(type_=ChatCompletionChunk, value=data)))

    def get_final_completion(self) -> ChatCompletion:
        completion = self._state.get_final_completion()
        top_level_fields = self._fields.get(None, {})
        constructed = construct_type(type_=ChatCompletion, value=top_level_fields)
        for key in top_level_fields:
            setattr(completion, key, getattr(constructed, key))
        for (index, key), entries in self._unindexed_lists.items():
            message = construct_type(type_=ChatCompletionMessage, value={key: entries})
            setattr(completion.choices[index].message, key, getattr(message, key))
        for choice in completion.choices:
            for key, value in self._fields.get(choice.index, {}).items():
                setattr(choice, key, value)
            # A complete message always has a role, which some OpenAI-compatible streams never send.
            choice.message.role = choice.message.role or 'assistant'
        return completion


def _client(interaction: dict[str, Any], events: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key='test',
        base_url=interaction['request']['uri'].removesuffix('/chat/completions'),
        http_client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(
                lambda _: httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=events.encode())
            )
        ),
    )


async def _open_stream(interaction: dict[str, Any], client: AsyncOpenAI) -> AsyncStream[ChatCompletionChunk]:
    return await client.chat.completions.create(
        model=interaction['request']['parsed_body']['model'], messages=[], stream=True
    )


async def _replay(interaction: dict[str, Any], events: str) -> ModelResponse:
    client = _client(interaction, events)
    model = _model(interaction, client)
    # Continuous usage stats change how chunks are read, and aren't recorded in the response.
    stream_options: dict[str, Any] = interaction['request']['parsed_body'].get('stream_options') or {}
    model_settings = OpenAIChatModelSettings(
        openai_continuous_usage_stats=stream_options.get('continuous_usage_stats', False)
    )
    streamed = await model._process_streamed_response(  # pyright: ignore[reportPrivateUsage]
        await _open_stream(interaction, client), ModelRequestParameters(), model_settings
    )
    async for _ in streamed:
        pass
    return streamed.get()


async def _streamed_and_complete(interaction: dict[str, Any]) -> tuple[dict[str, object], dict[str, object]] | None:
    events = _events(interaction)
    client = _client(interaction, events)
    accumulator = _CompletionAccumulator()
    try:
        async for chunk in await _open_stream(interaction, client):
            accumulator.handle_chunk(chunk)
    except APIError:
        return None
    complete = _model(interaction, client)._process_response(accumulator.get_final_completion())  # pyright: ignore[reportPrivateUsage]
    return comparable_response(await _replay(interaction, events)), comparable_response(complete)


async def test_recorded_streams_match_the_complete_completion() -> None:
    streams = _recorded_streams()
    # Guards against a cassette format change silently leaving nothing to compare.
    assert len(streams) > 60

    streamed: dict[str, dict[str, object]] = {}
    complete: dict[str, dict[str, object]] = {}
    errored: list[str] = []
    for name, interaction in streams.items():
        if (result := await _streamed_and_complete(interaction)) is None:
            errored.append(name)
        else:
            streamed[name], complete[name] = result
    # A stream that ends in an error has no complete response to compare with.
    assert errored == ['models/cassettes/test_openrouter/test_openrouter_stream_error.yaml#0']
    assert streamed == complete


def _rechunk(events: str, strategy: Strategy, rng: Random) -> str | None:
    return rechunk_sse(events, strategy, rng, split=split_chat_chunk, merge=merge_chat_chunks)


DRIVER = StreamParityDriver(wire=_events, replay=_replay, rechunk=_rechunk)


@pytest.fixture(scope='module')
def recorded_streams() -> dict[str, dict[str, Any]]:
    return {
        name: interaction
        for name, interaction in _recorded_streams().items()
        # A stream that ends in an error builds no response to rebuild.
        if not any('error' in (event.data or {}) for event in parse_sse(_events(interaction))[0])
    }


# Splits the replays across tests so that no single test carries all of them against the per-test budget.
_STRIDE = 2


# A full sweep over every recording under many seeds stays out of the suite on purpose; see `stream_parity.py`.
@pytest.mark.parametrize('offset', range(_STRIDE))
async def test_rechunked_streams_match_the_recorded_stream(
    offset: int, recorded_streams: dict[str, dict[str, Any]]
) -> None:
    await assert_rechunked_streams_match(DRIVER, itertools.islice(recorded_streams.items(), offset, None, _STRIDE))
