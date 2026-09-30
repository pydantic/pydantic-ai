"""Every recorded streamed Anthropic response builds the same `ModelResponse` as the message the SDK accumulates from it.

`run()` streams a request whose `max_tokens` is above the SDK's non-streaming limit, which is the default for current
models, so `AnthropicStreamedResponse` must build the same response as `AnthropicModel._process_response` does for a
complete `BetaMessage`. This replays each recorded stream through both.
"""

from __future__ import annotations as _annotations

import dataclasses
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml

from pydantic_ai import _utils
from pydantic_ai.messages import BaseToolCallPart, ModelResponse, ModelResponsePart
from pydantic_ai.models import ModelRequestParameters

from ...conftest import try_import

with try_import() as imports_successful:
    from anthropic import NOT_GIVEN, AsyncAnthropic, AsyncStream
    from anthropic.lib.streaming import BetaAsyncMessageStream
    from anthropic.types.beta import BetaRawMessageStreamEvent, BetaServerToolUseBlock

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='anthropic not installed')

_TESTS_DIR = Path(__file__).parents[2]


def _recorded_streams() -> dict[str, dict[str, Any]]:
    streams: dict[str, dict[str, Any]] = {}
    for path in sorted(_TESTS_DIR.rglob('*.yaml')):
        text = path.read_text()
        if 'event: message_start' not in text:
            continue
        for index, interaction in enumerate(yaml.safe_load(text)['interactions']):
            content = interaction['response']['body'].get('content')
            if isinstance(content, str) and 'event: message_start' in content:
                streams[f'{path.relative_to(_TESTS_DIR)}#{index}'] = interaction
    return streams


def _comparable(response: ModelResponse) -> dict[str, Any]:
    fields = {field.name: getattr(response, field.name) for field in dataclasses.fields(response)}
    del fields['timestamp']
    # A return part stamps the time it was built.
    fields['parts'] = [
        {'type': type(part).__name__, **dataclasses.asdict(part), 'timestamp': None} | _comparable_args(part)
        for part in response.parts
    ]
    return fields


def _comparable_args(part: ModelResponsePart) -> dict[str, Any]:
    # Streamed tool call args are the JSON string the deltas built, where a complete message has a dict.
    return {'args': part.args_as_dict()} if isinstance(part, BaseToolCallPart) else {}


async def _streamed_and_complete(
    interaction: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_name = interaction['request']['body']['content']['model']
    events = interaction['response']['body']['content'].encode()
    client = AsyncAnthropic(
        api_key='test',
        http_client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(
                lambda _: httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=events)
            )
        ),
    )
    model = AnthropicModel(model_name, provider=AnthropicProvider(anthropic_client=client))
    model_request_parameters = ModelRequestParameters()

    async def open_stream() -> AsyncStream[BetaRawMessageStreamEvent]:
        return await client.beta.messages.create(model=model_name, max_tokens=1, messages=[], stream=True)

    message = await BetaAsyncMessageStream(await open_stream(), output_format=NOT_GIVEN).get_final_message()
    # Both drop a server tool call whose tool the request didn't enable, which depends on request parameters the
    # recording doesn't keep, so every server tool in the response counts as enabled.
    server_tool_names = frozenset(block.name for block in message.content if isinstance(block, BetaServerToolUseBlock))

    def enabled_server_tool_names(*_: object) -> frozenset[str]:
        return server_tool_names

    monkeypatch.setattr(model, '_get_enabled_server_tool_names', enabled_server_tool_names)
    complete = model._process_response(message, model_request_parameters, {})  # pyright: ignore[reportPrivateUsage]

    streamed = await model._process_streamed_response(  # pyright: ignore[reportPrivateUsage]
        _utils.PeekableAsyncStream(await open_stream()), model_request_parameters, {}
    )
    async for _ in streamed:
        pass
    return _comparable(streamed.get()), _comparable(complete)


async def test_recorded_streams_match_the_complete_message(monkeypatch: pytest.MonkeyPatch) -> None:
    streams = _recorded_streams()
    # Guards against a cassette format change silently leaving nothing to compare.
    assert len(streams) > 200

    streamed: dict[str, dict[str, Any]] = {}
    complete: dict[str, dict[str, Any]] = {}
    for name, interaction in streams.items():
        streamed[name], complete[name] = await _streamed_and_complete(interaction, monkeypatch)
    assert streamed == complete
