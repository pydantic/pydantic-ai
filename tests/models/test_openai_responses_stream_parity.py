"""Every recorded streamed Responses API response builds the same `ModelResponse` as the complete response it ends with.

`OpenAIResponsesStreamedResponse` must build the same response as `OpenAIResponsesModel._process_response` does for the
complete `Response` that the stream's terminal event (`response.completed`, `.incomplete` or `.failed`) carries. This
replays each recorded stream, from OpenAI and the other providers recorded through the Responses API, through both.
"""

from __future__ import annotations as _annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml

from pydantic_ai.messages import (
    BaseToolCallPart,
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

from ..conftest import try_import

with try_import() as imports_successful:
    from openai import AsyncOpenAI
    from openai.types import responses

    from pydantic_ai.models.bedrock_mantle import BedrockMantleResponsesModel
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.models.openai_codex import OpenAICodexModel
    from pydantic_ai.providers.bedrock_mantle import BedrockMantleProvider
    from pydantic_ai.providers.deepseek import DeepSeekProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openai_codex import OpenAICodexProvider
    from pydantic_ai.toolsets._tool_search import TOOL_SEARCH_FUNCTION_TOOL_NAME

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')

_TESTS_DIR = Path(__file__).parents[1]


def _response_body(interaction: dict[str, Any]) -> str | None:
    body = interaction['response'].get('body', {})
    content = body.get('string', body.get('content'))
    return content if isinstance(content, str) else None


def _recorded_streams() -> dict[str, dict[str, Any]]:
    streams: dict[str, dict[str, Any]] = {}
    for path in sorted(_TESTS_DIR.rglob('*.yaml')):
        text = path.read_text()
        if '"type":"response.' not in text:
            continue
        for index, interaction in enumerate(yaml.safe_load(text)['interactions']):
            content = _response_body(interaction)
            # Some providers send each event's `data:` line before its `event:` line.
            if content is not None and content.startswith(('event:', 'data:')) and '"type":"response.' in content:
                streams[f'{path.relative_to(_TESTS_DIR)}#{index}'] = interaction
    return streams


def _request_body(interaction: dict[str, Any]) -> dict[str, Any]:
    return interaction['request'].get('parsed_body') or {}


def _model(interaction: dict[str, Any], client: AsyncOpenAI) -> OpenAIResponsesModel:
    # A resumed background stream is a `GET` without a body; its model comes from the stream itself.
    model_name: str = _request_body(interaction).get('model', 'gpt-5')
    host = httpx2.URL(interaction['request']['uri']).host
    if host == 'chatgpt.com':
        return OpenAICodexModel(model_name, provider=OpenAICodexProvider(openai_client=client))
    if host.startswith('bedrock-mantle.'):
        return BedrockMantleResponsesModel(model_name, provider=BedrockMantleProvider(openai_client=client))
    if host == 'api.deepseek.com':
        return OpenAIResponsesModel(model_name, provider=DeepSeekProvider(openai_client=client))
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


def _comparable(response: ModelResponse) -> dict[str, Any]:
    fields = {field.name: getattr(response, field.name) for field in dataclasses.fields(response)}
    del fields['timestamp']
    # A return part stamps the time it was built.
    fields['parts'] = [
        {'type': type(part).__name__, **dataclasses.asdict(part), 'timestamp': None} | _comparable_details(part)
        for part in _mcp_list_tools_results_after_calls(response.parts)
    ]
    return fields


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
    if isinstance(part, BaseToolCallPart):
        # Streamed tool call args are the JSON string the deltas built, where a complete response has a dict.
        return {'args': part.args_as_dict()}
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


async def _streamed_and_complete(interaction: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    events = _response_body(interaction)
    assert events is not None
    client = AsyncOpenAI(
        api_key='test',
        base_url=str(interaction['request']['uri']).split('/responses')[0],
        http_client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(
                lambda _: httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=events.encode())
            )
        ),
    )
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
