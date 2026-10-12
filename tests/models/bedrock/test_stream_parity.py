"""Every recorded Bedrock `ConverseStream` response builds the same `ModelResponse` as the equivalent `Converse` response.

`BedrockStreamedResponse` must build the same response from the stream's events as `BedrockConverseModel._process_response`
does from the complete `Converse` output. botocore has no `ConverseStream` accumulator, so this folds each recorded
stream's events into the `Converse` output shape (AWS documents `ConverseStream` as the streaming form of `Converse`)
and replays each recording through both.
"""

from __future__ import annotations as _annotations

import dataclasses
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import unquote

import anyio.to_thread
import pytest
from cassetter import Cassette, HttpInteraction, RecordMode

from pydantic_ai.messages import BaseToolCallPart, ModelRequest, ModelResponse, ModelResponsePart
from pydantic_ai.models import ModelRequestParameters

from ...conftest import try_import

with try_import() as imports_successful:
    import boto3
    from botocore.awsrequest import AWSPreparedRequest, AWSResponse
    from botocore.compat import HTTPHeaders

    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
    from mypy_boto3_bedrock_runtime.type_defs import ConverseResponseTypeDef

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='boto3 not installed'),
    pytest.mark.usefixtures('allow_model_requests'),
]

_TESTS_DIR = Path(__file__).parents[2]


def _recorded_streams() -> dict[str, HttpInteraction]:
    streams: dict[str, HttpInteraction] = {}
    for path in sorted(_TESTS_DIR.rglob('*.yaml')):
        if '/converse-stream' not in path.read_text():
            continue
        # cassetter reads both its own format and the vcrpy one older cassettes were recorded in.
        cassette = Cassette(path, record_mode=RecordMode.NONE)
        cassette.load()
        for index, interaction in enumerate(cassette.interactions):
            # An error response has no stream to compare.
            if interaction.request.uri.endswith('/converse-stream') and interaction.response.status == 200:
                streams[f'{path.relative_to(_TESTS_DIR)}#{index}'] = interaction
    return streams


class _RecordedBody:
    """The raw HTTP response botocore reads an event stream from."""

    def __init__(self, body: bytes):
        self.body = body

    def stream(self, *_: Any, **__: Any) -> Iterator[bytes]:
        yield self.body


@dataclass
class _Replay:
    """A real `bedrock-runtime` client whose `ConverseStream` calls get `interaction`'s recorded response."""

    client: BedrockRuntimeClient
    interaction: HttpInteraction | None = None

    def send(self, request: AWSPreparedRequest, **_: Any) -> AWSResponse:
        assert self.interaction is not None
        response = self.interaction.response
        headers = HTTPHeaders.from_dict({key: value[0] for key, value in response.headers.items()})
        body = response.body.content
        assert isinstance(body, bytes)
        return AWSResponse(request.url, response.status, headers, _RecordedBody(body))


@pytest.fixture
def replay() -> Iterator[_Replay]:
    client: BedrockRuntimeClient = boto3.client(  # pyright: ignore[reportUnknownMemberType]
        'bedrock-runtime', region_name='us-east-1', aws_access_key_id='test', aws_secret_access_key='test'
    )
    replay = _Replay(client)
    # A `before-send` handler's response replaces the HTTP request; the stubs type handlers as returning `None`.
    client.meta.events.register('before-send.bedrock-runtime.ConverseStream', replay.send)  # pyright: ignore[reportArgumentType]
    yield replay
    client.close()


def _fold(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold `ConverseStream` events into the `Converse` response they stream."""
    blocks: dict[int, dict[str, Any]] = {}
    tool_inputs: dict[int, str] = {}
    response: dict[str, Any] = {}
    for event in events:
        if start := event.get('contentBlockStart'):
            index, block = start['contentBlockIndex'], start['start']
            if 'toolUse' in block:
                blocks[index] = {'toolUse': dict(block['toolUse'])}
                tool_inputs[index] = ''
            elif 'toolResult' in block:  # pragma: no branch
                blocks[index] = {'toolResult': {**block['toolResult'], 'content': []}}
        elif delta_event := event.get('contentBlockDelta'):
            index, delta = delta_event['contentBlockIndex'], delta_event['delta']
            block = blocks.setdefault(index, {})
            if 'text' in delta:
                block['text'] = block.get('text', '') + delta['text']
            if 'toolUse' in delta:
                tool_inputs[index] += delta['toolUse'].get('input', '')
            if 'toolResult' in delta:
                block['toolResult']['content'].extend(delta['toolResult'])
            if reasoning := delta.get('reasoningContent'):
                target = block.setdefault('reasoningContent', {})
                if 'redactedContent' in reasoning:
                    target['redactedContent'] = target.get('redactedContent', b'') + reasoning['redactedContent']
                else:
                    reasoning_text = target.setdefault('reasoningText', {'text': ''})
                    reasoning_text['text'] += reasoning.get('text', '')
                    if 'signature' in reasoning:
                        reasoning_text['signature'] = reasoning_text.get('signature', '') + reasoning['signature']
        elif message_stop := event.get('messageStop'):
            response['stopReason'] = message_stop['stopReason']
        elif metadata := event.get('metadata'):
            response |= {key: metadata[key] for key in ('usage', 'metrics', 'trace') if key in metadata}
    for index, tool_input in tool_inputs.items():
        blocks[index]['toolUse']['input'] = json.loads(tool_input) if tool_input else {}
    # `Converse` leaves out the whitespace-only text blocks that `ConverseStream` sends, like Qwen3's `''` and `'\n\n'`
    # (live-verified on Qwen3 models; see `test_bedrock_qwen_stream_whitespace_text_blocks`).
    content = [block for _, block in sorted(blocks.items()) if 'text' not in block or block['text'].strip()]
    return {'output': {'message': {'role': 'assistant', 'content': content}}, **response}


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
    # Streamed tool call args are the JSON string the deltas built, where a complete response has a dict.
    return {'args': part.args_as_dict()} if isinstance(part, BaseToolCallPart) else {}


async def _streamed_and_complete(
    interaction: HttpInteraction, replay: _Replay
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_name = unquote(interaction.request.uri.split('/model/')[1].removesuffix('/converse-stream'))
    model = BedrockConverseModel(model_name, provider=BedrockProvider(bedrock_client=replay.client))
    replay.interaction = interaction

    def recorded_response() -> dict[str, Any]:
        output = replay.client.converse_stream(modelId=model_name, messages=[])
        # The request ID comes from the HTTP headers, which a stream and a complete response share.
        return _fold([dict(event) for event in output['stream']]) | {'ResponseMetadata': output['ResponseMetadata']}

    response = cast('ConverseResponseTypeDef', await anyio.to_thread.run_sync(recorded_response))
    complete = await model._process_response(response)  # pyright: ignore[reportPrivateUsage]

    async with model.request_stream([ModelRequest.user_text_prompt('')], None, ModelRequestParameters()) as streamed:
        async for _ in streamed:
            pass
    return _comparable(streamed.get()), _comparable(complete)


async def test_recorded_streams_match_the_complete_response(replay: _Replay) -> None:
    streams = _recorded_streams()
    # Guards against a cassette format change silently leaving nothing to compare.
    assert len(streams) > 10

    streamed: dict[str, dict[str, Any]] = {}
    complete: dict[str, dict[str, Any]] = {}
    for name, interaction in streams.items():
        streamed[name], complete[name] = await _streamed_and_complete(interaction, replay)
    assert streamed == complete
