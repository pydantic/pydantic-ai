from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from inline_snapshot import snapshot

from pydantic_ai.messages import ModelRequest
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.realtime import AudioInput, SessionUsageEvent, TextInput
from pydantic_ai.realtime.codec import AudioDelta, InputTranscript, Transcript
from pydantic_ai.usage import RequestUsage

from ..conftest import try_import

with try_import() as imports_successful:
    from aws_sdk_bedrock_runtime.models import (
        BidirectionalOutputPayloadPart,
        InvokeModelWithBidirectionalStreamInputChunk,
        InvokeModelWithBidirectionalStreamOperationInput,
        InvokeModelWithBidirectionalStreamOutputChunk,
    )

    from pydantic_ai.realtime.bedrock import (
        BedrockRealtimeConnection,
        BedrockRealtimeModel,
        BedrockRealtimeModelSettings,
    )

pytestmark = pytest.mark.skipif(
    not imports_successful(), reason='aws-sdk-bedrock-runtime not installed (requires Python 3.12+)'
)


class FakeInputStream:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.closed = False

    async def send(self, value: InvokeModelWithBidirectionalStreamInputChunk) -> None:
        payload = value.value.bytes_
        assert payload is not None
        self.events.append(json.loads(payload)['event'])

    async def close(self) -> None:
        self.closed = True


class FakeReceiver:
    def __init__(self, value: object) -> None:
        self.value = value

    async def receive(self) -> object:
        return self.value


class FakeStream:
    def __init__(self, outputs: list[dict[str, object]] | None = None) -> None:
        self._input_stream = FakeInputStream()
        self.outputs = iter(outputs or [])

    async def await_output(self) -> tuple[object, FakeReceiver]:
        event = next(self.outputs)
        chunk = InvokeModelWithBidirectionalStreamOutputChunk(
            BidirectionalOutputPayloadPart(bytes_=json.dumps({'event': event}).encode())
        )
        return None, FakeReceiver(chunk)

    @property
    def input_stream(self) -> FakeInputStream:
        return self._input_stream


class FakeClient:
    def __init__(self, stream: FakeStream) -> None:
        self.stream = stream
        self.model_id: str | None = None

    async def invoke_model_with_bidirectional_stream(
        self, input: InvokeModelWithBidirectionalStreamOperationInput
    ) -> FakeStream:
        self.model_id = input.model_id
        return self.stream


@pytest.mark.anyio
async def test_handshake_settings_merge_and_inputs() -> None:
    stream = FakeStream()
    client = FakeClient(stream)
    settings = BedrockRealtimeModelSettings(voice='tiffany', temperature=0.2, max_tokens=100)
    overrides = BedrockRealtimeModelSettings(temperature=0.4, bedrock_endpointing_sensitivity='HIGH')
    model = BedrockRealtimeModel(
        provider=client,
        settings=settings,
    )
    async with model.connect(
        messages=[ModelRequest(parts=[], instructions='Be concise.')],
        model_settings=overrides,
        model_request_parameters=ModelRequestParameters(),
    ) as connection:
        await connection.send(TextInput('hello'))
        await connection.send(AudioInput(b'pcm'))

    normalized = json.loads(json.dumps(stream.input_stream.events).replace(model.model_name, '<model>'))
    assert normalized[:3] == snapshot(
        [
            {
                'sessionStart': {
                    'inferenceConfiguration': {'maxTokens': 100, 'topP': 0.9, 'temperature': 0.4},
                    'turnDetectionConfiguration': {'endpointingSensitivity': 'HIGH'},
                }
            },
            {
                'promptStart': {
                    'promptName': normalized[1]['promptStart']['promptName'],
                    'textOutputConfiguration': {'mediaType': 'text/plain'},
                    'audioOutputConfiguration': {
                        'mediaType': 'audio/lpcm',
                        'sampleRateHertz': 24000,
                        'sampleSizeBits': 16,
                        'channelCount': 1,
                        'voiceId': 'tiffany',
                        'encoding': 'base64',
                        'audioType': 'SPEECH',
                    },
                }
            },
            {
                'contentStart': {
                    'promptName': normalized[1]['promptStart']['promptName'],
                    'contentName': normalized[2]['contentStart']['contentName'],
                    'type': 'TEXT',
                    'interactive': False,
                    'role': 'SYSTEM',
                    'textInputConfiguration': {'mediaType': 'text/plain'},
                }
            },
        ]
    )
    assert client.model_id == 'amazon.nova-2-sonic-v1:0'
    assert stream.input_stream.closed


async def collect(connection: BedrockRealtimeConnection) -> list[object]:
    values: list[object] = []
    iterator: AsyncIterator[object] = connection.__aiter__()
    async for event in iterator:
        values.append(event)
        if len(values) == 4:
            break
    return values


@pytest.mark.anyio
async def test_output_mapping_uses_final_transcript_and_usage() -> None:
    stream = FakeStream(
        [
            {'contentStart': {'role': 'ASSISTANT', 'additionalModelFields': '{"generationStage":"SPECULATIVE"}'}},
            {'textOutput': {'content': 'planned'}},
            {'contentStart': {'role': 'ASSISTANT', 'additionalModelFields': '{"generationStage":"FINAL"}'}},
            {'textOutput': {'content': 'spoken'}},
            {'audioOutput': {'content': 'cGNt'}},
            {'contentStart': {'role': 'USER', 'additionalModelFields': '{"generationStage":"FINAL"}'}},
            {'textOutput': {'content': 'hello'}},
            {
                'usageEvent': {
                    'details': {
                        # The adapter must forward the per-turn `delta`, not the cumulative `total`:
                        # the session `incr()`s each `SessionUsageEvent`, so forwarding `total` would
                        # re-add the whole session's usage every turn.
                        'delta': {
                            'input': {'speechTokens': 2, 'textTokens': 3},
                            'output': {'speechTokens': 5, 'textTokens': 7},
                        },
                        'total': {
                            'input': {'speechTokens': 200, 'textTokens': 300},
                            'output': {'speechTokens': 500, 'textTokens': 700},
                        },
                    }
                }
            },
        ]
    )
    connection = BedrockRealtimeConnection(stream, 'prompt', 'audio')
    assert await collect(connection) == snapshot(
        [
            Transcript(text='spoken', is_final=True, output_text=False),
            AudioDelta(data=b'pcm'),
            InputTranscript(text='hello', is_final=True),
            SessionUsageEvent(usage=RequestUsage(input_tokens=5, output_tokens=12)),
        ]
    )
