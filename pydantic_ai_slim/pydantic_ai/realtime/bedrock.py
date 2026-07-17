"""Amazon Nova Sonic realtime model support over Bedrock bidirectional HTTP/2 streaming."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import InitVar, dataclass, field
from typing import Literal, Protocol, cast

from .._utils import is_str_dict
from ..messages import ModelMessage, ModelRequest, ModelResponse, SpeechPart, TextPart, UserPromptPart
from ..native_tools import AbstractNativeTool
from ..tools import ToolDefinition
from ..usage import RequestUsage
from ._base import (
    AudioDelta,
    AudioInput,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    CreateResponse,
    ImageInput,
    InputTranscript,
    RealtimeConnection,
    RealtimeEvent,
    RealtimeInput,
    RealtimeModel,
    RealtimeModelProfile,
    RealtimeModelSettings,
    ReconnectPolicy,
    SessionErrorEvent,
    SessionUsageEvent,
    TextInput,
    ToolCall,
    ToolResult,
    Transcript,
    TruncateOutput,
    TurnCompleteEvent,
    user_prompt_text,
)

try:
    from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient
    from aws_sdk_bedrock_runtime.config import Config
    from aws_sdk_bedrock_runtime.models import (
        BidirectionalInputPayloadPart,
        InvokeModelWithBidirectionalStreamInputChunk,
        InvokeModelWithBidirectionalStreamOperationInput,
        InvokeModelWithBidirectionalStreamOutputChunk,
        InvokeModelWithBidirectionalStreamOutputInternalServerException,
        InvokeModelWithBidirectionalStreamOutputModelStreamErrorException,
        InvokeModelWithBidirectionalStreamOutputModelTimeoutException,
        InvokeModelWithBidirectionalStreamOutputServiceUnavailableException,
        InvokeModelWithBidirectionalStreamOutputThrottlingException,
        InvokeModelWithBidirectionalStreamOutputUnknown,
        InvokeModelWithBidirectionalStreamOutputValidationException,
    )
    from smithy_aws_core.identity import EnvironmentCredentialsResolver
except ImportError as _import_error:  # pragma: no cover - exercised by optional-dependency import tests
    raise ImportError(
        'Amazon Nova Sonic realtime support requires `aws-sdk-bedrock-runtime`. '
        'Install `pydantic-ai-slim[realtime-bedrock]`.'
    ) from _import_error


class BedrockRealtimeModelSettings(RealtimeModelSettings, total=False):
    """Settings for an Amazon Nova Sonic realtime session."""

    temperature: float
    """Sampling temperature."""

    top_p: float
    """Nucleus-sampling probability."""

    bedrock_endpointing_sensitivity: Literal['LOW', 'MEDIUM', 'HIGH']
    """Nova 2 Sonic endpointing sensitivity. Nova Sonic v1 ignores this setting."""


class _InputStream(Protocol):
    async def send(self, value: InvokeModelWithBidirectionalStreamInputChunk) -> None: ...

    async def close(self) -> None: ...


class _Receiver(Protocol):
    async def receive(self) -> object: ...


class _DuplexStream(Protocol):
    @property
    def input_stream(self) -> _InputStream: ...

    async def await_output(self) -> tuple[object, _Receiver]: ...


class _BedrockClient(Protocol):
    async def invoke_model_with_bidirectional_stream(
        self, input: InvokeModelWithBidirectionalStreamOperationInput
    ) -> _DuplexStream: ...


def _dict(value: dict[str, object], key: str) -> dict[str, object] | None:
    item = value.get(key)
    if not is_str_dict(item):
        return None
    return cast('dict[str, object]', item)


def _str(value: dict[str, object], key: str) -> str | None:
    item = value.get(key)
    return item if isinstance(item, str) else None


@dataclass
class BedrockRealtimeConnection(RealtimeConnection):
    """A Nova Sonic bidirectional stream."""

    _stream: _DuplexStream
    _prompt_name: str
    _audio_content_name: str
    _audio_started: bool = False
    _role: str | None = None
    _generation_stage: str | None = None
    _interrupted: bool = False

    async def send_event(self, event: dict[str, object]) -> None:
        payload = json.dumps({'event': event}, separators=(',', ':')).encode()
        await self._stream.input_stream.send(
            InvokeModelWithBidirectionalStreamInputChunk(value=BidirectionalInputPayloadPart(bytes_=payload))
        )

    async def _start_audio(self) -> None:
        if self._audio_started:
            return
        await self.send_event(
            {
                'contentStart': {
                    'promptName': self._prompt_name,
                    'contentName': self._audio_content_name,
                    'type': 'AUDIO',
                    'interactive': True,
                    'audioInputConfiguration': {
                        'mediaType': 'audio/lpcm',
                        'sampleRateHertz': 16000,
                        'sampleSizeBits': 16,
                        'channelCount': 1,
                        'audioType': 'SPEECH',
                        'encoding': 'base64',
                    },
                }
            }
        )
        self._audio_started = True

    async def send_text(self, text: str, role: Literal['USER', 'SYSTEM', 'ASSISTANT']) -> None:
        content_name = str(uuid.uuid4())
        await self.send_event(
            {
                'contentStart': {
                    'promptName': self._prompt_name,
                    'contentName': content_name,
                    'type': 'TEXT',
                    'interactive': role == 'USER',
                    'role': role,
                    'textInputConfiguration': {'mediaType': 'text/plain'},
                }
            }
        )
        await self.send_event(
            {'textInput': {'promptName': self._prompt_name, 'contentName': content_name, 'content': text}}
        )
        await self.send_event({'contentEnd': {'promptName': self._prompt_name, 'contentName': content_name}})

    async def send(self, content: RealtimeInput) -> None:
        if isinstance(content, AudioInput):
            await self._start_audio()
            await self.send_event(
                {
                    'audioInput': {
                        'promptName': self._prompt_name,
                        'contentName': self._audio_content_name,
                        'content': base64.b64encode(content.data).decode(),
                    }
                }
            )
        elif isinstance(content, TextInput):
            await self.send_text(content.text, 'USER')
        elif isinstance(content, ToolResult):
            content_name = str(uuid.uuid4())
            await self.send_event(
                {
                    'contentStart': {
                        'promptName': self._prompt_name,
                        'contentName': content_name,
                        'type': 'TOOL',
                        'interactive': False,
                        'role': 'TOOL',
                        'toolResultInputConfiguration': {
                            'toolUseId': content.tool_call_id,
                            'type': 'TEXT',
                            'textInputConfiguration': {'mediaType': 'text/plain'},
                        },
                    }
                }
            )
            await self.send_event(
                {
                    'toolResult': {
                        'promptName': self._prompt_name,
                        'contentName': content_name,
                        'content': content.output,
                    }
                }
            )
            await self.send_event({'contentEnd': {'promptName': self._prompt_name, 'contentName': content_name}})
        elif isinstance(content, CommitAudio):
            if self._audio_started:
                await self.send_event(
                    {
                        'contentEnd': {
                            'promptName': self._prompt_name,
                            'contentName': self._audio_content_name,
                        }
                    }
                )
                self._audio_started = False
                self._audio_content_name = str(uuid.uuid4())
        elif isinstance(content, (ImageInput, ClearAudio, CreateResponse, CancelResponse, TruncateOutput)):
            raise NotImplementedError(f'Amazon Nova Sonic does not support {type(content).__name__} input')

    def _map_json_event(self, data: dict[str, object]) -> list[RealtimeEvent]:
        events: list[RealtimeEvent] = []
        if content_start := _dict(data, 'contentStart'):
            self._role = _str(content_start, 'role')
            fields = _str(content_start, 'additionalModelFields')
            if fields:
                parsed: object = json.loads(fields)
                if is_str_dict(parsed):
                    parsed_dict = cast('dict[str, object]', parsed)
                    stage = parsed_dict.get('generationStage')
                    self._generation_stage = stage if isinstance(stage, str) else None
        elif text_output := _dict(data, 'textOutput'):
            text = _str(text_output, 'content')
            if text == '{ "interrupted" : true }':
                self._interrupted = True
            elif text and self._role == 'USER':
                events.append(InputTranscript(text=text, is_final=True))
            elif text and self._role == 'ASSISTANT' and self._generation_stage == 'FINAL':
                events.append(Transcript(text=text, is_final=True))
        elif audio_output := _dict(data, 'audioOutput'):
            if encoded := _str(audio_output, 'content'):
                events.append(AudioDelta(data=base64.b64decode(encoded)))
        elif tool_use := _dict(data, 'toolUse'):
            call_id = _str(tool_use, 'toolUseId') or ''
            name = _str(tool_use, 'toolName') or ''
            args = _str(tool_use, 'content') or ''
            events.append(ToolCall(tool_call_id=call_id, tool_name=name, args=args))
        elif usage_event := _dict(data, 'usageEvent'):
            # Nova emits a `usageEvent` per turn with both `delta` (this turn) and `total` (cumulative).
            # The session `incr()`s each `SessionUsageEvent` into its running `RunUsage`, so forward the
            # per-turn delta — forwarding `total` would re-add the whole session's usage every turn.
            details = _dict(usage_event, 'details')
            delta = _dict(details, 'delta') if details else None
            input_counts = _dict(delta, 'input') if delta else None
            output_counts = _dict(delta, 'output') if delta else None
            input_tokens = sum(v for v in (input_counts or {}).values() if isinstance(v, int))
            output_tokens = sum(v for v in (output_counts or {}).values() if isinstance(v, int))
            events.append(SessionUsageEvent(RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens)))
        elif completion_end := _dict(data, 'completionEnd'):
            interrupted = self._interrupted or _str(completion_end, 'stopReason') == 'INTERRUPTED'
            events.append(TurnCompleteEvent(interrupted=interrupted))
            self._interrupted = False
        return events

    async def __aiter__(self) -> AsyncIterator[RealtimeEvent]:
        while True:
            try:
                _, receiver = await self._stream.await_output()
                result = await receiver.receive()
            except (StopAsyncIteration, OSError, TimeoutError) as exc:
                yield SessionErrorEvent(message=f'Bedrock realtime connection closed: {exc}', recoverable=False)
                return
            if isinstance(result, InvokeModelWithBidirectionalStreamOutputChunk):
                value = result.value
                if value.bytes_:
                    try:
                        payload: object = json.loads(value.bytes_)
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        yield SessionErrorEvent(message=f'Failed to parse Bedrock realtime event: {exc}')
                        continue
                    if is_str_dict(payload):
                        payload_dict = cast('dict[str, object]', payload)
                        if event := _dict(payload_dict, 'event'):
                            for mapped in self._map_json_event(event):
                                yield mapped
            elif isinstance(result, InvokeModelWithBidirectionalStreamOutputUnknown):
                yield SessionErrorEvent(message='Bedrock returned an unknown stream event', recoverable=True)
            elif isinstance(
                result,
                (
                    InvokeModelWithBidirectionalStreamOutputInternalServerException,
                    InvokeModelWithBidirectionalStreamOutputModelStreamErrorException,
                    InvokeModelWithBidirectionalStreamOutputModelTimeoutException,
                    InvokeModelWithBidirectionalStreamOutputServiceUnavailableException,
                    InvokeModelWithBidirectionalStreamOutputThrottlingException,
                    InvokeModelWithBidirectionalStreamOutputValidationException,
                ),
            ):
                yield SessionErrorEvent(message=str(result.value), type=type(result.value).__name__)


@dataclass
class BedrockRealtimeModel(RealtimeModel):
    """Amazon Nova Sonic realtime model."""

    model: str = 'amazon.nova-2-sonic-v1:0'
    provider: InitVar[_BedrockClient | None] = None
    settings: RealtimeModelSettings | None = field(default=None, kw_only=True)
    handshake_timeout: float = 30.0
    reconnect: ReconnectPolicy | None = None
    _client: _BedrockClient = field(init=False, repr=False)

    def __post_init__(self, provider: _BedrockClient | None) -> None:
        if provider is not None:
            self._client = provider
            return
        region = os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION') or 'us-east-1'
        config = Config(
            endpoint_uri=f'https://bedrock-runtime.{region}.amazonaws.com',
            region=region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
        self._client = cast('_BedrockClient', BedrockRuntimeClient(config=config))

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def system(self) -> str:
        return 'bedrock'

    @property
    def profile(self) -> RealtimeModelProfile:
        return RealtimeModelProfile(
            supports_image_input=False,
            supports_manual_turn_control=True,
            supports_interruption=False,
            supports_output_truncation=False,
            supports_session_seeding=True,
            supported_native_tools=frozenset(),
        )

    @asynccontextmanager
    async def connect(
        self,
        *,
        instructions: str,
        tools: list[ToolDefinition] | None = None,
        native_tools: list[AbstractNativeTool] | None = None,
        model_settings: RealtimeModelSettings | None = None,
        messages: Sequence[ModelMessage] | None = None,
    ) -> AsyncGenerator[BedrockRealtimeConnection]:
        del native_tools
        settings = cast('BedrockRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        stream = await asyncio.wait_for(
            self._client.invoke_model_with_bidirectional_stream(
                InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model)
            ),
            timeout=self.handshake_timeout,
        )
        prompt_name = str(uuid.uuid4())
        connection = BedrockRealtimeConnection(stream, prompt_name, str(uuid.uuid4()))
        inference: dict[str, object] = {
            'maxTokens': settings.get('max_tokens', 1024),
            'topP': settings.get('top_p', 0.9),
            'temperature': settings.get('temperature', 0.7),
        }
        session_start: dict[str, object] = {'inferenceConfiguration': inference}
        if self.model.startswith('amazon.nova-2-') and (sensitivity := settings.get('bedrock_endpointing_sensitivity')):
            session_start['turnDetectionConfiguration'] = {'endpointingSensitivity': sensitivity}
        await connection.send_event({'sessionStart': session_start})
        prompt_start: dict[str, object] = {
            'promptName': prompt_name,
            'textOutputConfiguration': {'mediaType': 'text/plain'},
            'audioOutputConfiguration': {
                'mediaType': 'audio/lpcm',
                'sampleRateHertz': 24000,
                'sampleSizeBits': 16,
                'channelCount': 1,
                'voiceId': settings.get('voice', 'matthew'),
                'encoding': 'base64',
                'audioType': 'SPEECH',
            },
        }
        if tools:
            prompt_start['toolUseOutputConfiguration'] = {'mediaType': 'application/json'}
            prompt_start['toolConfiguration'] = {
                'tools': [
                    {
                        'toolSpec': {
                            'name': tool.name,
                            'description': tool.description or '',
                            'inputSchema': {'json': json.dumps(tool.parameters_json_schema)},
                        }
                    }
                    for tool in tools
                ]
            }
        await connection.send_event({'promptStart': prompt_start})
        if instructions:
            await connection.send_text(instructions, 'SYSTEM')
        for message in messages or ():
            if isinstance(message, ModelRequest):
                texts = [user_prompt_text(part) for part in message.parts if isinstance(part, UserPromptPart)]
                texts += [part.transcript for part in message.parts if isinstance(part, SpeechPart) and part.transcript]
                role: Literal['USER', 'ASSISTANT'] = 'USER'
            elif isinstance(message, ModelResponse):
                texts = [part.content for part in message.parts if isinstance(part, TextPart)]
                texts += [part.transcript for part in message.parts if isinstance(part, SpeechPart) and part.transcript]
                role = 'ASSISTANT'
            else:
                continue
            if texts:
                await connection.send_text('\n'.join(texts), role)
        try:
            yield connection
        finally:
            await connection.send(CommitAudio())
            await connection.send_event({'promptEnd': {'promptName': prompt_name}})
            await connection.send_event({'sessionEnd': {}})
            await stream.input_stream.close()
