"""Tests for the `pydantic_ai.transcription` package.

The OpenAI adapter tests run the real OpenAI SDK against an `httpx2.MockTransport` that answers with
the response shapes documented in OpenAI's API reference, rather than against VCR cassettes: recording
one needs an OpenAI API key. The transport still sees the multipart request the SDK actually built, so
the request assertions pin what goes out on the wire.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from email import policy
from email.parser import BytesParser
from typing import Any, cast, get_args
from unittest.mock import AsyncMock, MagicMock

import httpx2
import pytest

from pydantic_ai import AudioUrl, BinaryContent, BinaryImage, Transcriber
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UserError
from pydantic_ai.models import override_allow_model_requests
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.transcription import (
    InstrumentedTranscriptionModel,
    KnownTranscriptionModelName,
    TestTranscriptionModel,
    TranscriptionInput,
    TranscriptionResult,
    TranscriptionSegment,
    WrapperTranscriptionModel,
    infer_transcription_model,
    instrument_transcription_model,
)
from pydantic_ai.usage import RequestUsage

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, IsInt, IsStr, TestEnv, try_import

pytestmark = [
    pytest.mark.usefixtures('allow_model_requests'),
]

with try_import() as logfire_imports_successful:
    from logfire.testing import CaptureLogfire

with try_import() as openai_imports_successful:
    from openai import AsyncOpenAI
    from openai.types.audio_model import AudioModel

    import pydantic_ai.transcription.openai as openai_transcription
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.transcription.openai import OpenAITranscriptionModel, OpenAITranscriptionSettings

requires_openai = pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')

AUDIO = BinaryContent(data=b'fake ogg audio', media_type='audio/ogg')


async def test_transcriber_with_test_model():
    test_model = TestTranscriptionModel(text='Hello world.')
    transcriber = Transcriber(test_model)

    result = await transcriber.transcribe(AUDIO)

    assert result == snapshot(
        TranscriptionResult(
            text='Hello world.',
            model_name='test',
            provider_name='test',
            timestamp=IsDatetime(),
            usage=RequestUsage(output_tokens=3),
            provider_response_id=IsStr(),
        )
    )
    assert test_model.last_audio is AUDIO
    assert test_model.last_settings == {}


def test_transcribe_sync():
    result = Transcriber(TestTranscriptionModel()).transcribe_sync(AUDIO)

    assert result.text == 'success (no audio transcribed)'


async def test_settings_merge_model_then_transcriber_then_call():
    """Per-call settings override the transcriber's, which override the model's own defaults."""
    test_model = TestTranscriptionModel(settings={'language': 'fr', 'temperature': 0.5})
    transcriber = Transcriber(test_model, settings={'language': 'de', 'extra_headers': {'x-team': 'voice'}})

    await transcriber.transcribe(AUDIO, settings={'temperature': 0})

    assert test_model.last_settings == {'language': 'de', 'temperature': 0, 'extra_headers': {'x-team': 'voice'}}


@pytest.mark.parametrize(
    'audio',
    [
        AudioUrl('https://example.com/voice-message.ogg'),
        BinaryContent(data=b'fake wav audio', media_type='audio/wav'),
        BinaryContent(data=b'fake webm video', media_type='video/webm'),
    ],
)
async def test_accepts_audio_url_and_audio_or_video_binary_content(audio: TranscriptionInput):
    test_model = TestTranscriptionModel()

    await Transcriber(test_model).transcribe(audio)

    assert test_model.last_audio is audio


async def test_rejects_binary_content_without_audio_media_type():
    with pytest.raises(UserError, match="Transcription input must have an audio or video media type, got 'image/png'"):
        await TestTranscriptionModel().transcribe(BinaryImage(data=b'fake png', media_type='image/png'))


async def test_rejects_input_that_is_not_audio_content():
    with pytest.raises(UserError, match='Transcription input must be an `AudioUrl` or `BinaryContent` instance'):
        await TestTranscriptionModel().transcribe(cast(TranscriptionInput, b'raw bytes'))


async def test_override():
    transcriber = Transcriber(TestTranscriptionModel(text='original'))
    override_model = TestTranscriptionModel(text='overridden')

    with transcriber.override(model=override_model):
        assert (await transcriber.transcribe(AUDIO)).text == 'overridden'
        with transcriber.override():
            assert (await transcriber.transcribe(AUDIO)).text == 'overridden'

    assert (await transcriber.transcribe(AUDIO)).text == 'original'


@requires_openai
async def test_override_with_model_name(env: TestEnv):
    """A model name passed to `override` is inferred, here to the OpenAI model, which rejects AAC by name."""
    env.set('OPENAI_API_KEY', 'test-api-key')
    transcriber = Transcriber(TestTranscriptionModel())

    with transcriber.override(model='openai:whisper-1'):
        with pytest.raises(UserError, match='OpenAI transcription supports'):
            await transcriber.transcribe(BinaryContent(data=b'fake aac', media_type='audio/aac'))


@requires_openai
def test_transcriber_resolves_model_eagerly_without_deferred_check(env: TestEnv):
    env.set('OPENAI_API_KEY', 'test-api-key')

    transcriber = Transcriber('openai:gpt-4o-transcribe', defer_model_check=False)

    assert isinstance(transcriber.model, OpenAITranscriptionModel)
    assert transcriber.model.model_name == 'gpt-4o-transcribe'


def test_transcriber_defers_model_resolution():
    transcriber = Transcriber('openai:gpt-4o-transcribe')

    assert transcriber.model == 'openai:gpt-4o-transcribe'


def test_infer_transcription_model_returns_model_instances_unchanged():
    model = TestTranscriptionModel()

    assert infer_transcription_model(model) is model


def test_infer_transcription_model_requires_provider_prefix():
    with pytest.raises(
        ValueError, match='You must provide a provider prefix when specifying a transcription model name'
    ):
        infer_transcription_model('whisper-1')


@pytest.mark.parametrize('model', ['groq:whisper-large-v3', 'anthropic:claude-sonnet-4-5', 'gateway/groq:whisper'])
def test_infer_transcription_model_rejects_provider_without_transcription_support(model: str):
    """Unsupported providers are rejected before the provider resolves, so no credentials are needed."""
    provider_factory = MagicMock()
    provider_name = model.split(':', maxsplit=1)[0]

    with pytest.raises(UserError, match=f"Provider '{provider_name}' does not support transcription"):
        infer_transcription_model(model, provider_factory=provider_factory)

    provider_factory.assert_not_called()


@requires_openai
@pytest.mark.parametrize('provider_name', ['openai', 'azure', 'litellm', 'openrouter'])
def test_infer_transcription_model_openai_compatible_providers(provider_name: str):
    provider = OpenAIProvider(api_key='test-api-key')
    provider_factory = MagicMock(return_value=provider)

    model = infer_transcription_model(f'{provider_name}:whisper-1', provider_factory=provider_factory)

    assert isinstance(model, OpenAITranscriptionModel)
    assert model.model_name == 'whisper-1'
    provider_factory.assert_called_once_with(provider_name)


@requires_openai
def test_infer_transcription_model_gateway(env: TestEnv):
    env.set('PYDANTIC_AI_GATEWAY_API_KEY', 'test-api-key')
    env.set('PYDANTIC_AI_GATEWAY_BASE_URL', 'https://gateway.pydantic.dev/proxy')

    model = infer_transcription_model('gateway/openai:gpt-4o-transcribe')

    assert isinstance(model, OpenAITranscriptionModel)
    assert model.model_name == 'gpt-4o-transcribe'
    assert model.system == 'openai'
    assert model.base_url.startswith('https://gateway.pydantic.dev/proxy/')


@requires_openai
def test_known_transcription_model_names_match_openai_sdk():
    assert sorted(get_args(KnownTranscriptionModelName.__value__)) == sorted(
        f'openai:{name}' for name in get_args(AudioModel)
    )


@requires_openai
async def test_wrapper_transcription_model_delegates(env: TestEnv):
    env.set('OPENAI_API_KEY', 'test-api-key')
    from_name = WrapperTranscriptionModel('openai:whisper-1')
    assert isinstance(from_name.wrapped, OpenAITranscriptionModel)
    assert from_name.base_url == 'https://api.openai.com/v1/'

    wrapped = TestTranscriptionModel('my-model', provider_name='my-provider', settings={'language': 'en'})
    wrapper = WrapperTranscriptionModel(wrapped)

    result = await wrapper.transcribe(AUDIO)

    assert result.text == 'success (no audio transcribed)'
    assert wrapper.model_name == 'my-model'
    assert wrapper.system == 'my-provider'
    assert wrapper.settings == {'language': 'en'}
    assert wrapper.base_url is None
    # Attributes the wrapper doesn't define are read from the wrapped model.
    assert wrapper.last_settings == {'language': 'en'}
    with pytest.raises(AttributeError):
        wrapper.__getattr__('wrapped')


def test_instrument_transcription_model_wraps_once():
    model = TestTranscriptionModel()

    assert instrument_transcription_model(model, False) is model

    instrumented = instrument_transcription_model(model, True)
    assert isinstance(instrumented, InstrumentedTranscriptionModel)
    assert instrument_transcription_model(instrumented, True) is instrumented


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_instrumentation(capfire: CaptureLogfire):
    transcriber = Transcriber(TestTranscriptionModel(text='Book a table for two.'), instrument=True)

    await transcriber.transcribe(
        AudioUrl('https://example.com/private-voice-message.ogg'),
        settings={
            'language': 'en',
            'extra_headers': {'authorization': 'Bearer test'},
            'extra_body': {'application_data': 'test'},
        },
    )

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    span = next(span for span in spans if 'transcription' in span['name'])
    assert span == snapshot(
        {
            'name': 'transcription test',
            'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
            'parent': None,
            'start_time': IsInt(),
            'end_time': IsInt(),
            'attributes': {
                'gen_ai.operation.name': 'transcription',
                'gen_ai.output.type': 'text',
                'gen_ai.provider.name': 'test',
                'gen_ai.request.model': 'test',
                'transcription_settings': {'language': 'en'},
                'logfire.json_schema': {
                    'type': 'object',
                    'properties': {
                        'transcription_settings': {'type': 'object'},
                        'segment_count': {'type': 'integer'},
                        'transcript': {'type': 'string'},
                    },
                },
                'logfire.span_type': 'span',
                'logfire.msg': 'transcription test',
                'gen_ai.usage.output_tokens': 6,
                'gen_ai.response.model': 'test',
                'segment_count': 0,
                'gen_ai.response.id': IsStr(),
                'transcript': 'Book a table for two.',
            },
        }
    )
    assert 'private-voice-message' not in str(span)
    assert 'Bearer test' not in str(span)

    metrics = capfire.get_collected_metrics()
    assert [metric['name'] for metric in metrics] == ['gen_ai.client.token.usage']
    data_points = metrics[0]['data']['data_points']
    assert len(data_points) == 1
    assert data_points[0]['attributes'] == snapshot(
        {
            'gen_ai.provider.name': 'test',
            'gen_ai.operation.name': 'transcription',
            'gen_ai.request.model': 'test',
            'gen_ai.response.model': 'test',
            'gen_ai.token.type': 'output',
        }
    )
    assert data_points[0]['sum'] == 6


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_instrumentation_respects_content_and_request_parameter_flags(capfire: CaptureLogfire):
    transcriber = Transcriber(
        TestTranscriptionModel(text='Book a table for two.'),
        instrument=InstrumentationSettings(include_content=False, include_model_request_parameters=False),
    )

    await transcriber.transcribe(AUDIO, settings={'language': 'en'})

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    span = next(span for span in spans if 'transcription' in span['name'])
    assert 'transcript' not in span['attributes']
    assert 'transcription_settings' not in span['attributes']
    assert 'Book a table' not in str(span)


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_instrumentation_omits_empty_recorded_settings(capfire: CaptureLogfire):
    transcriber = Transcriber(TestTranscriptionModel(), instrument=True)

    await transcriber.transcribe(AUDIO, settings={'extra_headers': {'authorization': 'Bearer test'}})

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    span = next(span for span in spans if 'transcription' in span['name'])
    assert 'transcription_settings' not in span['attributes']


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_instrumentation_does_not_record_metrics_when_transcription_fails(capfire: CaptureLogfire):
    wrapped = TestTranscriptionModel()
    wrapped.transcribe = AsyncMock(side_effect=RuntimeError('transcription failed'))
    model = InstrumentedTranscriptionModel(wrapped)

    with pytest.raises(RuntimeError, match='transcription failed'):
        await model.transcribe(AUDIO)

    assert capfire.metrics_reader.get_metrics_data() is None
    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    span = next(span for span in spans if 'transcription' in span['name'])
    assert span['attributes']['logfire.level_num'] == 17


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_instrument_all(capfire: CaptureLogfire):
    transcriber = Transcriber(TestTranscriptionModel())
    Transcriber.instrument_all()
    try:
        await transcriber.transcribe(AUDIO)
    finally:
        Transcriber.instrument_all(False)

    await transcriber.transcribe(AUDIO)

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    assert len([span for span in spans if 'transcription' in span['name']]) == 1


def _openai_model(
    model_name: str,
    respond: Callable[[httpx2.Request], httpx2.Response],
    requests: list[httpx2.Request],
) -> OpenAITranscriptionModel:
    """An `OpenAITranscriptionModel` whose HTTP requests are recorded and answered by `respond`."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return respond(request)

    client = AsyncOpenAI(
        api_key='test-api-key',
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return OpenAITranscriptionModel(model_name, provider=OpenAIProvider(openai_client=client))


def _json_response(body: dict[str, Any]) -> Callable[[httpx2.Request], httpx2.Response]:
    return lambda request: httpx2.Response(200, json=body)


def _form(request: httpx2.Request) -> dict[str, str]:
    """The multipart form fields of a request, with the uploaded file summarized as its name and content type."""
    message = BytesParser(policy=policy.HTTP).parsebytes(
        f'Content-Type: {request.headers["content-type"]}\r\n\r\n'.encode() + request.content
    )
    fields: dict[str, str] = {}
    for part in message.iter_parts():
        name = cast(str, part.get_param('name', header='content-disposition'))
        payload = cast(bytes, part.get_payload(decode=True))
        if filename := part.get_filename():
            fields[name] = f'{filename} ({part.get_content_type()}, {len(payload)} bytes)'
        else:
            fields[name] = payload.decode()
    return fields


@requires_openai
async def test_openai_token_billed_transcription():
    requests: list[httpx2.Request] = []
    model = _openai_model(
        'gpt-4o-transcribe',
        _json_response(
            {
                'text': 'Hallo, ich hätte gern einen Tisch für zwei.',
                'usage': {
                    'type': 'tokens',
                    'input_tokens': 120,
                    'input_token_details': {'text_tokens': 20, 'audio_tokens': 100},
                    'output_tokens': 15,
                    'total_tokens': 135,
                },
            }
        ),
        requests,
    )

    result = await Transcriber(model).transcribe(
        AUDIO,
        settings=OpenAITranscriptionSettings(
            language='de', temperature=0, openai_prompt='Tisch', extra_headers={'x-request-tag': 'voice'}
        ),
    )

    assert [request.url.path for request in requests] == ['/v1/audio/transcriptions']
    assert requests[0].headers['x-request-tag'] == 'voice'
    assert _form(requests[0]) == snapshot(
        {
            'model': 'gpt-4o-transcribe',
            'language': 'de',
            'prompt': 'Tisch',
            'response_format': 'json',
            'temperature': '0',
            'file': 'audio.ogg (audio/ogg, 14 bytes)',
        }
    )
    assert result == snapshot(
        TranscriptionResult(
            text='Hallo, ich hätte gern einen Tisch für zwei.',
            model_name='gpt-4o-transcribe',
            provider_name='openai',
            timestamp=IsDatetime(),
            usage=RequestUsage(input_tokens=120, output_tokens=15, input_audio_tokens=100),
            provider_url='https://api.openai.com/v1/',
        )
    )
    assert result.cost().total_price == snapshot(Decimal('0.00080'))


@requires_openai
async def test_openai_verbose_json_segments():
    requests: list[httpx2.Request] = []
    model = _openai_model(
        'whisper-1',
        _json_response(
            {
                'task': 'transcribe',
                'language': 'english',
                'duration': 4.2,
                'text': 'Book a table. For two, please.',
                'segments': [
                    {
                        'id': 0,
                        'seek': 0,
                        'start': 0.0,
                        'end': 1.6,
                        'text': ' Book a table.',
                        'tokens': [50364, 9476],
                        'temperature': 0.0,
                        'avg_logprob': -0.2,
                        'compression_ratio': 0.9,
                        'no_speech_prob': 0.01,
                    },
                    {
                        'id': 1,
                        'seek': 0,
                        'start': 1.6,
                        'end': 4.2,
                        'text': ' For two, please.',
                        'tokens': [50444, 1171],
                        'temperature': 0.0,
                        'avg_logprob': -0.3,
                        'compression_ratio': 0.9,
                        'no_speech_prob': 0.02,
                    },
                ],
                'usage': {'type': 'duration', 'seconds': 5},
            }
        ),
        requests,
    )

    result = await model.transcribe(
        BinaryContent(data=b'fake wav audio', media_type='audio/x-wav'),
        settings=OpenAITranscriptionSettings(openai_response_format='verbose_json'),
    )

    assert _form(requests[0]) == snapshot(
        {'model': 'whisper-1', 'response_format': 'verbose_json', 'file': 'audio.wav (audio/x-wav, 14 bytes)'}
    )
    assert result.segments == snapshot(
        [
            TranscriptionSegment(text=' Book a table.', start=0.0, end=1.6),
            TranscriptionSegment(text=' For two, please.', start=1.6, end=4.2),
        ]
    )
    assert result.provider_details == snapshot({'language': 'english', 'duration': 4.2})
    assert result.usage == snapshot(RequestUsage(audio_seconds=5.0))
    assert result.cost().total_price == snapshot(Decimal('0.0005'))


@requires_openai
async def test_openai_duration_billed_json_with_detected_languages():
    model = _openai_model(
        'gpt-transcribe',
        _json_response(
            {
                'text': 'Bonjour.',
                'languages': [{'code': 'fr'}],
                'usage': {'type': 'duration', 'seconds': 2.5},
            }
        ),
        [],
    )

    result = await model.transcribe(AUDIO)

    assert result.provider_details == {'languages': ['fr']}
    assert result.usage == RequestUsage(audio_seconds=2.5)
    assert result.segments == []


@requires_openai
@pytest.mark.parametrize(
    'usage,expected',
    [
        pytest.param(None, RequestUsage(), id='no_usage'),
        pytest.param(
            {'type': 'tokens', 'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12},
            RequestUsage(input_tokens=10, output_tokens=2),
            id='tokens_without_details',
        ),
    ],
)
async def test_openai_usage_variants(usage: dict[str, Any] | None, expected: RequestUsage):
    """OpenAI-compatible providers can omit `usage`, and token usage can come without the per-modality breakdown."""
    model = _openai_model('gpt-4o-mini-transcribe', _json_response({'text': 'Hi.', 'usage': usage}), [])

    result = await model.transcribe(AUDIO)

    assert result.usage == expected
    assert result.provider_details is None


@requires_openai
async def test_openai_downloads_audio_url(monkeypatch: pytest.MonkeyPatch):
    download_item = AsyncMock(return_value={'data': b'fake mp3 audio', 'data_type': 'audio/mpeg'})
    monkeypatch.setattr(openai_transcription, 'download_item', download_item)
    requests: list[httpx2.Request] = []
    model = _openai_model('whisper-1', _json_response({'text': 'Hi.'}), requests)
    audio = AudioUrl('https://example.com/voice-message.mp3')

    await model.transcribe(audio)

    download_item.assert_awaited_once_with(audio, data_format='bytes')
    assert _form(requests[0])['file'] == 'audio.mp3 (audio/mpeg, 14 bytes)'


@requires_openai
async def test_openai_rejects_unsupported_media_type_before_requesting():
    requests: list[httpx2.Request] = []
    model = _openai_model('whisper-1', _json_response({'text': 'unreachable'}), requests)

    with pytest.raises(UserError) as exc_info:
        await model.transcribe(BinaryContent(data=b'fake aac audio', media_type='audio/aac'))

    assert str(exc_info.value) == snapshot(
        "OpenAI transcription supports FLAC, MP3, MP4, MPEG, M4A, Ogg, WAV, and WebM audio, got media type 'audio/aac'"
    )
    assert requests == []


@requires_openai
async def test_openai_http_error():
    model = _openai_model(
        'gpt-4o-transcribe',
        lambda request: httpx2.Response(
            400,
            json={
                'error': {'message': 'Audio file might be corrupted or unsupported', 'type': 'invalid_request_error'}
            },
        ),
        [],
    )

    with pytest.raises(ModelHTTPError) as exc_info:
        await model.transcribe(AUDIO)

    assert exc_info.value.status_code == 400
    assert exc_info.value.model_name == 'gpt-4o-transcribe'
    assert exc_info.value.body == snapshot(
        {'message': 'Audio file might be corrupted or unsupported', 'type': 'invalid_request_error'}
    )


@requires_openai
async def test_openai_connection_error():
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError('connection refused', request=request)

    model = _openai_model('gpt-4o-transcribe', refuse, [])

    with pytest.raises(ModelAPIError, match='Connection error'):
        await model.transcribe(AUDIO)


@requires_openai
async def test_openai_non_json_response():
    model = _openai_model(
        'gpt-4o-transcribe',
        lambda request: httpx2.Response(200, text='Bad Gateway', headers={'content-type': 'text/plain'}),
        [],
    )

    with pytest.raises(ModelAPIError, match="Expected a JSON response, got: 'Bad Gateway'"):
        await model.transcribe(AUDIO)


@requires_openai
async def test_openai_respects_allow_model_requests():
    model = _openai_model('gpt-4o-transcribe', _json_response({'text': 'unreachable'}), [])

    with override_allow_model_requests(False):
        with pytest.raises(RuntimeError, match='Model requests are not allowed'):
            await model.transcribe(AUDIO)


@requires_openai
def test_openai_model_infers_string_provider(env: TestEnv):
    env.set('OPENAI_API_KEY', 'test-api-key')

    model = OpenAITranscriptionModel('whisper-1')

    assert model.model_name == 'whisper-1'
    assert model.system == 'openai'
    assert model.base_url == 'https://api.openai.com/v1/'


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
@requires_openai
async def test_instrumentation_records_duration_and_cost(capfire: CaptureLogfire):
    """A duration-billed model's span carries `audio_seconds` and the cost genai-prices derives from it."""
    model = _openai_model(
        'whisper-1', _json_response({'text': 'Hi.', 'usage': {'type': 'duration', 'seconds': 36}}), []
    )

    await Transcriber(model, instrument=True).transcribe(AUDIO)

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    span = next(span for span in spans if 'transcription' in span['name'])
    assert span['attributes'] == snapshot(
        {
            'gen_ai.operation.name': 'transcription',
            'gen_ai.output.type': 'text',
            'gen_ai.provider.name': 'openai',
            'gen_ai.request.model': 'whisper-1',
            'server.address': 'api.openai.com',
            'logfire.json_schema': {
                'type': 'object',
                'properties': {'segment_count': {'type': 'integer'}, 'transcript': {'type': 'string'}},
            },
            'logfire.span_type': 'span',
            'logfire.msg': 'transcription whisper-1',
            'gen_ai.response.model': 'whisper-1',
            'segment_count': 0,
            'audio_seconds': 36.0,
            'operation.cost': 0.0036,
            'transcript': 'Hi.',
        }
    )
    metrics = capfire.get_collected_metrics()
    assert [metric['name'] for metric in metrics] == snapshot(['operation.cost'])


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
@requires_openai
async def test_instrumentation_records_token_metrics(capfire: CaptureLogfire):
    model = _openai_model(
        'gpt-4o-transcribe',
        _json_response(
            {
                'text': 'Hi.',
                'usage': {'type': 'tokens', 'input_tokens': 50, 'output_tokens': 2, 'total_tokens': 52},
            }
        ),
        [],
    )

    await Transcriber(model, instrument=True).transcribe(AUDIO)

    metrics = capfire.get_collected_metrics()
    token_metric = next(metric for metric in metrics if metric['name'] == 'gen_ai.client.token.usage')
    assert {
        point['attributes']['gen_ai.token.type']: point['sum'] for point in token_metric['data']['data_points']
    } == {'input': 50, 'output': 2}


def test_instrumentation_span_without_a_result_records_no_metrics():
    """A span whose request never produced a result has no metrics to record when it closes."""
    model = InstrumentedTranscriptionModel(TestTranscriptionModel())

    with model._instrument({}):  # pyright: ignore[reportPrivateUsage]
        pass
