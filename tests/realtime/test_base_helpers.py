"""Direct tests for the provider-support helpers in `pydantic_ai.realtime._base`.

These are the pieces a concrete [`RealtimeModel`][pydantic_ai.realtime.RealtimeModel] implementation
builds on — profile resolution and merging, settings merging, reconnect backoff, and history-seeding
normalization — rather than anything the session drives itself. They have no provider dependency, so
they are exercised here against a minimal in-test model. (`inject_trace_context` lives in
`test_instrumentation.py`, which already gates on the OpenTelemetry SDK.)
"""

from __future__ import annotations as _annotations

import io
import wave
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    CachePoint,
    ImageUrl,
    ModelMessage,
    SpeechPart,
    TextContent,
    UserPromptPart,
)
from pydantic_ai.models import DownloadedItem, ModelRequestParameters
from pydantic_ai.native_tools import AbstractNativeTool, WebFetchTool, WebSearchTool
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers import Provider
from pydantic_ai.realtime import (
    RealtimeModel,
    RealtimeModelProfile,
    RealtimeModelSettings,
    ReconnectPolicy,
)
from pydantic_ai.realtime._base import (
    reconnect_with_backoff,
    seed_pcm_audio,
    seed_speech_content,
    seed_user_content,
)
from pydantic_ai.realtime.codec import (
    DEFAULT_REALTIME_PROFILE,
    RealtimeConnection,
    merge_realtime_profile,
)

pytestmark = pytest.mark.anyio


class _BareModel(RealtimeModel):
    """A model with no provider, exercising the `RealtimeModel` defaults."""

    @property
    def model_name(self) -> str:
        return 'bare-realtime'

    @property
    def system(self) -> str:
        return 'bare'

    def connect(
        self,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AbstractAsyncContextManager[RealtimeConnection]:
        raise NotImplementedError('this model exists only to exercise the `RealtimeModel` defaults')


class _SearchOnlyModel(_BareModel):
    """A model whose implementation covers only part of what its provider advertises."""

    def __init__(self, provider: Provider[Any]) -> None:
        self._provider = provider

    @classmethod
    def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
        return frozenset({WebSearchTool})


class _VoiceProvider(Provider[None]):
    """A provider that advertises realtime facts, as a real provider does."""

    @property
    def name(self) -> str:
        return 'voice'

    @property
    def base_url(self) -> str:
        return 'https://voice.example.com'

    @property
    def client(self) -> None:
        return None  # pragma: no cover

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        return None  # pragma: no cover

    @staticmethod
    def realtime_model_profile(model_name: str) -> RealtimeModelProfile:
        return {
            'supports_interruption': True,
            'supported_native_tools': frozenset({WebSearchTool, WebFetchTool}),
        }


def test_merge_realtime_profile_layers_later_overrides_last() -> None:
    assert merge_realtime_profile(None) == {}
    assert merge_realtime_profile(
        {'supports_interruption': True, 'audio_input_sample_rate': 16000},
        None,
        {'audio_input_sample_rate': 24000},
    ) == {'supports_interruption': True, 'audio_input_sample_rate': 24000}


def test_bare_model_falls_back_to_the_default_profile() -> None:
    model = _BareModel()

    assert model.model_name == 'bare-realtime'
    assert model.system == 'bare'
    assert model.base_url is None
    assert model.settings is None
    assert model.profile == DEFAULT_REALTIME_PROFILE
    assert model.supported_native_tools() == frozenset()


def test_model_profile_intersects_provider_tools_with_the_implementation() -> None:
    """A provider may advertise more native tools than this model class actually implements."""
    provider = _VoiceProvider()
    model = _SearchOnlyModel(provider)

    assert provider.name == 'voice'
    assert model.base_url == 'https://voice.example.com'
    assert model.profile.get('supports_interruption') is True
    assert model.profile.get('supported_native_tools') == frozenset({WebSearchTool})


def test_merge_model_settings_layers_connection_overrides_on_model_defaults() -> None:
    without_defaults = _BareModel()
    assert without_defaults._merge_model_settings(None) is None  # pyright: ignore[reportPrivateUsage]
    assert without_defaults._merge_model_settings({'max_tokens': 10}) == {'max_tokens': 10}  # pyright: ignore[reportPrivateUsage]

    with_defaults = _BareModel()
    with_defaults.settings = RealtimeModelSettings(max_tokens=10, output_modality='text')
    assert with_defaults._merge_model_settings(None) == {'max_tokens': 10, 'output_modality': 'text'}  # pyright: ignore[reportPrivateUsage]
    assert with_defaults._merge_model_settings({'max_tokens': 20}) == {  # pyright: ignore[reportPrivateUsage]
        'max_tokens': 20,
        'output_modality': 'text',
    }
    # The model's own settings are not mutated by a per-connection override.
    assert with_defaults.settings == {'max_tokens': 10, 'output_modality': 'text'}


async def test_reconnect_with_backoff_retries_until_a_dial_succeeds() -> None:
    attempts = 0

    async def attempt() -> bool:
        nonlocal attempts
        attempts += 1
        return attempts == 2

    policy = ReconnectPolicy(max_attempts=3, base_delay=0, jitter=False)
    assert await reconnect_with_backoff(policy, attempt) is True
    assert attempts == 2


async def test_reconnect_with_backoff_gives_up_after_max_attempts() -> None:
    attempts = 0

    async def attempt() -> bool:
        nonlocal attempts
        attempts += 1
        return False

    # `jitter` is on so the randomized delay path runs too.
    policy = ReconnectPolicy(max_attempts=2, base_delay=0, jitter=True)
    assert await reconnect_with_backoff(policy, attempt) is False
    assert attempts == 2


async def test_reconnect_with_backoff_respects_the_session_wide_budget() -> None:
    async def attempt() -> bool:  # pragma: no cover
        raise AssertionError('the session budget should be checked before dialing')

    policy = ReconnectPolicy(max_reconnects=2)
    assert await reconnect_with_backoff(policy, attempt, reconnects_used=2) is False


def _patch_download(monkeypatch: pytest.MonkeyPatch, *, data: bytes, data_type: str) -> None:
    """Stand in for the network fetch `seed_user_content` performs for an `ImageUrl`."""

    async def fake_download_item(item: Any, data_format: str = 'bytes', type_format: str = 'mime') -> Any:
        return DownloadedItem[bytes](data=data, data_type=data_type)

    monkeypatch.setattr('pydantic_ai.realtime._base.download_item', fake_download_item)


async def test_seed_user_content_normalizes_text_and_images(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_download(monkeypatch, data=b'fake-png', data_type='image/png')

    part = UserPromptPart(
        content=[
            'plain',
            TextContent(content='wrapped'),
            CachePoint(),
            ImageUrl(url='https://example.com/cat.png'),
            BinaryContent(data=b'inline', media_type='image/jpeg'),
        ]
    )
    content = await seed_user_content(part, provider_name='voice', supports_images=True)

    assert content == [
        'plain',
        'wrapped',
        BinaryContent(data=b'fake-png', media_type='image/png'),
        BinaryContent(data=b'inline', media_type='image/jpeg'),
    ]


async def test_seed_user_content_accepts_a_plain_string_prompt() -> None:
    assert await seed_user_content(UserPromptPart(content='hello'), provider_name='voice', supports_images=False) == [
        'hello'
    ]


@pytest.mark.parametrize(
    'item,expected_error',
    [
        (ImageUrl(url='https://example.com/cat.png'), 'does not support images'),
        (BinaryContent(data=b'inline', media_type='image/jpeg'), 'does not support images'),
    ],
)
async def test_seed_user_content_rejects_images_a_provider_cannot_seed(item: Any, expected_error: str) -> None:
    with pytest.raises(UserError, match=expected_error):
        await seed_user_content(UserPromptPart(content=[item]), provider_name='voice', supports_images=False)


async def test_seed_user_content_rejects_a_non_image_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_download(monkeypatch, data=b'%PDF-', data_type='application/pdf')

    with pytest.raises(UserError, match='resolved to unsupported media type'):
        await seed_user_content(
            UserPromptPart(content=[ImageUrl(url='https://example.com/doc.pdf')]),
            provider_name='voice',
            supports_images=True,
        )


async def test_seed_user_content_rejects_non_image_binary_content() -> None:
    with pytest.raises(UserError, match='cannot be seeded into voice realtime history'):
        await seed_user_content(
            UserPromptPart(content=[BinaryContent(data=b'\x00', media_type='audio/wav')]),
            provider_name='voice',
            supports_images=True,
        )


async def test_seed_user_content_rejects_unrepresentable_file_kinds() -> None:
    with pytest.raises(UserError, match='`AudioUrl` cannot be seeded'):
        await seed_user_content(
            UserPromptPart(content=[AudioUrl(url='https://example.com/a.wav')]),
            provider_name='voice',
            supports_images=True,
        )


def _wav(pcm: bytes, *, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> BinaryContent:
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return BinaryContent(data=buffer.getvalue(), media_type='audio/wav')


def test_seed_speech_content_prefers_the_transcript() -> None:
    part = SpeechPart(speaker='user', transcript='hello', audio=_wav(b'\x00\x01'))
    assert seed_speech_content(part, provider_name='voice', supports_audio=False) == 'hello'


def test_seed_speech_content_keeps_a_content_less_turn_as_empty_text() -> None:
    part = SpeechPart(speaker='assistant')
    assert seed_speech_content(part, provider_name='voice', supports_audio=True) == ''


def test_seed_speech_content_rejects_transcript_less_assistant_audio() -> None:
    part = SpeechPart(speaker='assistant', audio=_wav(b'\x00\x01'))
    with pytest.raises(UserError, match='assistant `SpeechPart` without a transcript'):
        seed_speech_content(part, provider_name='voice', supports_audio=True)


def test_seed_speech_content_rejects_non_audio_bytes() -> None:
    part = SpeechPart(speaker='user', audio=BinaryContent(data=b'\x00', media_type='image/png'))
    with pytest.raises(UserError, match='cannot be seeded into realtime history'):
        seed_speech_content(part, provider_name='voice', supports_audio=True)


def test_seed_speech_content_rejects_audio_a_provider_cannot_replay() -> None:
    part = SpeechPart(speaker='user', audio=_wav(b'\x00\x01'))
    with pytest.raises(UserError, match='does not support retained user audio'):
        seed_speech_content(part, provider_name='voice', supports_audio=False)


def test_seed_speech_content_returns_retained_user_audio() -> None:
    audio = _wav(b'\x00\x01')
    part = SpeechPart(speaker='user', audio=audio)
    assert seed_speech_content(part, provider_name='voice', supports_audio=True) is audio


def test_seed_pcm_audio_extracts_raw_frames() -> None:
    assert seed_pcm_audio(_wav(b'\x00\x01\x02\x03'), provider_name='voice', sample_rate=24000) == b'\x00\x01\x02\x03'


def test_seed_pcm_audio_rejects_a_non_wav_container() -> None:
    audio = BinaryContent(data=b'\x00', media_type='audio/mpeg')
    with pytest.raises(UserError, match='cannot be seeded into voice realtime history'):
        seed_pcm_audio(audio, provider_name='voice', sample_rate=24000)


def test_seed_pcm_audio_refuses_to_resample() -> None:
    with pytest.raises(UserError, match='recorded at 16000 Hz'):
        seed_pcm_audio(_wav(b'\x00\x01', sample_rate=16000), provider_name='voice', sample_rate=24000)


def test_seed_pcm_audio_rejects_a_non_mono_pcm16_wav() -> None:
    with pytest.raises(UserError, match='expected mono 16-bit PCM WAV'):
        seed_pcm_audio(_wav(b'\x00\x01\x02\x03', channels=2), provider_name='voice', sample_rate=24000)
