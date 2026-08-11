"""Shared implementation helpers for realtime providers."""

from __future__ import annotations

import asyncio
import io
import random
import wave
from collections.abc import Awaitable, Callable, MutableMapping, Sequence
from typing import Literal, overload

from typing_extensions import assert_never

from ..exceptions import UserError
from ..messages import (
    AudioUrl,
    BinaryAudio,
    BinaryContent,
    CachePoint,
    DocumentUrl,
    ImageUrl,
    SpeechPart,
    TextContent,
    UploadedFile,
    UserContent,
    UserPromptPart,
    VideoUrl,
)
from ..models import ModelRequestParameters, download_item
from ..models._tool_choice import ResolvedToolChoice, resolve_tool_choice
from ..settings import ToolChoice
from ..tools import ToolDefinition
from .codec import RealtimeSessionInput
from .settings import ReconnectPolicy


def resolve_advertised_tools(
    tools: list[ToolDefinition] | None, tool_choice: ToolChoice
) -> tuple[list[ToolDefinition], ResolvedToolChoice | None]:
    """Resolve the tools and tool-choice mode advertised for a realtime session."""
    tools = tools or []
    if tool_choice is None:
        return tools, None
    resolved = resolve_tool_choice(
        {'tool_choice': tool_choice}, ModelRequestParameters(function_tools=tools, allow_text_output=True)
    )
    if resolved == 'none':
        return [], resolved
    if isinstance(resolved, tuple):
        _, allowed = resolved
        return [tool for tool in tools if tool.name in allowed], resolved
    return tools, resolved


async def reconnect_with_backoff(
    policy: ReconnectPolicy, attempt: Callable[[], Awaitable[bool]], *, reconnects_used: int = 0
) -> bool:
    """Retry an attempt with exponential backoff until it succeeds or its budget is exhausted."""
    if reconnects_used >= policy.get('max_reconnects', 50):
        return False
    for i in range(policy.get('max_attempts', 3)):
        delay = min(policy.get('max_delay', 30.0), policy.get('base_delay', 0.5) * (2**i))
        if policy.get('jitter', True):
            delay *= 0.5 + random.random() * 0.5
        await asyncio.sleep(delay)
        if await attempt():
            return True
    return False


def inject_trace_context(headers: MutableMapping[str, str]) -> None:
    """Add the current W3C trace context to WebSocket handshake headers."""
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    TraceContextTextMapPropagator().inject(headers)


async def seed_user_content(
    *, part: UserPromptPart, provider_name: str, supports_images: bool
) -> list[RealtimeSessionInput]:
    """Normalize a user prompt to replayable text and image content."""
    content: Sequence[UserContent] = [part.content] if isinstance(part.content, str) else part.content
    result: list[RealtimeSessionInput] = []
    for item in content:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, TextContent):
            result.append(item.content)
        elif isinstance(item, CachePoint):
            continue
        elif isinstance(item, ImageUrl):
            if not supports_images:
                raise UserError(
                    f'{provider_name} realtime sessions do not support images in seeded history or tool results. '
                    'Remove the image, or use a realtime provider that supports images.'
                )
            downloaded = await download_item(item, data_format='bytes')
            image = BinaryContent(data=downloaded['data'], media_type=downloaded['data_type'])
            if not image.is_image:
                raise UserError(
                    f'`ImageUrl` resolved to unsupported media type {image.media_type!r} in a '
                    f'{provider_name} realtime session. Use a URL that returns an image, or remove it.'
                )
            result.append(image)
        elif isinstance(item, BinaryContent):
            if not item.is_image:
                raise UserError(
                    f'`BinaryContent` with media type {item.media_type!r} cannot be sent to {provider_name} '
                    'in a realtime session. Convert it to text or an image, or remove it '
                    'from `message_history` or the tool result.'
                )
            if not supports_images:
                raise UserError(
                    f'{provider_name} realtime sessions do not support images in seeded history or tool results. '
                    'Remove the image, or use a realtime provider that supports images.'
                )
            result.append(item)
        elif isinstance(item, (AudioUrl, VideoUrl, DocumentUrl, UploadedFile)):
            content_type = item.__class__.__name__
            raise UserError(
                f'`{content_type}` cannot be sent to {provider_name} in a realtime session. '
                'Convert it to text or an inline image, or remove it from `message_history` or the tool result.'
            )
        else:
            assert_never(item)
    return result


@overload
def seed_speech_content(*, part: SpeechPart, provider_name: str, supports_audio: Literal[False]) -> str: ...


@overload
def seed_speech_content(*, part: SpeechPart, provider_name: str, supports_audio: bool) -> RealtimeSessionInput: ...


def seed_speech_content(*, part: SpeechPart, provider_name: str, supports_audio: bool) -> RealtimeSessionInput:
    """Return replayable speech content, preferring its transcript."""
    if part.transcript is not None:
        return part.transcript
    if part.audio is None:
        return ''
    if part.speaker == 'assistant':
        raise UserError(
            f'An assistant `SpeechPart` without a transcript cannot be seeded into {provider_name} realtime history. '
            'Enable output transcription or filter the part from `message_history` before connecting.'
        )
    if not part.audio.is_audio:
        raise UserError(
            f'`SpeechPart.audio` with media type {part.audio.media_type!r} cannot be seeded into realtime history. '
            'Use retained audio bytes or filter the part from `message_history` before connecting.'
        )
    if not supports_audio:
        raise UserError(
            f'{provider_name} realtime history seeding does not support retained user audio. '
            'Enable input transcription so the turn has a transcript, or filter the part from `message_history`.'
        )
    return part.audio


def seed_pcm_audio(*, audio: BinaryContent, provider_name: str, sample_rate: int) -> bytes:
    """Extract mono PCM16 bytes from retained WAV audio."""
    if audio.media_type != 'audio/wav':
        raise UserError(
            f'`SpeechPart.audio` with media type {audio.media_type!r} cannot be seeded into '
            f'{provider_name} realtime history. Use WAV audio matching the target session input format.'
        )
    try:
        with wave.open(io.BytesIO(audio.data), 'rb') as wav:
            source_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            compression = wav.getcomptype()
            if source_rate != sample_rate:
                raise UserError(
                    f'Cannot seed retained audio recorded at {source_rate} Hz into a {provider_name} realtime session '
                    f'expecting {sample_rate} Hz. Resample it before passing `message_history`.'
                )
            if channels != 1 or sample_width != 2 or compression != 'NONE':
                raise UserError(
                    f'Cannot seed retained audio into {provider_name} realtime history: expected mono 16-bit PCM WAV, '
                    f'got {channels} channel(s), {sample_width * 8}-bit samples, compression {compression!r}.'
                )
            frame_count = wav.getnframes()
            pcm = wav.readframes(frame_count)
            if len(pcm) != frame_count * channels * sample_width:
                raise wave.Error('truncated audio data')
    except (EOFError, wave.Error) as e:
        raise UserError(
            f'`SpeechPart.audio` cannot be seeded into {provider_name} realtime history because it is not valid WAV audio.'
        ) from e
    return pcm


def require_pcm_audio(audio: BinaryAudio, *, provider_name: str) -> None:
    """Require the raw PCM media type accepted by realtime wire protocols."""
    if audio.media_type != 'audio/pcm':
        raise UserError(
            f'{provider_name} realtime connections require raw PCM audio (`media_type="audio/pcm"`), '
            f'not {audio.media_type!r}. Send WAV audio through `RealtimeSession.send_audio()` so it can be unwrapped.'
        )
