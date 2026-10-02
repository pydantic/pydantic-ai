"""Shared implementation helpers for realtime providers."""

from __future__ import annotations

import io
import random
import wave
from collections.abc import Awaitable, Callable, MutableMapping, Sequence
from typing import Literal, overload

import anyio
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
    SpeechPartDelta,
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


DEFAULT_MAX_RECONNECTS = 50
"""The `ReconnectPolicy.max_reconnects` default."""


async def reconnect_with_backoff(
    policy: ReconnectPolicy, attempt: Callable[[], Awaitable[bool]], *, reconnects_used: int = 0
) -> bool:
    """Retry an attempt with exponential backoff until it succeeds or its budget is exhausted."""
    if reconnects_used >= policy.get('max_reconnects', DEFAULT_MAX_RECONNECTS):
        return False
    for i in range(policy.get('max_attempts', 3)):
        delay = min(policy.get('max_delay', 30.0), policy.get('base_delay', 0.5) * (2**i))
        if policy.get('jitter', True):
            delay *= 0.5 + random.random() * 0.5
        await anyio.sleep(max(delay, 0))
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


def pcm_to_wav(data: bytes, sample_rate: int) -> bytes:
    """Wrap mono 16-bit PCM bytes in a WAV container at `sample_rate`."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(data)
    return buffer.getvalue()


def accumulate_transcript(accumulated: str, text: str) -> tuple[str, str]:
    """Fold a transcript event's `text` into the running transcript, returning `(new_accumulated, appended)`.

    Providers deliver transcripts two different ways: as incremental deltas (each event carries a new
    piece) or as a single final event carrying the full text. Both are handled by one rule: if `text`
    extends what we already have (the accumulated transcript is a prefix of it), it is a cumulative/full
    update and only the new suffix is appended; otherwise `text` is an incremental piece appended as-is.
    The second element is the newly appended text (empty when a final event merely repeats the deltas),
    suitable for a [`PartDeltaEvent`][pydantic_ai.messages.PartDeltaEvent].

    A cumulative/final snapshot can differ from the accumulated deltas by leading/trailing whitespace —
    OpenAI's input-audio-transcription deltas start with a leading space that the `.completed` snapshot
    trims — so the prefix check is applied to the stripped text too, adopting the snapshot as
    authoritative rather than concatenating a near-duplicate.
    """
    if accumulated and text.startswith(accumulated):
        return text, text[len(accumulated) :]
    stripped = accumulated.strip()
    if stripped and (stripped_text := text.strip()).startswith(stripped):
        return text, stripped_text[len(stripped) :]
    return accumulated + text, text


def user_transcript_update(previous: str, text: str, *, cumulative: bool) -> tuple[str, SpeechPartDelta | None]:
    """Fold a user transcript event into the running text, returning it with the delta to emit.

    An incremental piece is accumulated by [`accumulate_transcript`][pydantic_ai.realtime._utils.accumulate_transcript]
    and surfaced as an appended delta. A cumulative snapshot is adopted wholesale, because a provider
    that sends snapshots may revise earlier words rather than only extend them: when it merely extends,
    the new suffix is still an appended delta (what a live transcript wants), but a revision can't be
    expressed by appending, so it goes out as a replacement instead. `None` when nothing changed.
    """

    def delta(transcript: str, added: str) -> SpeechPartDelta:
        return SpeechPartDelta(speaker='user', transcript_delta=added, transcript=transcript)

    if not cumulative:
        transcript, appended = accumulate_transcript(previous, text)
        return transcript, delta(transcript, appended) if appended else None
    if text == previous:
        return previous, None
    if previous and text.startswith(previous):
        return text, delta(text, text[len(previous) :])
    stripped = previous.strip()
    if stripped and (stripped_text := text.strip()).startswith(stripped):
        return text, delta(text, stripped_text[len(stripped) :])
    if not previous:
        return text, delta(text, text)
    # A revision: nothing was *added*, so only the corrected whole is reported.
    return text, delta(text, '')
