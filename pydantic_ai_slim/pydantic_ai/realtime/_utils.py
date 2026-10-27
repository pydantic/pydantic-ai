"""Shared implementation helpers for realtime providers."""

from __future__ import annotations

import io
import random
import wave
from collections.abc import Awaitable, Callable, MutableMapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, assert_never, overload

import anyio

from .._framing import SYSTEM_TAG, wrap
from .._instrumentation import get_instructions
from ..exceptions import UserError
from ..messages import (
    AudioUrl,
    BinaryAudio,
    BinaryContent,
    CachePoint,
    DocumentUrl,
    FilePart,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    RetryFeedbackPart,
    SpeechPart,
    SpeechPartDelta,
    SystemPromptPart,
    TextContent,
    TextPart,
    ToolReturnPart,
    UploadedFile,
    UserContent,
    UserPromptPart,
    VideoUrl,
    _retry_feedback_speaks_for_the_harness,  # pyright: ignore[reportPrivateUsage]
)
from ..models import (
    ModelRequestParameters,
    _drop_superseded_request_scoped_prompts,  # pyright: ignore[reportPrivateUsage]
    _standing_system_prompt_count,  # pyright: ignore[reportPrivateUsage]
    _wrap_non_leading_system_prompts,  # pyright: ignore[reportPrivateUsage]
    download_item,
)
from ..models._tool_choice import ResolvedToolChoice, resolve_tool_choice
from ..settings import ToolChoice
from ..tools import ToolDefinition
from .codec import TextContext
from .settings import ReconnectPolicy


def _standing_system_prompt(messages: Sequence[ModelMessage]) -> list[SystemPromptPart]:
    """The `SystemPromptPart`s the first request of `messages` opens with."""
    first_request = next((message for message in messages if isinstance(message, ModelRequest)), None)
    if first_request is None:
        return []
    standing = first_request.parts[: _standing_system_prompt_count(first_request)]
    return [part for part in standing if isinstance(part, SystemPromptPart)]


def session_instructions(messages: Sequence[ModelMessage], model_request_parameters: ModelRequestParameters) -> str:
    """The instructions a realtime session is configured with.

    The standing system prompt the first request opens with comes first, then the resolved
    instructions, the order a standard run's providers send them in. `ModelRequest.instructions`
    recorded in `messages` are never read: only `model_request_parameters.instruction_parts` count.
    """
    blocks = [part.content for part in _standing_system_prompt(messages) if part.content]
    if instructions := get_instructions(messages, model_request_parameters):
        blocks.append(instructions)
    return '\n\n'.join(blocks)


def wrap_seeded_system_prompts(messages: Sequence[ModelMessage]) -> list[ModelMessage]:
    """Turn the mid-conversation `SystemPromptPart`s in seeded history into `<system>`-tagged user text.

    The standing system prompt is left for the seeders to skip, as `session_instructions` carries it.
    A later system prompt stays where it was in the conversation, and a request-scoped one whose request
    has been answered is left out: what `Model.prepare_messages` does for a model without inline or
    request-scoped system prompts.
    """
    # Counted on the history as authored, as `prepare_messages` does: dropping a leading request-scoped
    # prompt mustn't promote the system prompt after it into the standing one `session_instructions` skips.
    standing_prompt_count = len(_standing_system_prompt(messages))
    messages = _drop_superseded_request_scoped_prompts(list(messages))
    return _wrap_non_leading_system_prompts(messages, standing_prompt_count=standing_prompt_count)


def replay_recorded_system_prompts_in_place(messages: Sequence[ModelMessage]) -> Sequence[ModelMessage]:
    """Turn the system prompts the first request of a session's own history opens with into `<system>`-tagged user text.

    For a session that opened with no request of its own: whatever it then recorded first was sent mid-session, such
    as `send(SystemPromptPart(...))` before anything else, not the standing system prompt the connection was configured
    with. Its replay keeps the instructions it opened with and leaves the standing prompt to them, so a prompt taken
    for it would be lost; as `<system>`-tagged text it is replayed where it was sent, as a later one is.
    """
    index = next((i for i, message in enumerate(messages) if isinstance(message, ModelRequest)), None)
    if index is None:
        return messages
    first_request = messages[index]
    assert isinstance(first_request, ModelRequest)
    if not (standing_count := _standing_system_prompt_count(first_request)):
        return messages
    parts = [
        UserPromptPart(content=wrap(SYSTEM_TAG, part.content), timestamp=part.timestamp)
        if position < standing_count and isinstance(part, SystemPromptPart)
        else part
        for position, part in enumerate(first_request.parts)
    ]
    return [*messages[:index], replace(first_request, parts=parts), *messages[index + 1 :]]


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


def reject_forced_tool_choice(tool_choice: ResolvedToolChoice | None, message: str) -> None:
    """Raise `UserError(message)` for a resolved `tool_choice` that forces a tool call.

    No realtime session can honor one: a provider that takes a tool choice applies it to every response,
    including the one after a tool result, so the model could never answer, and one that takes none
    can't express it. `message` gives the provider's reason and what to use instead.
    """
    if (tool_choice[0] if isinstance(tool_choice, tuple) else tool_choice) == 'required':
        raise UserError(message)


class SystemPromptText(str):
    """A mid-session system prompt sent as a text turn: `send(SystemPromptPart(...))`, or `enqueue()` of system prompts alone.

    Its value is the prompt `<system>`-tagged, so a connection that doesn't look for it delivers it as user text that
    says where it came from; one whose protocol has a channel for the operator's own text delivers `system_prompt`
    there instead.
    """

    system_prompt: str
    """The prompt as written, without the tags."""

    def __new__(cls, system_prompt: str) -> SystemPromptText:
        text = super().__new__(cls, wrap(SYSTEM_TAG, system_prompt))
        text.system_prompt = system_prompt
        return text


@dataclass
class SystemPromptContext(TextContext):
    """A mid-session system prompt added as context, without asking for a response: `send(SystemPromptPart(...), respond=False)`.

    Like `SystemPromptText`, `text` is the prompt `<system>`-tagged, for a connection that delivers it as any other
    context, and `system_prompt` the prompt as written, for one with a channel for the operator's own text.
    """

    system_prompt: str


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
) -> list[str | BinaryContent]:
    """Normalize a user prompt to replayable text and image content."""
    content: Sequence[UserContent] = [part.content] if isinstance(part.content, str) else part.content
    result: list[str | BinaryContent] = []
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


def seed_retry_feedback_text(part: RetryFeedbackPart) -> str:
    """The text a realtime session seeds retry feedback as.

    A seeded turn has no mid-conversation system role, and a retry answers one response rather than
    standing over the session, so it goes in as user text. Feedback bound for the system voice takes the
    `<system>` tagging every model without a mid-conversation system message gets, which keeps the
    model from reading it as something a person said (https://github.com/pydantic/pydantic-ai/issues/6404).
    """
    text = part.model_response()
    return wrap(SYSTEM_TAG, text) if _retry_feedback_speaks_for_the_harness(part) else text


@overload
def seed_speech_content(*, part: SpeechPart, provider_name: str, supports_audio: Literal[False]) -> str: ...


@overload
def seed_speech_content(*, part: SpeechPart, provider_name: str, supports_audio: bool) -> str | BinaryContent: ...


def seed_speech_content(*, part: SpeechPart, provider_name: str, supports_audio: bool) -> str | BinaryContent:
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


@overload
def fold_part_transcript(part: TextPart, text: str) -> tuple[TextPart, str]: ...


@overload
def fold_part_transcript(part: SpeechPart, text: str) -> tuple[SpeechPart, str]: ...


def fold_part_transcript(part: SpeechPart | TextPart, text: str) -> tuple[SpeechPart | TextPart, str]:
    """Fold a transcript event's `text` into a model part's text by the rule of `accumulate_transcript`.

    Returns the updated part and the newly appended text.
    """
    if isinstance(part, TextPart):
        content, appended = accumulate_transcript(part.content, text)
        return replace(part, content=content), appended
    transcript, appended = accumulate_transcript(part.transcript or '', text)
    return replace(part, transcript=transcript or None), appended


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


# What a spoken user turn with no transcript (transcription off, or the transcript never arrived)
# replays as. Without it the turn would vanish and the fresh session would read the answer to it as the
# model speaking unprompted. Written here, on the way to the provider, and never persisted: history
# keeps the turn as the `SpeechPart` it was.
_TRANSCRIPTLESS_SPEECH_MARKER = '[The user spoke; no transcript is available.]'


def without_media(message: ModelMessage) -> ModelMessage | None:
    """Strip audio and images from a message, dropping it entirely when nothing is left to replay.

    A user `SpeechPart` without a transcript has nothing left once its audio goes, so it is replaced by
    a marker saying the user spoke.
    """
    if isinstance(message, ModelRequest):
        request_parts: list[ModelRequestPart] = []
        for part in message.parts:
            if isinstance(part, SpeechPart):
                if part.transcript:
                    request_parts.append(replace(part, audio=None))
                else:
                    request_parts.append(UserPromptPart(content=_TRANSCRIPTLESS_SPEECH_MARKER))
            elif isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                if text := [item for item in part.content if isinstance(item, (str, TextContent))]:
                    request_parts.append(replace(part, content=text))
            elif isinstance(part, ToolReturnPart) and part.files:
                request_parts.append(replace(part, content=part.model_response_str(wrap_if_error=False)))
            else:
                request_parts.append(part)
        return replace(message, parts=request_parts) if request_parts else None
    response_parts = [
        replace(part, audio=None) if isinstance(part, SpeechPart) else part
        for part in message.parts
        if not isinstance(part, FilePart)
    ]
    return replace(message, parts=response_parts) if response_parts else None
