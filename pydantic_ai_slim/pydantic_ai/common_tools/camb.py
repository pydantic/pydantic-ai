"""Camb.ai tools for Pydantic AI agents.

Provides multilingual voice AI capabilities including text-to-speech,
translation, transcription, voice cloning, and audio processing
using the Camb.ai API.
"""

from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import dataclass
from typing import overload

from typing_extensions import Any, TypedDict

from pydantic_ai import FunctionToolset
from pydantic_ai.tools import Tool

try:
    from camb.client import AsyncCambAI
except ImportError as _import_error:
    raise ImportError(
        'Please install `camb-sdk` to use the Camb.ai tools, '
        'you can use the `camb` optional group — `pip install "pydantic-ai-slim[camb]"`'
    ) from _import_error

__all__ = (
    'CambToolset',
    'camb_tts_tool',
    'camb_list_voices_tool',
    'camb_translate_tool',
    'camb_transcribe_tool',
    'camb_translated_tts_tool',
    'camb_clone_voice_tool',
    'camb_voice_from_description_tool',
    'camb_text_to_sound_tool',
    'camb_separate_audio_tool',
)


class CambTTSResult(TypedDict):
    """Result from text-to-speech synthesis.

    See [Camb.ai TTS documentation](https://docs.camb.ai) for more information.
    """

    text: str
    """The input text that was synthesized."""
    voice_id: int
    """The voice ID used for synthesis."""
    language: str
    """The language code used."""
    model: str
    """The speech model used."""
    audio_base64: str
    """Base64-encoded audio data."""
    file_path: str | None
    """Path to saved audio file, if `output_path` was provided."""


class CambVoice(TypedDict):
    """A voice available in the Camb.ai platform."""

    id: int
    """The unique voice identifier."""
    voice_name: str
    """The name of the voice."""
    gender: int | None
    """The gender identifier."""
    age: int | None
    """The age of the voice."""
    language: int | None
    """The language identifier."""
    description: str | None
    """Description of the voice."""


class CambTranslationResult(TypedDict):
    """Result from text translation."""

    translated_text: str
    """The translated text."""
    source_language: int
    """The source language identifier."""
    target_language: int
    """The target language identifier."""


class CambTranscriptionResult(TypedDict):
    """Result from audio transcription."""

    text: str
    """The full transcribed text."""
    segments: list[dict[str, Any]]
    """Transcript segments with start, end, text, and speaker fields."""


class CambTranslatedTTSResult(TypedDict):
    """Result from translated text-to-speech."""

    status: str
    """The task status."""
    run_id: int | None
    """The run ID for fetching results."""
    source_language: int
    """The source language identifier."""
    target_language: int
    """The target language identifier."""
    voice_id: int
    """The voice ID used."""


class CambCloneVoiceResult(TypedDict):
    """Result from voice cloning."""

    voice_id: int
    """The ID of the newly cloned voice."""


class CambVoiceFromDescriptionResult(TypedDict):
    """Result from text-to-voice generation."""

    previews: list[str]
    """URLs to preview audio files."""
    status: str
    """The task status."""


class CambTextToSoundResult(TypedDict):
    """Result from text-to-sound generation."""

    audio_base64: str
    """Base64-encoded audio data."""
    file_path: str | None
    """Path to saved audio file, if `output_path` was provided."""
    prompt: str
    """The prompt used to generate the sound."""


class CambAudioSeparationResult(TypedDict):
    """Result from audio separation."""

    foreground_audio_url: str
    """URL to the separated foreground (vocals) audio."""
    background_audio_url: str
    """URL to the separated background audio."""
    status: str
    """The task status."""


async def _poll_task(
    poll_fn: Any,
    task_id: str,
    *,
    max_attempts: int = 60,
    interval: float = 2.0,
) -> Any:
    """Poll a task until it completes or fails.

    Args:
        poll_fn: Async function to call for polling status.
        task_id: The task ID to poll.
        max_attempts: Maximum number of poll attempts.
        interval: Seconds between poll attempts.

    Returns:
        The final status result.

    Raises:
        TimeoutError: If max_attempts is exceeded.
        RuntimeError: If the task fails.
    """
    for _ in range(max_attempts):
        result = await poll_fn(task_id)
        status = result.status
        if status == 'SUCCESS':
            return result
        if status == 'ERROR':
            msg = getattr(result, 'exception_reason', None) or 'Task failed'
            raise RuntimeError(f'Camb.ai task failed: {msg}')
        await asyncio.sleep(interval)
    raise TimeoutError(f'Camb.ai task {task_id} did not complete within {max_attempts * interval}s')


@dataclass
class CambTTSTool:
    """Text-to-speech synthesis tool."""

    client: AsyncCambAI
    """The Camb.ai async client."""

    voice_id: int
    """Default voice ID for synthesis."""

    language: str
    """Default language code."""

    model: str
    """Default speech model."""

    async def __call__(
        self,
        text: str,
        voice_id: int | None = None,
        language: str | None = None,
        model: str | None = None,
        output_path: str | None = None,
    ) -> CambTTSResult:
        """Synthesizes speech from text using Camb.ai TTS.

        Args:
            text: The text to synthesize into speech.
            voice_id: Voice ID to use. Defaults to the tool's configured voice.
            language: Language code (e.g. 'en-us'). Defaults to the tool's configured language.
            model: Speech model to use. Defaults to the tool's configured model.
            output_path: Optional file path to save the audio to.

        Returns:
            The TTS result with base64-encoded audio.
        """
        effective_voice_id = voice_id or self.voice_id
        effective_language = language or self.language
        effective_model = model or self.model

        chunks: list[bytes] = []
        async for chunk in self.client.text_to_speech.tts(
            text=text,
            language=effective_language,
            voice_id=effective_voice_id,
            speech_model=effective_model,
        ):
            chunks.append(chunk)

        audio_data = b''.join(chunks)
        audio_b64 = base64.b64encode(audio_data).decode('ascii')

        file_path: str | None = None
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(audio_data)
            file_path = output_path

        return CambTTSResult(
            text=text,
            voice_id=effective_voice_id,
            language=effective_language,
            model=effective_model,
            audio_base64=audio_b64,
            file_path=file_path,
        )


@dataclass
class CambListVoicesTool:
    """Tool to list available voices."""

    client: AsyncCambAI
    """The Camb.ai async client."""

    async def __call__(self) -> list[CambVoice]:
        """Lists all available voices in the Camb.ai platform.

        Returns:
            A list of available voices with their metadata.
        """
        voices = await self.client.voice_cloning.list_voices()
        return [
            CambVoice(
                id=v.id,  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownArgumentType]
                voice_name=v.voice_name,  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownArgumentType]
                gender=v.gender,  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownArgumentType]
                age=v.age,  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownArgumentType]
                language=v.language,  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownArgumentType]
                description=v.description,  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownArgumentType]
            )
            for v in voices
            if hasattr(v, 'id')
        ]


@dataclass
class CambTranslateTool:
    """Text translation tool."""

    client: AsyncCambAI
    """The Camb.ai async client."""

    async def __call__(
        self,
        text: str,
        source_language: int,
        target_language: int,
    ) -> CambTranslationResult:
        """Translates text between languages using Camb.ai.

        Args:
            text: The text to translate.
            source_language: Source language identifier (integer code).
            target_language: Target language identifier (integer code).

        Returns:
            The translation result.
        """
        try:
            result = await self.client.translation.translation_stream(
                text=text,
                source_language=source_language,
                target_language=target_language,
            )
            translated = str(result) if not isinstance(result, str) else result
        except Exception as e:
            # The SDK raises ApiError for streaming translation responses that return
            # plain text instead of JSON. The translated text is in the error body.
            body = getattr(e, 'body', None)
            if body and isinstance(body, str) and getattr(e, 'status_code', None) == 200:
                translated = body
            else:
                raise

        return CambTranslationResult(
            translated_text=translated,
            source_language=source_language,
            target_language=target_language,
        )


@dataclass
class CambTranscribeTool:
    """Audio transcription tool."""

    client: AsyncCambAI
    """The Camb.ai async client."""

    max_poll_attempts: int
    """Maximum number of polling attempts."""

    poll_interval: float
    """Seconds between poll attempts."""

    async def __call__(
        self,
        file_path: str,
        language: int,
    ) -> CambTranscriptionResult:
        """Transcribes audio from a file using Camb.ai.

        Args:
            file_path: Path to the audio file to transcribe.
            language: Language identifier (integer code) for the audio.

        Returns:
            The transcription result with text and segments.
        """
        with open(file_path, 'rb') as f:
            task = await self.client.transcription.create_transcription(
                language=language,
                media_file=f,
            )

        assert task.task_id is not None, 'Camb.ai transcription task did not return a task_id'
        status = await _poll_task(
            self.client.transcription.get_transcription_task_status,
            task.task_id,
            max_attempts=self.max_poll_attempts,
            interval=self.poll_interval,
        )

        result = await self.client.transcription.get_transcription_result(
            status.run_id,
        )

        segments = [
            {
                'start': t.start,
                'end': t.end,
                'text': t.text,
                'speaker': t.speaker,
            }
            for t in result.transcript
        ]
        full_text = ' '.join(t.text for t in result.transcript)

        return CambTranscriptionResult(
            text=full_text,
            segments=segments,
        )


@dataclass
class CambTranslatedTTSTool:
    """Translated text-to-speech tool."""

    client: AsyncCambAI
    """The Camb.ai async client."""

    voice_id: int
    """Default voice ID."""

    max_poll_attempts: int
    """Maximum number of polling attempts."""

    poll_interval: float
    """Seconds between poll attempts."""

    async def __call__(
        self,
        text: str,
        source_language: int,
        target_language: int,
        voice_id: int | None = None,
    ) -> CambTranslatedTTSResult:
        """Translates text and synthesizes speech in the target language.

        Args:
            text: The text to translate and synthesize.
            source_language: Source language identifier (integer code).
            target_language: Target language identifier (integer code).
            voice_id: Voice ID to use. Defaults to the tool's configured voice.

        Returns:
            The translated TTS result with task status and run ID.
        """
        effective_voice_id = voice_id or self.voice_id

        task = await self.client.translated_tts.create_translated_tts(
            text=text,
            voice_id=effective_voice_id,
            source_language=source_language,
            target_language=target_language,
        )

        assert task.task_id is not None, 'Camb.ai translated TTS task did not return a task_id'
        status = await _poll_task(
            self.client.translated_tts.get_translated_tts_task_status,
            task.task_id,
            max_attempts=self.max_poll_attempts,
            interval=self.poll_interval,
        )

        return CambTranslatedTTSResult(
            status=str(status.status),
            run_id=status.run_id,
            source_language=source_language,
            target_language=target_language,
            voice_id=effective_voice_id,
        )


@dataclass
class CambCloneVoiceTool:
    """Voice cloning tool."""

    client: AsyncCambAI
    """The Camb.ai async client."""

    async def __call__(
        self,
        voice_name: str,
        gender: int,
        file_path: str,
        description: str | None = None,
    ) -> CambCloneVoiceResult:
        """Clones a voice from an audio file.

        Args:
            voice_name: Name for the cloned voice.
            gender: Gender identifier (integer code).
            file_path: Path to the audio file to clone from.
            description: Optional description of the voice.

        Returns:
            The clone result with the new voice ID.
        """
        with open(file_path, 'rb') as f:
            result = await self.client.voice_cloning.create_custom_voice(
                voice_name=voice_name,
                gender=gender,
                file=f,
                description=description,
            )

        return CambCloneVoiceResult(
            voice_id=result.voice_id,
        )


@dataclass
class CambVoiceFromDescriptionTool:
    """Text-to-voice generation tool."""

    client: AsyncCambAI
    """The Camb.ai async client."""

    max_poll_attempts: int
    """Maximum number of polling attempts."""

    poll_interval: float
    """Seconds between poll attempts."""

    async def __call__(
        self,
        text: str,
        voice_description: str,
    ) -> CambVoiceFromDescriptionResult:
        """Generates a voice from a text description.

        Args:
            text: Sample text for the voice to speak.
            voice_description: Description of the desired voice characteristics.

        Returns:
            The result with preview audio URLs.
        """
        task = await self.client.text_to_voice.create_text_to_voice(
            text=text,
            voice_description=voice_description,
        )

        assert task.task_id is not None, 'Camb.ai text-to-voice task did not return a task_id'
        status = await _poll_task(
            self.client.text_to_voice.get_text_to_voice_status,
            task.task_id,
            max_attempts=self.max_poll_attempts,
            interval=self.poll_interval,
        )

        result = await self.client.text_to_voice.get_text_to_voice_result(
            status.run_id,
        )

        return CambVoiceFromDescriptionResult(
            previews=list(result.previews),
            status=str(status.status),
        )


@dataclass
class CambTextToSoundTool:
    """Text-to-sound generation tool."""

    client: AsyncCambAI
    """The Camb.ai async client."""

    max_poll_attempts: int
    """Maximum number of polling attempts."""

    poll_interval: float
    """Seconds between poll attempts."""

    async def __call__(
        self,
        prompt: str,
        output_path: str | None = None,
    ) -> CambTextToSoundResult:
        """Generates sound effects or audio from a text prompt.

        Args:
            prompt: Description of the sound to generate.
            output_path: Optional file path to save the audio to.

        Returns:
            The result with base64-encoded audio.
        """
        task = await self.client.text_to_audio.create_text_to_audio(
            prompt=prompt,
        )

        assert task.task_id is not None, 'Camb.ai text-to-audio task did not return a task_id'
        status = await _poll_task(
            self.client.text_to_audio.get_text_to_audio_status,
            task.task_id,
            max_attempts=self.max_poll_attempts,
            interval=self.poll_interval,
        )

        chunks: list[bytes] = []
        async for chunk in self.client.text_to_audio.get_text_to_audio_result(
            status.run_id,
        ):
            chunks.append(chunk)

        audio_data = b''.join(chunks)
        audio_b64 = base64.b64encode(audio_data).decode('ascii')

        file_path: str | None = None
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(audio_data)
            file_path = output_path

        return CambTextToSoundResult(
            audio_base64=audio_b64,
            file_path=file_path,
            prompt=prompt,
        )


@dataclass
class CambSeparateAudioTool:
    """Audio separation tool."""

    client: AsyncCambAI
    """The Camb.ai async client."""

    max_poll_attempts: int
    """Maximum number of polling attempts."""

    poll_interval: float
    """Seconds between poll attempts."""

    async def __call__(
        self,
        file_path: str,
    ) -> CambAudioSeparationResult:
        """Separates audio into foreground (vocals) and background tracks.

        Args:
            file_path: Path to the audio file to separate.

        Returns:
            The result with URLs to separated audio tracks.
        """
        with open(file_path, 'rb') as f:
            task = await self.client.audio_separation.create_audio_separation(
                media_file=f,
            )

        assert task.task_id is not None, 'Camb.ai audio separation task did not return a task_id'
        status = await _poll_task(
            self.client.audio_separation.get_audio_separation_status,
            task.task_id,
            max_attempts=self.max_poll_attempts,
            interval=self.poll_interval,
        )

        result = await self.client.audio_separation.get_audio_separation_run_info(
            status.run_id,
        )

        return CambAudioSeparationResult(
            foreground_audio_url=result.foreground_audio_url,
            background_audio_url=result.background_audio_url,
            status=str(status.status),
        )


# --- Factory functions ---


def _make_client(api_key: str | None, client: AsyncCambAI | None) -> AsyncCambAI:
    if client is not None:
        return client
    resolved_key = api_key or os.environ.get('CAMB_API_KEY')
    if resolved_key is None:
        raise ValueError(
            'Camb.ai API key must be provided via `api_key` parameter or `CAMB_API_KEY` environment variable'
        )
    return AsyncCambAI(api_key=resolved_key)


@overload
def camb_tts_tool(
    api_key: str,
    *,
    voice_id: int = 147320,
    language: str = 'en-us',
    model: str = 'mars-flash',
) -> Tool[Any]: ...


@overload
def camb_tts_tool(
    *,
    client: AsyncCambAI,
    voice_id: int = 147320,
    language: str = 'en-us',
    model: str = 'mars-flash',
) -> Tool[Any]: ...


@overload
def camb_tts_tool(
    *,
    voice_id: int = 147320,
    language: str = 'en-us',
    model: str = 'mars-flash',
) -> Tool[Any]: ...


def camb_tts_tool(
    api_key: str | None = None,
    *,
    client: AsyncCambAI | None = None,
    voice_id: int = 147320,
    language: str = 'en-us',
    model: str = 'mars-flash',
) -> Tool[Any]:
    """Creates a Camb.ai text-to-speech tool.

    Args:
        api_key: The Camb.ai API key. Required if `client` is not provided.

            Set via `CAMB_API_KEY` environment variable or pass directly.
        client: An existing AsyncCambAI client. If provided, `api_key` is ignored.
        voice_id: Default voice ID for synthesis. Defaults to 147320.
        language: Default language code. Defaults to 'en-us'.
        model: Default speech model. Defaults to 'mars-flash'.
    """
    return Tool[Any](
        CambTTSTool(
            client=_make_client(api_key, client),
            voice_id=voice_id,
            language=language,
            model=model,
        ).__call__,
        name='camb_tts',
        description='Synthesizes speech from text using Camb.ai TTS. Returns base64-encoded audio data.',
    )


@overload
def camb_list_voices_tool(api_key: str) -> Tool[Any]: ...


@overload
def camb_list_voices_tool(*, client: AsyncCambAI) -> Tool[Any]: ...


@overload
def camb_list_voices_tool() -> Tool[Any]: ...


def camb_list_voices_tool(
    api_key: str | None = None,
    *,
    client: AsyncCambAI | None = None,
) -> Tool[Any]:
    """Creates a Camb.ai list voices tool.

    Args:
        api_key: The Camb.ai API key. Required if `client` is not provided.

            Set via `CAMB_API_KEY` environment variable or pass directly.
        client: An existing AsyncCambAI client. If provided, `api_key` is ignored.
    """
    return Tool[Any](
        CambListVoicesTool(client=_make_client(api_key, client)).__call__,
        name='camb_list_voices',
        description='Lists all available voices in the Camb.ai platform with their metadata.',
    )


@overload
def camb_translate_tool(api_key: str) -> Tool[Any]: ...


@overload
def camb_translate_tool(*, client: AsyncCambAI) -> Tool[Any]: ...


@overload
def camb_translate_tool() -> Tool[Any]: ...


def camb_translate_tool(
    api_key: str | None = None,
    *,
    client: AsyncCambAI | None = None,
) -> Tool[Any]:
    """Creates a Camb.ai translation tool.

    Args:
        api_key: The Camb.ai API key. Required if `client` is not provided.

            Set via `CAMB_API_KEY` environment variable or pass directly.
        client: An existing AsyncCambAI client. If provided, `api_key` is ignored.
    """
    return Tool[Any](
        CambTranslateTool(client=_make_client(api_key, client)).__call__,
        name='camb_translate',
        description='Translates text between languages using Camb.ai. Provide source and target language IDs.',
    )


@overload
def camb_transcribe_tool(
    api_key: str,
    *,
    max_poll_attempts: int = 60,
    poll_interval: float = 2.0,
) -> Tool[Any]: ...


@overload
def camb_transcribe_tool(
    *,
    client: AsyncCambAI,
    max_poll_attempts: int = 60,
    poll_interval: float = 2.0,
) -> Tool[Any]: ...


@overload
def camb_transcribe_tool(
    *,
    max_poll_attempts: int = 60,
    poll_interval: float = 2.0,
) -> Tool[Any]: ...


def camb_transcribe_tool(
    api_key: str | None = None,
    *,
    client: AsyncCambAI | None = None,
    max_poll_attempts: int = 60,
    poll_interval: float = 2.0,
) -> Tool[Any]:
    """Creates a Camb.ai transcription tool.

    Args:
        api_key: The Camb.ai API key. Required if `client` is not provided.

            Set via `CAMB_API_KEY` environment variable or pass directly.
        client: An existing AsyncCambAI client. If provided, `api_key` is ignored.
        max_poll_attempts: Maximum polling attempts for task completion. Defaults to 60.
        poll_interval: Seconds between poll attempts. Defaults to 2.0.
    """
    return Tool[Any](
        CambTranscribeTool(
            client=_make_client(api_key, client),
            max_poll_attempts=max_poll_attempts,
            poll_interval=poll_interval,
        ).__call__,
        name='camb_transcribe',
        description='Transcribes audio from a file using Camb.ai. Returns text with speaker-attributed segments.',
    )


@overload
def camb_translated_tts_tool(
    api_key: str,
    *,
    voice_id: int = 147320,
    max_poll_attempts: int = 60,
    poll_interval: float = 2.0,
) -> Tool[Any]: ...


@overload
def camb_translated_tts_tool(
    *,
    client: AsyncCambAI,
    voice_id: int = 147320,
    max_poll_attempts: int = 60,
    poll_interval: float = 2.0,
) -> Tool[Any]: ...


@overload
def camb_translated_tts_tool(
    *,
    voice_id: int = 147320,
    max_poll_attempts: int = 60,
    poll_interval: float = 2.0,
) -> Tool[Any]: ...


def camb_translated_tts_tool(
    api_key: str | None = None,
    *,
    client: AsyncCambAI | None = None,
    voice_id: int = 147320,
    max_poll_attempts: int = 60,
    poll_interval: float = 2.0,
) -> Tool[Any]:
    """Creates a Camb.ai translated text-to-speech tool.

    Args:
        api_key: The Camb.ai API key. Required if `client` is not provided.

            Set via `CAMB_API_KEY` environment variable or pass directly.
        client: An existing AsyncCambAI client. If provided, `api_key` is ignored.
        voice_id: Default voice ID. Defaults to 147320.
        max_poll_attempts: Maximum polling attempts. Defaults to 60.
        poll_interval: Seconds between poll attempts. Defaults to 2.0.
    """
    return Tool[Any](
        CambTranslatedTTSTool(
            client=_make_client(api_key, client),
            voice_id=voice_id,
            max_poll_attempts=max_poll_attempts,
            poll_interval=poll_interval,
        ).__call__,
        name='camb_translated_tts',
        description='Translates text and synthesizes speech in the target language using Camb.ai.',
    )


@overload
def camb_clone_voice_tool(api_key: str) -> Tool[Any]: ...


@overload
def camb_clone_voice_tool(*, client: AsyncCambAI) -> Tool[Any]: ...


@overload
def camb_clone_voice_tool() -> Tool[Any]: ...


def camb_clone_voice_tool(
    api_key: str | None = None,
    *,
    client: AsyncCambAI | None = None,
) -> Tool[Any]:
    """Creates a Camb.ai voice cloning tool.

    Args:
        api_key: The Camb.ai API key. Required if `client` is not provided.

            Set via `CAMB_API_KEY` environment variable or pass directly.
        client: An existing AsyncCambAI client. If provided, `api_key` is ignored.
    """
    return Tool[Any](
        CambCloneVoiceTool(client=_make_client(api_key, client)).__call__,
        name='camb_clone_voice',
        description='Clones a voice from an audio file using Camb.ai. Returns the new voice ID.',
    )


@overload
def camb_voice_from_description_tool(
    api_key: str,
    *,
    max_poll_attempts: int = 60,
    poll_interval: float = 2.0,
) -> Tool[Any]: ...


@overload
def camb_voice_from_description_tool(
    *,
    client: AsyncCambAI,
    max_poll_attempts: int = 60,
    poll_interval: float = 2.0,
) -> Tool[Any]: ...


@overload
def camb_voice_from_description_tool(
    *,
    max_poll_attempts: int = 60,
    poll_interval: float = 2.0,
) -> Tool[Any]: ...


def camb_voice_from_description_tool(
    api_key: str | None = None,
    *,
    client: AsyncCambAI | None = None,
    max_poll_attempts: int = 60,
    poll_interval: float = 2.0,
) -> Tool[Any]:
    """Creates a Camb.ai voice from description tool.

    Args:
        api_key: The Camb.ai API key. Required if `client` is not provided.

            Set via `CAMB_API_KEY` environment variable or pass directly.
        client: An existing AsyncCambAI client. If provided, `api_key` is ignored.
        max_poll_attempts: Maximum polling attempts. Defaults to 60.
        poll_interval: Seconds between poll attempts. Defaults to 2.0.
    """
    return Tool[Any](
        CambVoiceFromDescriptionTool(
            client=_make_client(api_key, client),
            max_poll_attempts=max_poll_attempts,
            poll_interval=poll_interval,
        ).__call__,
        name='camb_voice_from_description',
        description='Generates a voice from a text description using Camb.ai. Returns preview audio URLs.',
    )


@overload
def camb_text_to_sound_tool(
    api_key: str,
    *,
    max_poll_attempts: int = 60,
    poll_interval: float = 2.0,
) -> Tool[Any]: ...


@overload
def camb_text_to_sound_tool(
    *,
    client: AsyncCambAI,
    max_poll_attempts: int = 60,
    poll_interval: float = 2.0,
) -> Tool[Any]: ...


@overload
def camb_text_to_sound_tool(
    *,
    max_poll_attempts: int = 60,
    poll_interval: float = 2.0,
) -> Tool[Any]: ...


def camb_text_to_sound_tool(
    api_key: str | None = None,
    *,
    client: AsyncCambAI | None = None,
    max_poll_attempts: int = 60,
    poll_interval: float = 2.0,
) -> Tool[Any]:
    """Creates a Camb.ai text-to-sound tool.

    Args:
        api_key: The Camb.ai API key. Required if `client` is not provided.

            Set via `CAMB_API_KEY` environment variable or pass directly.
        client: An existing AsyncCambAI client. If provided, `api_key` is ignored.
        max_poll_attempts: Maximum polling attempts. Defaults to 60.
        poll_interval: Seconds between poll attempts. Defaults to 2.0.
    """
    return Tool[Any](
        CambTextToSoundTool(
            client=_make_client(api_key, client),
            max_poll_attempts=max_poll_attempts,
            poll_interval=poll_interval,
        ).__call__,
        name='camb_text_to_sound',
        description='Generates sound effects or audio from a text prompt using Camb.ai. Returns base64-encoded audio.',
    )


@overload
def camb_separate_audio_tool(
    api_key: str,
    *,
    max_poll_attempts: int = 60,
    poll_interval: float = 2.0,
) -> Tool[Any]: ...


@overload
def camb_separate_audio_tool(
    *,
    client: AsyncCambAI,
    max_poll_attempts: int = 60,
    poll_interval: float = 2.0,
) -> Tool[Any]: ...


@overload
def camb_separate_audio_tool(
    *,
    max_poll_attempts: int = 60,
    poll_interval: float = 2.0,
) -> Tool[Any]: ...


def camb_separate_audio_tool(
    api_key: str | None = None,
    *,
    client: AsyncCambAI | None = None,
    max_poll_attempts: int = 60,
    poll_interval: float = 2.0,
) -> Tool[Any]:
    """Creates a Camb.ai audio separation tool.

    Args:
        api_key: The Camb.ai API key. Required if `client` is not provided.

            Set via `CAMB_API_KEY` environment variable or pass directly.
        client: An existing AsyncCambAI client. If provided, `api_key` is ignored.
        max_poll_attempts: Maximum polling attempts. Defaults to 60.
        poll_interval: Seconds between poll attempts. Defaults to 2.0.
    """
    return Tool[Any](
        CambSeparateAudioTool(
            client=_make_client(api_key, client),
            max_poll_attempts=max_poll_attempts,
            poll_interval=poll_interval,
        ).__call__,
        name='camb_separate_audio',
        description='Separates audio into foreground (vocals) and background tracks using Camb.ai.',
    )


class CambToolset(FunctionToolset):
    """A toolset that provides Camb.ai voice and translation tools with a shared client.

    This is more efficient than creating individual tools when using multiple
    Camb.ai tools, as it shares a single API client across all tools.

    Example:
    ```python
    from pydantic_ai import Agent
    from pydantic_ai.common_tools.camb import CambToolset

    toolset = CambToolset(api_key='your-api-key')
    agent = Agent('openai:gpt-5.2', toolsets=[toolset])
    ```
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        voice_id: int = 147320,
        language: str = 'en-us',
        model: str = 'mars-flash',
        max_poll_attempts: int = 60,
        poll_interval: float = 2.0,
        include_tts: bool = True,
        include_list_voices: bool = True,
        include_translate: bool = True,
        include_transcribe: bool = True,
        include_translated_tts: bool = True,
        include_clone_voice: bool = True,
        include_voice_from_description: bool = True,
        include_text_to_sound: bool = True,
        include_separate_audio: bool = True,
        id: str | None = None,
    ):
        """Creates a Camb.ai toolset with a shared client.

        Args:
            api_key: The Camb.ai API key. Falls back to `CAMB_API_KEY` environment variable.
            voice_id: Default voice ID for TTS tools. Defaults to 147320.
            language: Default language code for TTS. Defaults to 'en-us'.
            model: Default speech model for TTS. Defaults to 'mars-flash'.
            max_poll_attempts: Maximum polling attempts for async tasks. Defaults to 60.
            poll_interval: Seconds between poll attempts. Defaults to 2.0.
            include_tts: Whether to include the TTS tool. Defaults to True.
            include_list_voices: Whether to include the list voices tool. Defaults to True.
            include_translate: Whether to include the translation tool. Defaults to True.
            include_transcribe: Whether to include the transcription tool. Defaults to True.
            include_translated_tts: Whether to include the translated TTS tool. Defaults to True.
            include_clone_voice: Whether to include the voice cloning tool. Defaults to True.
            include_voice_from_description: Whether to include the voice from description tool. Defaults to True.
            include_text_to_sound: Whether to include the text-to-sound tool. Defaults to True.
            include_separate_audio: Whether to include the audio separation tool. Defaults to True.
            id: Optional ID for the toolset, used for durable execution environments.
        """
        resolved_key = api_key or os.environ.get('CAMB_API_KEY')
        if resolved_key is None:
            raise ValueError(
                'Camb.ai API key must be provided via `api_key` parameter or `CAMB_API_KEY` environment variable'
            )
        client = AsyncCambAI(api_key=resolved_key)
        tools: list[Tool[Any]] = []

        if include_tts:
            tools.append(camb_tts_tool(client=client, voice_id=voice_id, language=language, model=model))

        if include_list_voices:
            tools.append(camb_list_voices_tool(client=client))

        if include_translate:
            tools.append(camb_translate_tool(client=client))

        if include_transcribe:
            tools.append(
                camb_transcribe_tool(client=client, max_poll_attempts=max_poll_attempts, poll_interval=poll_interval)
            )

        if include_translated_tts:
            tools.append(
                camb_translated_tts_tool(
                    client=client,
                    voice_id=voice_id,
                    max_poll_attempts=max_poll_attempts,
                    poll_interval=poll_interval,
                )
            )

        if include_clone_voice:
            tools.append(camb_clone_voice_tool(client=client))

        if include_voice_from_description:
            tools.append(
                camb_voice_from_description_tool(
                    client=client, max_poll_attempts=max_poll_attempts, poll_interval=poll_interval
                )
            )

        if include_text_to_sound:
            tools.append(
                camb_text_to_sound_tool(client=client, max_poll_attempts=max_poll_attempts, poll_interval=poll_interval)
            )

        if include_separate_audio:
            tools.append(
                camb_separate_audio_tool(
                    client=client, max_poll_attempts=max_poll_attempts, poll_interval=poll_interval
                )
            )

        super().__init__(tools, id=id)
