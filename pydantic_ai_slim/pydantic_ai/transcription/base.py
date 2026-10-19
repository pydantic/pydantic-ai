from abc import ABC, abstractmethod

from typing_extensions import TypeAliasType

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AudioUrl, BinaryContent

from .result import TranscriptionResult
from .settings import TranscriptionSettings, merge_transcription_settings

TranscriptionInput = TypeAliasType('TranscriptionInput', AudioUrl | BinaryContent)
"""The audio to transcribe: an [`AudioUrl`][pydantic_ai.messages.AudioUrl] to download, or
[`BinaryContent`][pydantic_ai.messages.BinaryContent] holding the audio itself.

Video containers like MP4 and WebM are accepted too, as long as the provider supports them, since their
audio track is what gets transcribed.
"""


class TranscriptionModel(ABC):
    """Abstract base class for transcription (speech-to-text) models.

    Implement this class to create a custom transcription model. For most use cases, use one of the
    built-in implementations, like
    [`OpenAITranscriptionModel`][pydantic_ai.transcription.openai.OpenAITranscriptionModel].
    """

    _settings: TranscriptionSettings | None = None

    def __init__(
        self,
        *,
        settings: TranscriptionSettings | None = None,
    ) -> None:
        """Initialize the model with optional settings.

        Args:
            settings: Model-specific settings that will be used as defaults for this model.
        """
        self._settings = settings

    @property
    def settings(self) -> TranscriptionSettings | None:
        """Get the default settings for this model."""
        return self._settings

    @property
    def base_url(self) -> str | None:
        """The base URL for the provider API, if available."""
        return None

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The name of the transcription model."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def system(self) -> str:
        """The transcription model provider/system identifier (e.g. `'openai'`)."""
        raise NotImplementedError()

    @abstractmethod
    async def transcribe(
        self, audio: TranscriptionInput, *, settings: TranscriptionSettings | None = None
    ) -> TranscriptionResult:
        """Transcribe the given audio to text.

        Args:
            audio: The audio to transcribe.
            settings: Optional settings to override the model's defaults.

        Returns:
            A [`TranscriptionResult`][pydantic_ai.transcription.TranscriptionResult] containing the
            transcript and metadata.
        """
        raise NotImplementedError

    def prepare_transcribe(
        self, audio: TranscriptionInput, settings: TranscriptionSettings | None = None
    ) -> TranscriptionSettings:
        """Validate the audio input and merge the settings for a transcription request.

        Subclasses should call this at the start of their `transcribe()` implementation.

        Args:
            audio: The audio to transcribe.
            settings: Optional settings to merge with the model's defaults.

        Returns:
            The merged settings.

        Raises:
            UserError: If `audio` isn't an `AudioUrl`, or `BinaryContent` with an audio or video media type.
        """
        if not isinstance(audio, (AudioUrl, BinaryContent)):
            raise UserError('Transcription input must be an `AudioUrl` or `BinaryContent` instance')
        if isinstance(audio, BinaryContent) and not (audio.is_audio or audio.is_video):
            raise UserError(f'Transcription input must have an audio or video media type, got {audio.media_type!r}')

        # Read the property, not the field: wrappers leave `_settings` unset and delegate `settings`
        # to the model they wrap, so the field would drop that model's defaults.
        return merge_transcription_settings(self.settings, settings) or {}
