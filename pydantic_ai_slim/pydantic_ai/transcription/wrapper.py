from __future__ import annotations

from dataclasses import dataclass

from .base import TranscriptionInput, TranscriptionModel
from .result import TranscriptionResult
from .settings import TranscriptionSettings


@dataclass(init=False)
class WrapperTranscriptionModel(TranscriptionModel):
    """Base class for transcription models that wrap another model.

    Use this as a base class to create custom transcription model wrappers
    that modify behavior (e.g., caching, logging, rate limiting) while
    delegating to an underlying model.

    By default, all methods are passed through to the wrapped model.
    Override specific methods to customize behavior.
    """

    wrapped: TranscriptionModel
    """The underlying transcription model being wrapped."""

    def __init__(self, wrapped: TranscriptionModel | str):
        """Initialize the wrapper with a transcription model.

        Args:
            wrapped: The model to wrap. Can be a
                [`TranscriptionModel`][pydantic_ai.transcription.TranscriptionModel] instance
                or a model name string (e.g., `'openai:gpt-transcribe'`).
        """
        from . import infer_transcription_model

        super().__init__()
        self.wrapped = infer_transcription_model(wrapped) if isinstance(wrapped, str) else wrapped

    async def transcribe(
        self, audio: TranscriptionInput, *, settings: TranscriptionSettings | None = None
    ) -> TranscriptionResult:
        return await self.wrapped.transcribe(audio, settings=settings)

    @property
    def model_name(self) -> str:
        return self.wrapped.model_name

    @property
    def system(self) -> str:
        return self.wrapped.system

    @property
    def settings(self) -> TranscriptionSettings | None:
        """Get the settings from the wrapped transcription model."""
        return self.wrapped.settings

    @property
    def base_url(self) -> str | None:
        return self.wrapped.base_url

    def __getattr__(self, item: str):
        if item == 'wrapped':
            raise AttributeError(item)
        return getattr(self.wrapped, item)
