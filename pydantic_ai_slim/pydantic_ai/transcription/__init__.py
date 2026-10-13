from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, get_args

from typing_extensions import TypeAliasType

from pydantic_ai import _utils
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import OpenAIChatCompatibleProvider, OpenAIResponsesCompatibleProvider
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.providers import Provider, infer_provider

from .base import TranscriptionInput, TranscriptionModel
from .instrumented import InstrumentedTranscriptionModel, instrument_transcription_model
from .result import TranscriptionResult, TranscriptionSegment
from .settings import TranscriptionSettings, merge_transcription_settings
from .test import TestTranscriptionModel
from .wrapper import WrapperTranscriptionModel

__all__ = [
    'InstrumentedTranscriptionModel',
    'KnownTranscriptionModelName',
    'TestTranscriptionModel',
    'Transcriber',
    'TranscriptionInput',
    'TranscriptionModel',
    'TranscriptionResult',
    'TranscriptionSegment',
    'TranscriptionSettings',
    'WrapperTranscriptionModel',
    'infer_transcription_model',
    'instrument_transcription_model',
    'merge_transcription_settings',
]

KnownTranscriptionModelName = TypeAliasType(
    'KnownTranscriptionModelName',
    Literal[
        'openai:gpt-transcribe',
        'openai:gpt-4o-transcribe',
        'openai:gpt-4o-mini-transcribe',
        'openai:gpt-4o-mini-transcribe-2025-12-15',
        'openai:gpt-4o-transcribe-diarize',
        'openai:whisper-1',
    ],
)
"""Known model names that can be used with the `model` parameter of [`Transcriber`][pydantic_ai.transcription.Transcriber].

`KnownTranscriptionModelName` is provided as a concise way to specify a transcription model.
"""

# As with embeddings, we assume that every chat and responses-compatible provider also supports the
# transcriptions endpoint, as at worst the user would get a `ModelHTTPError`.
OpenAITranscriptionCompatibleProvider = OpenAIChatCompatibleProvider | OpenAIResponsesCompatibleProvider


def infer_transcription_model(
    model: TranscriptionModel | KnownTranscriptionModelName | str,
    *,
    provider_factory: Callable[[str], Provider[Any]] = infer_provider,
) -> TranscriptionModel:
    """Infer the transcription model from the name."""
    if isinstance(model, TranscriptionModel):
        return model

    try:
        provider_name, model_name = model.split(':', maxsplit=1)
    except ValueError as e:
        raise ValueError('You must provide a provider prefix when specifying a transcription model name') from e

    model_kind = provider_name
    if model_kind.startswith('gateway/'):
        from ..providers.gateway import normalize_gateway_provider

        model_kind = normalize_gateway_provider(model_kind)

    # `openai-chat` / `openai-responses` aren't listed: there's no chat-vs-responses split for the
    # transcriptions API, and `normalize_gateway_provider` returns `gateway/openai` as `openai`.
    if model_kind not in (
        'openai',
        *get_args(OpenAIChatCompatibleProvider.__value__),
        *get_args(OpenAIResponsesCompatibleProvider.__value__),
    ):
        raise UserError(f'Provider {provider_name!r} does not support transcription.')

    from .openai import OpenAITranscriptionModel

    return OpenAITranscriptionModel(model_name, provider=provider_factory(provider_name))


# `eq=False`: a facade over a live model and the provider client behind it compares by identity.
# `instrument` is its only dataclass field, so generated field equality would compare two transcribers
# over different models as equal, and would leave the class unhashable.
@dataclass(init=False, eq=False)
class Transcriber:
    """High-level interface for transcribing audio to text.

    The `Transcriber` class provides a convenient way to turn recorded speech into text using various
    speech-to-text model providers. It handles model inference, settings management, and optional
    OpenTelemetry instrumentation.

    Example:
    ```python
    from pydantic_ai import BinaryContent, Transcriber

    transcriber = Transcriber('openai:gpt-4o-transcribe')


    async def main():
        audio = BinaryContent(data=b'...', media_type='audio/ogg')
        result = await transcriber.transcribe(audio)
        print(result.text)
        #> Hi, I'd like to book a table for two tonight at seven.
    ```
    """

    instrument: InstrumentationSettings | bool | None
    """Options to automatically instrument with OpenTelemetry.

    Set to `True` to use default instrumentation settings, which will use Logfire if it's configured.
    Set to an instance of [`InstrumentationSettings`][pydantic_ai.models.instrumented.InstrumentationSettings] to customize.
    If this isn't set, then the last value set by
    [`Transcriber.instrument_all()`][pydantic_ai.transcription.Transcriber.instrument_all]
    will be used, which defaults to False.
    See the [Debugging and Monitoring guide](https://pydantic.dev/docs/ai/integrations/logfire/) for more info.
    """

    _instrument_default: ClassVar[InstrumentationSettings | bool] = False

    def __init__(
        self,
        model: TranscriptionModel | KnownTranscriptionModelName | str,
        *,
        settings: TranscriptionSettings | None = None,
        defer_model_check: bool = True,
        instrument: InstrumentationSettings | bool | None = None,
    ) -> None:
        """Initialize a Transcriber.

        Args:
            model: The transcription model to use. Can be specified as:

                - A model name string in the format `'provider:model-name'`
                  (e.g., `'openai:gpt-4o-transcribe'`)
                - A [`TranscriptionModel`][pydantic_ai.transcription.TranscriptionModel] instance
            settings: Optional [`TranscriptionSettings`][pydantic_ai.transcription.TranscriptionSettings]
                to use as defaults for all transcribe calls.
            defer_model_check: Whether to defer resolving the model name to a model instance, and the
                provider authentication that resolution requires, until the first transcribe call.
                Set to `False` to resolve the model immediately on construction.
            instrument: OpenTelemetry instrumentation settings. Set to `True` to enable with defaults,
                or pass an [`InstrumentationSettings`][pydantic_ai.models.instrumented.InstrumentationSettings]
                instance to customize. If `None`, uses the value from
                [`Transcriber.instrument_all()`][pydantic_ai.transcription.Transcriber.instrument_all].
        """
        self._model = model if defer_model_check else infer_transcription_model(model)
        self._settings = settings
        self.instrument = instrument

        self._override_model: ContextVar[TranscriptionModel | None] = ContextVar('_override_model', default=None)

    @staticmethod
    def instrument_all(instrument: InstrumentationSettings | bool = True) -> None:
        """Set the default instrumentation options for all transcribers where `instrument` is not explicitly set.

        This is useful for enabling instrumentation globally without modifying each transcriber individually.

        Args:
            instrument: Instrumentation settings to use as the default. Set to `True` for default settings,
                `False` to disable, or pass an
                [`InstrumentationSettings`][pydantic_ai.models.instrumented.InstrumentationSettings]
                instance to customize.
        """
        Transcriber._instrument_default = instrument

    @property
    def model(self) -> TranscriptionModel | KnownTranscriptionModelName | str:
        """The transcription model used by this transcriber."""
        return self._model

    @contextmanager
    def override(
        self,
        *,
        model: TranscriptionModel | KnownTranscriptionModelName | str | _utils.Unset = _utils.UNSET,
    ) -> Generator[None]:
        """Context manager to temporarily override the transcription model.

        Useful for testing or dynamically switching models.

        Args:
            model: The transcription model to use within this context.

        Example:
        ```python
        from pydantic_ai import BinaryContent, Transcriber

        transcriber = Transcriber('openai:gpt-4o-transcribe')


        async def main():
            audio = BinaryContent(data=b'...', media_type='audio/ogg')
            # Temporarily use a different model
            with transcriber.override(model='openai:whisper-1'):
                result = await transcriber.transcribe(audio)
                print(result.model_name)
                #> whisper-1
        ```
        """
        if _utils.is_set(model):
            model_token = self._override_model.set(infer_transcription_model(model))
        else:
            model_token = None

        try:
            yield
        finally:
            if model_token is not None:
                self._override_model.reset(model_token)

    async def transcribe(
        self, audio: TranscriptionInput, *, settings: TranscriptionSettings | None = None
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio: The audio to transcribe, as an [`AudioUrl`][pydantic_ai.messages.AudioUrl] or
                [`BinaryContent`][pydantic_ai.messages.BinaryContent] with an audio (or video) media type.
            settings: Optional settings to override the transcriber's default settings for this call.

        Returns:
            A [`TranscriptionResult`][pydantic_ai.transcription.TranscriptionResult] containing the
            transcript and metadata about the operation.

        Raises:
            UserError: If the audio input isn't supported by the model.
        """
        model = self._get_model()
        settings = merge_transcription_settings(self._settings, settings)
        return await model.transcribe(audio, settings=settings)

    def transcribe_sync(
        self, audio: TranscriptionInput, *, settings: TranscriptionSettings | None = None
    ) -> TranscriptionResult:
        """Synchronous version of [`transcribe()`][pydantic_ai.transcription.Transcriber.transcribe]."""
        return _utils.run_until_complete(self.transcribe(audio, settings=settings))

    def _get_model(self) -> TranscriptionModel:
        """Create a model configured for this transcriber.

        Returns:
            The transcription model to use, with instrumentation applied if configured.
        """
        model_: TranscriptionModel
        if some_model := self._override_model.get():
            model_ = some_model
        else:
            model_ = self._model = infer_transcription_model(self.model)

        instrument = self.instrument
        if instrument is None:
            instrument = self._instrument_default

        return instrument_transcription_model(model_, instrument)
