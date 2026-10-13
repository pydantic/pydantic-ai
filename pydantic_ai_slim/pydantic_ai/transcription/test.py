import uuid
from dataclasses import dataclass

from pydantic_ai._utils import estimate_string_tokens
from pydantic_ai.usage import RequestUsage

from .base import TranscriptionInput, TranscriptionModel
from .result import TranscriptionResult
from .settings import TranscriptionSettings


# `eq=False`: a model that records the last call into itself compares by identity. `_settings` lives
# on the non-dataclass base, so it is outside generated field equality, which would call two
# differently configured models equal, merge them to the first, and leave the class unhashable.
@dataclass(init=False, eq=False)
class TestTranscriptionModel(TranscriptionModel):
    """A deterministic transcription model for testing.

    This model returns a fixed transcript without making any API calls, and records the audio and
    settings used in the last call via the `last_audio` and `last_settings` attributes.

    Example:
    ```python
    from pydantic_ai import BinaryContent, Transcriber
    from pydantic_ai.transcription import TestTranscriptionModel

    test_model = TestTranscriptionModel(text='Hello from the test model.')
    transcriber = Transcriber('openai:gpt-4o-transcribe')


    async def main():
        with transcriber.override(model=test_model):
            audio = BinaryContent(data=b'...', media_type='audio/wav')
            result = await transcriber.transcribe(audio, settings={'language': 'en'})
            print(result.text)
            #> Hello from the test model.
            print(test_model.last_settings)
            #> {'language': 'en'}
    ```
    """

    # NOTE: Avoid test discovery by pytest.
    __test__ = False

    text: str
    """The transcript to return from every call."""

    _model_name: str
    """The model name to report in results."""

    _provider_name: str
    """The provider name to report in results."""

    last_audio: TranscriptionInput | None = None
    """The audio passed to the most recent transcribe call."""

    last_settings: TranscriptionSettings | None = None
    """The settings used in the most recent transcribe call."""

    def __init__(
        self,
        model_name: str = 'test',
        *,
        provider_name: str = 'test',
        text: str = 'success (no audio transcribed)',
        settings: TranscriptionSettings | None = None,
    ):
        """Initialize the test transcription model.

        Args:
            model_name: The model name to report in results.
            provider_name: The provider name to report in results.
            text: The transcript to return from every call.
            settings: Optional default settings for the model.
        """
        self._model_name = model_name
        self._provider_name = provider_name
        self.text = text
        self.last_audio = None
        self.last_settings = None
        super().__init__(settings=settings)

    @property
    def model_name(self) -> str:
        """The transcription model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The transcription model provider."""
        return self._provider_name

    async def transcribe(
        self, audio: TranscriptionInput, *, settings: TranscriptionSettings | None = None
    ) -> TranscriptionResult:
        settings = self.prepare_transcribe(audio, settings)
        self.last_audio = audio
        self.last_settings = settings

        return TranscriptionResult(
            self.text,
            usage=RequestUsage(output_tokens=estimate_string_tokens(self.text)),
            model_name=self.model_name,
            provider_name=self.system,
            provider_response_id=str(uuid.uuid4()),
        )
