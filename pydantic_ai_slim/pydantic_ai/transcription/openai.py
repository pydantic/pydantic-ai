from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, assert_never, cast

from typing_extensions import TypedDict

from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UserError
from pydantic_ai.messages import AudioUrl, BinaryContent
from pydantic_ai.models import check_allow_model_requests, download_item
from pydantic_ai.models._decode_errors import check_json_response, map_decode_errors
from pydantic_ai.providers import Provider, infer_provider
from pydantic_ai.usage import RequestUsage

from . import OpenAITranscriptionCompatibleProvider
from .base import TranscriptionInput, TranscriptionModel
from .result import TranscriptionResult, TranscriptionSegment
from .settings import TranscriptionSettings

try:
    from openai import APIConnectionError, APIStatusError, AsyncOpenAI, Omit
    from openai.types.audio import Transcription, TranscriptionVerbose
    from openai.types.audio.transcription import UsageDuration, UsageTokens
    from openai.types.audio.transcription_verbose import Usage as VerboseUsageDuration
    from openai.types.audio_model import AudioModel as LatestOpenAITranscriptionModelNames

    from pydantic_ai.models.openai import OMIT
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the OpenAI transcription model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

OpenAITranscriptionModelName = str | LatestOpenAITranscriptionModelNames
"""Possible OpenAI transcription model names.

See the [OpenAI speech-to-text guide](https://developers.openai.com/api/docs/guides/speech-to-text)
for available models.
"""

_FILE_EXTENSIONS: dict[str, str] = {
    'audio/flac': 'flac',
    'audio/x-flac': 'flac',
    'audio/mpeg': 'mp3',
    'audio/mp3': 'mp3',
    'audio/mpga': 'mpga',
    'audio/mp4': 'm4a',
    'audio/m4a': 'm4a',
    'audio/x-m4a': 'm4a',
    'audio/mp4a-latm': 'm4a',
    'audio/ogg': 'ogg',
    'audio/wav': 'wav',
    'audio/x-wav': 'wav',
    'audio/wave': 'wav',
    'audio/webm': 'webm',
    'video/webm': 'webm',
    'video/mp4': 'mp4',
    'video/mpeg': 'mpeg',
}
"""Filename extension per media type the transcription endpoint accepts.

The endpoint identifies the audio format by the uploaded file's extension, and accepts flac, mp3, mp4,
mpeg, mpga, m4a, ogg, wav and webm. Several aliases are listed per format because `mimetypes` guesses
differ across platforms and Python versions.
"""


class OpenAITranscriptionSettings(TranscriptionSettings, total=False):
    """Settings used for an OpenAI transcription request.

    All fields from [`TranscriptionSettings`][pydantic_ai.transcription.TranscriptionSettings] are
    supported, plus OpenAI-specific settings prefixed with `openai_`.
    """

    # ALL FIELDS MUST BE `openai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    openai_prompt: str
    """Text to guide the model's style, or to continue a previous audio segment.

    The prompt should match the audio language. It's a good place for the spelling of names and
    domain-specific terms. See [OpenAI's prompting guide](https://developers.openai.com/api/docs/guides/speech-to-text#prompting).
    """

    openai_response_format: Literal['json', 'verbose_json']
    """The response format to request. Defaults to `'json'`, which every OpenAI transcription model supports.

    Set to `'verbose_json'` to also receive timestamped
    [`segments`][pydantic_ai.transcription.TranscriptionResult.segments], and the detected language and
    audio duration in [`provider_details`][pydantic_ai.transcription.TranscriptionResult.provider_details].
    Supported by `whisper-1`.
    """


# `eq=False`: a model bound to a live provider client compares by identity. `_settings` lives on the
# non-dataclass base, so it is outside generated field equality, which would call two differently
# configured models equal, merge them to the first, and leave the class unhashable.
@dataclass(init=False, eq=False)
class OpenAITranscriptionModel(TranscriptionModel):
    """OpenAI transcription model implementation.

    This model works with OpenAI's audio transcriptions API and any
    [OpenAI-compatible provider](../models/openai.md#openai-compatible-models) that implements it.

    Example:
    ```python
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.transcription.openai import OpenAITranscriptionModel

    # Using OpenAI directly
    model = OpenAITranscriptionModel('gpt-transcribe')

    # Using an OpenAI-compatible provider
    model = OpenAITranscriptionModel(
        'whisper-large-v3',
        provider=OpenAIProvider(base_url='https://my-provider.com/v1', api_key='...'),
    )
    ```
    """

    _model_name: OpenAITranscriptionModelName = field(repr=False)
    _provider: Provider[AsyncOpenAI] = field(repr=False)

    def __init__(
        self,
        model_name: OpenAITranscriptionModelName,
        *,
        provider: OpenAITranscriptionCompatibleProvider | Literal['openai'] | Provider[AsyncOpenAI] = 'openai',
        settings: TranscriptionSettings | None = None,
    ):
        """Initialize an OpenAI transcription model.

        Args:
            model_name: The name of the OpenAI model to use.
                See [OpenAI's speech-to-text guide](https://developers.openai.com/api/docs/guides/speech-to-text)
                for available options.
            provider: The provider to use for authentication and API access. Can be:

                - `'openai'` (default): Uses the standard OpenAI API
                - A provider name string (e.g., `'azure'`, `'litellm'`)
                - A [`Provider`][pydantic_ai.providers.Provider] instance for custom configuration

                See [OpenAI-compatible providers](../models/openai.md#openai-compatible-models)
                for a list of supported providers.
            settings: Model-specific [`TranscriptionSettings`][pydantic_ai.transcription.TranscriptionSettings]
                to use as defaults for this model.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider(provider)
        self._provider = provider

        super().__init__(settings=settings)

    @property
    def _client(self) -> AsyncOpenAI:
        return self._provider.client

    @property
    def base_url(self) -> str:
        return str(self._client.base_url)

    @property
    def model_name(self) -> OpenAITranscriptionModelName:
        """The transcription model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The transcription model provider."""
        return self._provider.name

    async def transcribe(
        self, audio: TranscriptionInput, *, settings: TranscriptionSettings | None = None
    ) -> TranscriptionResult:
        check_allow_model_requests()
        settings = cast(OpenAITranscriptionSettings, self.prepare_transcribe(audio, settings))
        file = await _map_audio(audio)

        params = _CreateParams(
            file=file,
            model=self.model_name,
            language=settings.get('language') or OMIT,
            prompt=settings.get('openai_prompt') or OMIT,
            # `temperature` is forwarded on presence: `0` is a meaningful value, not an unset one.
            temperature=settings.get('temperature', OMIT),
            extra_headers=settings.get('extra_headers'),
            extra_body=settings.get('extra_body'),
        )
        create = self._client.audio.transcriptions.create
        response: Transcription | TranscriptionVerbose
        try:
            with map_decode_errors(self.model_name):
                # The SDK's overloads type the response by `response_format`, so each format is its own call.
                if settings.get('openai_response_format') == 'verbose_json':
                    response = await create(**params, response_format='verbose_json')
                else:
                    # The SDK parses a `json` response as `Transcription`, but its overload is typed with the
                    # union of every response shape, `TranscriptionDiarized` and `TranscriptionVerbose` included.
                    response = cast(Transcription, await create(**params, response_format='json'))
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(
                    status_code=status_code, model_name=self.model_name, body=e.body, headers=dict(e.response.headers)
                ) from e
            raise  # pragma: lax no cover
        except APIConnectionError as e:
            raise ModelAPIError(model_name=self.model_name, message=e.message) from e

        return self._map_response(check_json_response(self.model_name, response))

    def _map_response(self, response: Transcription | TranscriptionVerbose) -> TranscriptionResult:
        segments: list[TranscriptionSegment] = []
        provider_details: dict[str, Any] = {}
        if isinstance(response, TranscriptionVerbose):
            segments = [
                TranscriptionSegment(segment.text, start=segment.start, end=segment.end)
                for segment in response.segments or ()
            ]
            provider_details['language'] = response.language
            provider_details['duration'] = response.duration
        elif response.languages is not None:
            provider_details['languages'] = [language.code for language in response.languages]

        return TranscriptionResult(
            response.text,
            segments=segments,
            usage=_map_usage(response.usage),
            model_name=self.model_name,
            provider_name=self.system,
            provider_url=self.base_url,
            provider_details=provider_details or None,
        )


async def _map_audio(audio: TranscriptionInput) -> tuple[str, bytes, str]:
    """Map the audio to the `(filename, content, media type)` file tuple the SDK uploads."""
    if isinstance(audio, AudioUrl):
        downloaded = await download_item(audio, data_format='bytes')
        data, media_type = downloaded['data'], downloaded['data_type']
    elif isinstance(audio, BinaryContent):
        data, media_type = audio.data, audio.media_type
    else:
        assert_never(audio)

    extension = _FILE_EXTENSIONS.get(media_type)
    if extension is None:
        raise UserError(
            'OpenAI transcription supports FLAC, MP3, MP4, MPEG, M4A, Ogg, WAV, and WebM audio, '
            f'got media type {media_type!r}'
        )
    return f'audio.{extension}', data, media_type


class _CreateParams(TypedDict):
    """The arguments shared by every `audio.transcriptions.create` overload this model calls."""

    file: tuple[str, bytes, str]
    model: str
    language: str | Omit
    prompt: str | Omit
    temperature: float | Omit
    extra_headers: dict[str, str] | None
    extra_body: object | None


def _map_usage(
    usage: UsageTokens | UsageDuration | VerboseUsageDuration | None,
) -> RequestUsage:
    # Token-billed models (`gpt-4o-transcribe` and friends) report tokens; models billed by audio
    # duration (`whisper-1`, `gpt-transcribe`) report the seconds of audio instead. genai-prices prices both.
    if isinstance(usage, UsageTokens):
        details = usage.input_token_details
        return RequestUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            input_audio_tokens=(details.audio_tokens if details else None) or 0,
        )
    elif usage is not None:
        return RequestUsage(audio_seconds=usage.seconds)
    else:
        return RequestUsage()
