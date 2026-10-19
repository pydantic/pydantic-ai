from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, field
from datetime import datetime
from typing import Any

from genai_prices import types as genai_types

from pydantic_ai._genai_prices import calculate_price_for_usage
from pydantic_ai._utils import now_utc as _now_utc
from pydantic_ai.usage import RequestUsage


@dataclass
class TranscriptionSegment:
    """A timestamped span of a transcript."""

    text: str
    """The transcribed text of this segment."""

    _: KW_ONLY

    start: float
    """When the segment starts, in seconds from the beginning of the audio."""

    end: float
    """When the segment ends, in seconds from the beginning of the audio."""


@dataclass
class TranscriptionResult:
    """The result of a transcription operation.

    Example:
    ```python
    from pydantic_ai import BinaryContent, Transcriber

    transcriber = Transcriber('openai:gpt-transcribe')


    async def main():
        audio = BinaryContent(data=b'...', media_type='audio/ogg')
        result = await transcriber.transcribe(audio, settings={'language': 'en'})
        print(result.text)
        #> Hi, I'd like to book a table for two tonight at seven.
        print(result.model_name)
        #> gpt-transcribe
    ```
    """

    text: str
    """The full transcribed text."""

    _: KW_ONLY

    model_name: str
    """The name of the model that produced the transcript."""

    provider_name: str
    """The name of the provider (e.g. `'openai'`)."""

    segments: Sequence[TranscriptionSegment] = field(default_factory=list[TranscriptionSegment])
    """Timestamped segments of the transcript, if the provider returned them.

    Empty when the provider or the requested response format doesn't include segments; see the
    provider's settings for how to request them.
    """

    timestamp: datetime = field(default_factory=_now_utc)
    """When the transcription request was made."""

    usage: RequestUsage = field(default_factory=RequestUsage)
    """Usage statistics for this request.

    Token-billed models report `input_tokens`, `output_tokens` and `input_audio_tokens`; models billed by
    audio duration report `audio_seconds`.
    """

    provider_details: dict[str, Any] | None = None
    """Provider-specific details from the response, such as the detected language."""

    provider_response_id: str | None = None
    """Unique identifier for this response from the provider, if available."""

    provider_url: str | None = None
    """Provider API URL, if available."""

    def cost(self) -> genai_types.PriceCalculation:
        """Calculate the cost of the transcription request.

        Uses [`genai-prices`](https://github.com/pydantic/genai-prices) for pricing data, which covers
        both models billed by audio duration (like `gpt-transcribe`) and token-billed models.

        Returns:
            A price calculation object with `total_price`, `input_price`, and other cost details.

        Raises:
            LookupError: If pricing data is not available for this model/provider.
        """
        assert self.model_name, 'Model name is required to calculate price'
        return calculate_price_for_usage(
            self.usage,
            model_name=self.model_name,
            provider_api_url=self.provider_url,
            provider_name=self.provider_name,
            genai_request_timestamp=self.timestamp,
        )
