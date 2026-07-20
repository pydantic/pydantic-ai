from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, field
from datetime import datetime

from genai_prices import types as genai_types

from pydantic_ai._utils import now_utc as _now_utc
from pydantic_ai.messages import BinaryImage
from pydantic_ai.usage import RequestUsage

from .settings import ImageGenerationSettings


@dataclass
class GeneratedImage:
    """One generated image with normalized content and provider metadata."""

    content: BinaryImage
    """The generated image as normalized binary content."""

    _: KW_ONLY

    revised_prompt: str | None = None
    """Provider-revised or enhanced prompt, if available."""

    size: str | None = None
    """Generated image size, if reported by the provider."""

    quality: str | None = None
    """Generated image quality, if reported by the provider."""

    output_format: str | None = None
    """Generated image output format, if reported by the provider."""

    background: str | None = None
    """Generated image background mode, if reported by the provider."""

    provider_details: dict[str, object] | None = None
    """Provider-specific details for this generated image."""

    provider_image_id: str | None = None
    """Provider-specific identifier for this image, if available."""


@dataclass
class ImageGenerationResult:
    """The result of an image generation operation."""

    images: Sequence[GeneratedImage]
    """Generated images."""

    _: KW_ONLY

    prompt: str
    """The input prompt used for generation."""

    model_name: str
    """The name of the model that generated the images."""

    provider_name: str
    """The name of the provider."""

    timestamp: datetime = field(default_factory=_now_utc)
    """When the image generation request was made."""

    usage: RequestUsage = field(default_factory=RequestUsage)
    """Usage statistics for this request."""

    settings: ImageGenerationSettings | None = None
    """The normalized settings used for this request, if available."""

    provider_details: dict[str, object] | None = None
    """Provider-specific details from the response."""

    provider_response_id: str | None = None
    """Unique identifier for this response from the provider, if available."""

    provider_url: str | None = None
    """Provider API URL, if available."""

    def cost(self) -> genai_types.PriceCalculation:
        """Calculate the cost of the image generation request when image pricing is supported.

        Image generation pricing is temporarily unavailable while [`genai-prices`](https://github.com/pydantic/genai-prices)
        adds support for image-token and per-image pricing.

        Raises:
            LookupError: Until image generation pricing is supported.
        """
        # TODO: Enable image generation pricing once the data-driven unit registry lands and
        # genai-prices supports image-token and per-image pricing:
        # https://github.com/pydantic/genai-prices/pull/351
        # https://github.com/pydantic/genai-prices/issues/185
        # https://github.com/pydantic/genai-prices/issues/410
        raise LookupError('`ImageGenerationResult.cost()` is unavailable until `genai-prices` supports image pricing')
