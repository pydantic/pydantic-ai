from abc import ABC, abstractmethod
from collections.abc import Sequence

from typing_extensions import TypeAliasType

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import BinaryImage, ImageUrl, UploadedFile

from .result import ImageGenerationResult
from .settings import ImageGenerationSettings, merge_image_generation_settings, validate_image_generation_settings

ImageGenerationInput = TypeAliasType('ImageGenerationInput', ImageUrl | BinaryImage | UploadedFile)
"""An image input that can be used as a reference for image generation."""


class ImageGenerationModel(ABC):
    """Abstract base class for image generation models."""

    _settings: ImageGenerationSettings | None = None

    def __init__(
        self,
        *,
        settings: ImageGenerationSettings | None = None,
    ) -> None:
        self._settings = settings

    @property
    def settings(self) -> ImageGenerationSettings | None:
        """Get the default settings for this model."""
        return self._settings

    @property
    def base_url(self) -> str | None:
        """The base URL for the provider API, if available."""
        return None

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The name of the image generation model."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def system(self) -> str:
        """The image generation model provider/system identifier."""
        raise NotImplementedError()

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        images: Sequence[ImageGenerationInput] | None = None,
        settings: ImageGenerationSettings | None = None,
    ) -> ImageGenerationResult:
        """Generate images for the given prompt."""
        raise NotImplementedError

    def prepare_generate(
        self,
        prompt: str,
        *,
        images: Sequence[ImageGenerationInput] | None = None,
        settings: ImageGenerationSettings | None = None,
    ) -> tuple[str, list[ImageGenerationInput], ImageGenerationSettings]:
        """Prepare the prompt, reference images, and settings for image generation."""
        if not prompt.strip():
            raise UserError('Image generation requires a non-empty prompt')

        prepared_images = list(images or ())
        for image in prepared_images:
            if not isinstance(image, (ImageUrl, BinaryImage, UploadedFile)):
                raise UserError(
                    'Image generation inputs must be `ImageUrl`, `BinaryImage`, or `UploadedFile` instances'
                )
            if isinstance(image, UploadedFile) and not image.media_type.startswith('image/'):
                raise UserError('Image generation `UploadedFile` inputs must have an image media type')

        settings = merge_image_generation_settings(self._settings, settings) or {}
        validate_image_generation_settings(settings)
        return prompt, prepared_images, settings
