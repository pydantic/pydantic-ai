from abc import ABC, abstractmethod

from .result import ImageGenerationResult
from .settings import ImageGenerationSettings, merge_image_generation_settings


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
    async def generate(self, prompt: str, *, settings: ImageGenerationSettings | None = None) -> ImageGenerationResult:
        """Generate images for the given prompt."""
        raise NotImplementedError

    def prepare_generate(
        self, prompt: str, settings: ImageGenerationSettings | None = None
    ) -> tuple[str, ImageGenerationSettings]:
        """Prepare the prompt and settings for image generation."""
        settings = merge_image_generation_settings(self._settings, settings) or {}
        return prompt, settings
