from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .base import ImageGenerationInput, ImageGenerationModel
from .result import ImageGenerationResult
from .settings import ImageGenerationSettings


@dataclass(init=False)
class WrapperImageGenerationModel(ImageGenerationModel):
    """Base class for image generation models that wrap another model.

    Use this as a base class to create custom image generation model wrappers
    that modify behavior (e.g., caching, logging, rate limiting) while
    delegating to an underlying model.

    By default, all methods are passed through to the wrapped model.
    Override specific methods to customize behavior.
    """

    wrapped: ImageGenerationModel
    """The underlying image generation model being wrapped."""

    def __init__(self, wrapped: ImageGenerationModel | str):
        """Initialize the wrapper with an image generation model.

        Args:
            wrapped: The model to wrap. Can be an
                [`ImageGenerationModel`][pydantic_ai.images.ImageGenerationModel] instance
                or a model name string (e.g., `'openai:gpt-image-1'`).
        """
        from . import infer_image_generation_model

        super().__init__()
        self.wrapped = infer_image_generation_model(wrapped) if isinstance(wrapped, str) else wrapped

    async def generate(
        self,
        prompt: str,
        *,
        images: Sequence[ImageGenerationInput] | None = None,
        settings: ImageGenerationSettings | None = None,
    ) -> ImageGenerationResult:
        return await self.wrapped.generate(prompt, images=images, settings=settings)

    @property
    def model_name(self) -> str:
        return self.wrapped.model_name

    @property
    def system(self) -> str:
        return self.wrapped.system

    @property
    def settings(self) -> ImageGenerationSettings | None:
        """Get the settings from the wrapped image generation model."""
        return self.wrapped.settings

    @property
    def base_url(self) -> str | None:
        return self.wrapped.base_url

    def __getattr__(self, item: str):
        return getattr(self.wrapped, item)  # pragma: no cover
