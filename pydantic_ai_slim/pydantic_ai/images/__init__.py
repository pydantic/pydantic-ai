from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from typing_extensions import TypeAliasType

from pydantic_ai import _utils
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.providers import Provider, infer_provider

from .base import ImageGenerationInput, ImageGenerationModel
from .instrumented import InstrumentedImageGenerationModel, instrument_image_generation_model
from .result import GeneratedImage, ImageGenerationResult
from .settings import (
    ImageDimensions,
    ImageGenerationAspectRatio,
    ImageGenerationSettings,
    ImageGenerationSize,
    merge_image_generation_settings,
)
from .test import TestImageGenerationModel
from .wrapper import WrapperImageGenerationModel

__all__ = [
    'GeneratedImage',
    'ImageGenerationInput',
    'ImageDimensions',
    'ImageGenerationAspectRatio',
    'ImageGenerationModel',
    'ImageGenerationResult',
    'ImageGenerationSettings',
    'ImageGenerationSize',
    'ImageGenerator',
    'InstrumentedImageGenerationModel',
    'KnownImageGenerationModelName',
    'TestImageGenerationModel',
    'WrapperImageGenerationModel',
    'infer_image_generation_model',
    'instrument_image_generation_model',
    'merge_image_generation_settings',
]

KnownImageGenerationModelName = TypeAliasType(
    'KnownImageGenerationModelName',
    Literal[
        'google:gemini-2.5-flash-image',
        'google:gemini-3-pro-image',
        'google:gemini-3.1-flash-image',
        'google:gemini-3.1-flash-lite-image',
        'openai:gpt-image-1',
        'openai:gpt-image-1-mini',
        'openai:gpt-image-1.5',
        'openai:gpt-image-2',
        'xai:grok-imagine-image',
        'xai:grok-imagine-image-quality',
    ],
)
"""Known model names that can be used with the `model` parameter of `ImageGenerator`."""


def infer_image_generation_model(
    model: ImageGenerationModel | KnownImageGenerationModelName | str,
    *,
    provider_factory: Callable[[str], Provider[Any]] = infer_provider,
) -> ImageGenerationModel:
    """Infer the image generation model from the name."""
    if isinstance(model, ImageGenerationModel):
        return model

    try:
        provider_name, model_name = model.split(':', maxsplit=1)
    except ValueError as e:
        raise ValueError('You must provide a provider prefix when specifying an image generation model name') from e

    provider = provider_factory(provider_name)

    if provider_name == 'openai':
        from .openai import OpenAIImageGenerationModel

        return OpenAIImageGenerationModel(model_name, provider=provider)
    elif provider_name == 'google':
        from .google import GoogleImageGenerationModel

        return GoogleImageGenerationModel(model_name, provider=provider)
    elif provider_name == 'xai':
        from .xai import XaiImageGenerationModel

        return XaiImageGenerationModel(model_name, provider=provider)
    else:
        raise UserError(f'Unknown image generation model: {model}')  # pragma: no cover


@dataclass(init=False)
class ImageGenerator:
    """High-level interface for generating images."""

    instrument: InstrumentationSettings | bool | None
    """Options to automatically instrument with OpenTelemetry.

    Set to `True` to use default instrumentation settings, which will use Logfire if it's configured.
    Set to an instance of [`InstrumentationSettings`][pydantic_ai.models.instrumented.InstrumentationSettings]
    to customize. If this isn't set, then the last value set by
    [`ImageGenerator.instrument_all()`][pydantic_ai.images.ImageGenerator.instrument_all]
    will be used, which defaults to False.
    """

    _instrument_default: ClassVar[InstrumentationSettings | bool] = False

    def __init__(
        self,
        model: ImageGenerationModel | KnownImageGenerationModelName | str,
        *,
        settings: ImageGenerationSettings | None = None,
        defer_model_check: bool = True,
        instrument: InstrumentationSettings | bool | None = None,
    ) -> None:
        self._model = model if defer_model_check else infer_image_generation_model(model)
        self._settings = settings
        self.instrument = instrument

        self._override_model: ContextVar[ImageGenerationModel | None] = ContextVar('_override_model', default=None)

    @staticmethod
    def instrument_all(instrument: InstrumentationSettings | bool = True) -> None:
        """Set the default instrumentation options for all image generators."""
        ImageGenerator._instrument_default = instrument

    @property
    def model(self) -> ImageGenerationModel | KnownImageGenerationModelName | str:
        """The image generation model used by this generator."""
        return self._model

    @contextmanager
    def override(
        self,
        *,
        model: ImageGenerationModel | KnownImageGenerationModelName | str | _utils.Unset = _utils.UNSET,
    ) -> Generator[None]:
        """Context manager to temporarily override the image generation model."""
        if _utils.is_set(model):
            model_token = self._override_model.set(infer_image_generation_model(model))
        else:
            model_token = None

        try:
            yield
        finally:
            if model_token is not None:
                self._override_model.reset(model_token)

    async def generate(
        self,
        prompt: str,
        *,
        images: Sequence[ImageGenerationInput] | None = None,
        settings: ImageGenerationSettings | None = None,
    ) -> ImageGenerationResult:
        """Generate images from a prompt and optional reference images."""
        model = self._get_model()
        settings = merge_image_generation_settings(self._settings, settings)
        return await model.generate(prompt, images=images, settings=settings)

    def generate_sync(
        self,
        prompt: str,
        *,
        images: Sequence[ImageGenerationInput] | None = None,
        settings: ImageGenerationSettings | None = None,
    ) -> ImageGenerationResult:
        """Synchronous version of [`generate()`][pydantic_ai.images.ImageGenerator.generate]."""
        return _utils.run_until_complete(self.generate(prompt, images=images, settings=settings))

    def _get_model(self) -> ImageGenerationModel:
        """Create a model configured for this generator."""
        model_: ImageGenerationModel
        if some_model := self._override_model.get():
            model_ = some_model
        else:
            model_ = self._model = infer_image_generation_model(self.model)

        instrument = self.instrument
        if instrument is None:
            instrument = self._instrument_default

        return instrument_image_generation_model(model_, instrument)
