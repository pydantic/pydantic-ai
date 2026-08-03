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
        raise UserError(f'Provider {provider_name!r} does not support direct image generation.')


@dataclass(init=False)
class ImageGenerator:
    """High-level interface for generating images.

    The `ImageGenerator` class provides a convenient way to generate images from a prompt, and to edit
    or transform reference images, using dedicated image models. It handles model inference, settings
    management, and optional OpenTelemetry instrumentation.

    Example:
    ```python
    from pydantic_ai import ImageGenerator

    generator = ImageGenerator('openai:gpt-image-2')


    async def main():
        result = await generator.generate('A watercolor map of a floating city.')
        print(result.images[0].content.media_type)
        #> image/png
    ```
    """

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
        """Initialize an ImageGenerator.

        Args:
            model: The image generation model to use. Can be specified as:

                - A model name string in the format `'provider:model-name'`
                  (e.g., `'openai:gpt-image-2'`)
                - An [`ImageGenerationModel`][pydantic_ai.images.ImageGenerationModel] instance
            settings: Optional [`ImageGenerationSettings`][pydantic_ai.images.ImageGenerationSettings]
                to use as defaults for all generate calls.
            defer_model_check: Whether to defer resolving the model name to a model instance, and the
                provider authentication that resolution requires, until the first generate call.
                Set to `False` to resolve the model immediately on construction.
            instrument: OpenTelemetry instrumentation settings. Set to `True` to enable with defaults,
                or pass an [`InstrumentationSettings`][pydantic_ai.models.instrumented.InstrumentationSettings]
                instance to customize. If `None`, uses the value from
                [`ImageGenerator.instrument_all()`][pydantic_ai.images.ImageGenerator.instrument_all].
        """
        self._model = model if defer_model_check else infer_image_generation_model(model)
        self._settings = settings
        self.instrument = instrument

        self._override_model: ContextVar[ImageGenerationModel | None] = ContextVar('_override_model', default=None)

    @staticmethod
    def instrument_all(instrument: InstrumentationSettings | bool = True) -> None:
        """Set the default instrumentation options for all image generators where `instrument` is not explicitly set.

        This is useful for enabling instrumentation globally without modifying each generator individually.

        Args:
            instrument: Instrumentation settings to use as the default. Set to `True` for default settings,
                `False` to disable, or pass an
                [`InstrumentationSettings`][pydantic_ai.models.instrumented.InstrumentationSettings]
                instance to customize.
        """
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
        """Context manager to temporarily override the image generation model.

        Useful for testing or dynamically switching models.

        Args:
            model: The image generation model to use within this context.

        Example:
        ```python
        from pydantic_ai import ImageGenerator

        generator = ImageGenerator('openai:gpt-image-2')


        async def main():
            # Temporarily use a different model
            with generator.override(model='google:gemini-3.1-flash-image'):
                result = await generator.generate('A watercolor map of a floating city.')
                print(result.model_name)
                #> gemini-3.1-flash-image
        ```
        """
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
        """Generate images from a prompt and optional reference images.

        Args:
            prompt: The text prompt describing the image to generate.
            images: Optional reference images to edit or transform. Passing reference images sends the
                request to the provider's image-editing path, preserving the order of the images. Each
                item can be a [`BinaryImage`][pydantic_ai.messages.BinaryImage],
                [`ImageUrl`][pydantic_ai.messages.ImageUrl], or
                [`UploadedFile`][pydantic_ai.messages.UploadedFile]; see the
                [Image Generation guide](../image-generation.md#editing-images) for the reference-input
                types each provider accepts.
            settings: Optional settings to override the generator's default settings for this call.

        Returns:
            An [`ImageGenerationResult`][pydantic_ai.images.ImageGenerationResult] containing the
            generated images and metadata about the operation.

        Raises:
            ContentFilterError: If the provider blocked the request, or every generated image, for
                content moderation.
            UserError: If the prompt is empty, a setting is invalid, or the model cannot produce the
                requested dimensions.
        """
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
        """Synchronous version of [`generate()`][pydantic_ai.images.ImageGenerator.generate].

        Args:
            prompt: The text prompt describing the image to generate.
            images: Optional reference images to edit or transform.
            settings: Optional settings to override the generator's default settings for this call.

        Returns:
            An [`ImageGenerationResult`][pydantic_ai.images.ImageGenerationResult] containing the
            generated images and metadata about the operation.

        Raises:
            ContentFilterError: If the provider blocked the request, or every generated image, for
                content moderation.
            UserError: If the prompt is empty, a setting is invalid, or the model cannot produce the
                requested dimensions.
        """
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
