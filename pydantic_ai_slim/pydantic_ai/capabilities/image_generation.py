from __future__ import annotations

import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.images import (
    ImageDimensions,
    ImageGenerationAspectRatio,
    ImageGenerationModel,
    ImageGenerationSettings,
    ImageGenerationSize,
    ImageGenerator,
)
from pydantic_ai.images.settings import image_generation_tool_settings
from pydantic_ai.messages import BinaryImage
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.native_tools import ImageGenerationModelName, ImageGenerationTool
from pydantic_ai.tools import AgentDepsT, RunContext, Tool
from pydantic_ai.toolsets import AbstractToolset

from .native_or_local import NativeOrLocalTool

if TYPE_CHECKING:
    from pydantic_ai.common_tools.image_generation import ImageGenerationFallbackModel


@dataclass(kw_only=True)
class _DirectImageGenerationTool:
    """Local capability tool backed directly by the image generation API."""

    generator: ImageGenerator | ImageGenerationModel
    settings: ImageGenerationSettings
    action: Literal['generate', 'edit', 'auto'] | None
    image_model: ImageGenerationModelName | None

    async def __call__(self, prompt: str) -> BinaryImage:
        if self.action == 'edit':
            raise UserError(
                'The direct `ImageGeneration` fallback cannot honor `action="edit"` because the '
                '`generate_image` tool does not receive reference images. Use '
                '`ImageGenerator.generate(..., images=...)` directly for image editing.'
            )
        if self.image_model is not None:
            warnings.warn(
                'Direct `ImageGeneration` fallback ignored `image_model`; `local` already selects the direct image model',
                UserWarning,
                stacklevel=2,
            )

        result = await self.generator.generate(prompt, settings=self.settings)
        if len(result.images) != 1:
            raise UnexpectedModelBehavior(
                f'Direct image generation fallback returned {len(result.images)} images; expected exactly one'
            )
        return result.images[0].content


def _direct_image_generation_tool(
    generator: ImageGenerator | ImageGenerationModel,
    *,
    settings: ImageGenerationSettings,
    action: Literal['generate', 'edit', 'auto'] | None,
    image_model: ImageGenerationModelName | None,
) -> Tool[Any]:
    return Tool[Any](
        _DirectImageGenerationTool(
            generator=generator,
            settings=settings,
            action=action,
            image_model=image_model,
        ).__call__,
        name='generate_image',
        description='Generate an image based on the given prompt.',
    )


@dataclass(init=False)
class ImageGeneration(NativeOrLocalTool[AgentDepsT]):
    """Image generation capability.

    Uses the model's native image generation when available. When the model doesn't
    support it, pass an `ImageGenerator` or `ImageGenerationModel` to `local` to use
    the direct image generation API as a fallback.

    The existing `fallback_model` path delegates to a subagent running an
    image-capable conversational model and is preserved for backwards compatibility.

    Image generation settings are applied to the direct fallback using
    `ImageGenerationSettings`. Each direct provider maps the normalized settings it
    supports and warns about settings it cannot apply. The native path and the legacy
    `fallback_model` subagent retain the existing `ImageGenerationTool` surface; direct-only
    geometry settings are ignored there with a migration warning.
    """

    fallback_model: ImageGenerationFallbackModel
    """Model to use for image generation when the agent's model doesn't support it natively.

    Must be a model that supports image generation via the
    [`ImageGenerationTool`][pydantic_ai.native_tools.ImageGenerationTool] native tool.
    This requires a conversational model with image generation support, not a dedicated
    image-only API. Examples:

    * `'openai-responses:gpt-5.4'` — OpenAI model with image generation support
    * `'google:gemini-3-pro-image-preview'` — Google image generation model

    Can be a model name string, `Model` instance, or a callable taking `RunContext`
    that returns a `Model` instance or model name string.
    """

    # Keep these fields in sync with ImageGenerationTool in native_tools.py.

    action: Literal['generate', 'edit', 'auto'] | None
    """Whether to generate a new image or edit an existing image.

    Supported by: OpenAI Responses. Default: `'auto'`.
    """

    background: Literal['transparent', 'opaque', 'auto'] | None
    """Background type for the generated image.

    Supported by: OpenAI Responses. `'transparent'` only supported for `'png'` and `'webp'`.
    """

    input_fidelity: Literal['high', 'low'] | None
    """Input fidelity for matching style/features of input images.

    Supported by: OpenAI Responses. Default: `'low'`.
    """

    moderation: Literal['auto', 'low'] | None
    """Moderation level for the generated image.

    Supported by: OpenAI Responses.
    """

    image_model: ImageGenerationModelName | None
    """The image generation model to use.

    Supported by: OpenAI Responses.
    """

    output_compression: int | None
    """Compression level for the output image.

    Supported by: OpenAI Responses (jpeg/webp, default: 100), Google Cloud (jpeg, default: 75).
    """

    output_format: Literal['png', 'webp', 'jpeg'] | None
    """Output format of the generated image.

    Supported by: OpenAI Responses (default: `'png'`), Google Cloud.
    """

    quality: Literal['low', 'medium', 'high', 'auto'] | None
    """Quality of the generated image.

    Supported by: OpenAI Responses.
    """

    size: ImageGenerationSize | None
    """Size of the generated image.

    For direct image models, this is a provider-dependent compatibility setting: OpenAI uses
    pixel-size strings, while Google and xAI use resolution tiers. Prefer `dimensions` for exact
    cross-provider pixel semantics. Direct-only values are ignored by the legacy path with a warning.
    """

    dimensions: ImageDimensions | None
    """Exact direct-model output dimensions as `(width, height)` in pixels.

    This is mutually exclusive with `aspect_ratio` and `size`. The legacy native/fallback-model
    path ignores it with a warning.
    """

    aspect_ratio: ImageGenerationAspectRatio | None
    """Aspect ratio for generated images.

    Direct adapters map this to a canonical geometry supported by the selected model. Ratios outside
    the existing native-tool vocabulary are ignored by the legacy path with a warning.
    """

    def __init__(
        self,
        *,
        native: ImageGenerationTool
        | Callable[[RunContext[AgentDepsT]], Awaitable[ImageGenerationTool | None] | ImageGenerationTool | None]
        | bool = True,
        local: Tool[AgentDepsT]
        | Callable[..., Any]
        | ImageGenerator
        | ImageGenerationModel
        | str
        | Literal[False]
        | None = None,
        fallback_model: Model
        | KnownModelName
        | str
        | Callable[[RunContext[AgentDepsT]], Awaitable[Model | KnownModelName | str] | Model | KnownModelName | str]
        | None = None,
        action: Literal['generate', 'edit', 'auto'] | None = None,
        background: Literal['transparent', 'opaque', 'auto'] | None = None,
        input_fidelity: Literal['high', 'low'] | None = None,
        moderation: Literal['auto', 'low'] | None = None,
        image_model: ImageGenerationModelName | None = None,
        output_compression: int | None = None,
        output_format: Literal['png', 'webp', 'jpeg'] | None = None,
        quality: Literal['low', 'medium', 'high', 'auto'] | None = None,
        size: ImageGenerationSize | None = None,
        dimensions: ImageDimensions | None = None,
        aspect_ratio: ImageGenerationAspectRatio | None = None,
        id: str | None = None,
        defer_loading: bool = False,
        description: str | None = None,
    ) -> None:
        self.id = id
        self.description = description
        self.defer_loading = defer_loading
        if fallback_model is not None and local is not None:
            raise UserError(
                'ImageGeneration: cannot specify both `fallback_model` and `local` — '
                'use `fallback_model` for the default subagent fallback, or `local` for a custom tool'
            )
        self.native = native
        self.fallback_model = fallback_model
        self.action = action
        self.background = background
        self.input_fidelity = input_fidelity
        self.moderation = moderation
        self.image_model = image_model
        self.output_compression = output_compression
        self.output_format = output_format
        self.quality = quality
        self.size = size
        self.dimensions = dimensions
        self.aspect_ratio = aspect_ratio
        if isinstance(local, (ImageGenerator, ImageGenerationModel)):
            local = self._direct_local_tool(local)
        self.local = local
        self.__post_init__()

    @classmethod
    def from_spec(
        cls,
        *,
        native: ImageGenerationTool | bool = True,
        local: str | Literal[False] | None = None,
        fallback_model: KnownModelName | str | None = None,
        action: Literal['generate', 'edit', 'auto'] | None = None,
        background: Literal['transparent', 'opaque', 'auto'] | None = None,
        input_fidelity: Literal['high', 'low'] | None = None,
        moderation: Literal['auto', 'low'] | None = None,
        image_model: ImageGenerationModelName | None = None,
        output_compression: int | None = None,
        output_format: Literal['png', 'webp', 'jpeg'] | None = None,
        quality: Literal['low', 'medium', 'high', 'auto'] | None = None,
        size: ImageGenerationSize | None = None,
        dimensions: ImageDimensions | None = None,
        aspect_ratio: ImageGenerationAspectRatio | None = None,
        id: str | None = None,
        defer_loading: bool = False,
        description: str | None = None,
    ) -> ImageGeneration[AgentDepsT]:
        """Construct from the JSON/YAML-serializable subset of the runtime API.

        Runtime objects accepted by `local`, such as `ImageGenerator`, `ImageGenerationModel`,
        `Tool`, and callables, can be passed to `ImageGeneration(...)` directly but cannot be
        represented in an agent spec. A direct image model name is serializable and can be passed
        as `local='provider:model'`.
        """
        return cls(
            native=native,
            local=local,
            fallback_model=fallback_model,
            action=action,
            background=background,
            input_fidelity=input_fidelity,
            moderation=moderation,
            image_model=image_model,
            output_compression=output_compression,
            output_format=output_format,
            quality=quality,
            size=size,
            dimensions=dimensions,
            aspect_ratio=aspect_ratio,
            id=id,
            defer_loading=defer_loading,
            description=description,
        )

    def _image_settings(self) -> ImageGenerationSettings:
        """Collect normalized settings shared by native and direct image generation."""
        settings: ImageGenerationSettings = {}
        if self.background is not None:
            settings['background'] = self.background
        if self.input_fidelity is not None:
            settings['input_fidelity'] = self.input_fidelity
        if self.moderation is not None:
            settings['moderation'] = self.moderation
        if self.output_compression is not None:
            settings['output_compression'] = self.output_compression
        if self.output_format is not None:
            settings['output_format'] = self.output_format
        if self.quality is not None:
            settings['quality'] = self.quality
        if self.size is not None:
            settings['size'] = self.size
        if self.dimensions is not None:
            settings['dimensions'] = self.dimensions
        if self.aspect_ratio is not None:
            settings['aspect_ratio'] = self.aspect_ratio
        return settings

    def _image_gen_kwargs(self) -> dict[str, Any]:
        """Collect settings supported by the legacy `ImageGenerationTool` path."""
        settings, ignored = image_generation_tool_settings(self._image_settings())
        if ignored:
            warnings.warn(
                'The legacy `ImageGeneration` native/fallback_model path ignored direct-only '
                f'setting(s): {", ".join(ignored)}. Use `native=False` with '
                "`local='provider:image-model'` or an `ImageGenerator` to apply them.",
                UserWarning,
                stacklevel=3,
            )

        kwargs: dict[str, Any] = dict(settings)
        if self.action is not None:
            kwargs['action'] = self.action
        if self.image_model is not None:
            kwargs['model'] = self.image_model
        return kwargs

    def _default_native(self) -> ImageGenerationTool:
        return ImageGenerationTool(**self._image_gen_kwargs())

    def _resolve_local_strategy(self, name: str | bool) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT]:
        if isinstance(name, str):
            return self._direct_local_tool(ImageGenerator(name))
        return super()._resolve_local_strategy(name)

    def _direct_local_tool(self, generator: ImageGenerator | ImageGenerationModel) -> Tool[Any]:
        return _direct_image_generation_tool(
            generator,
            settings={'n': 1, **self._image_settings()},
            action=self.action,
            image_model=self.image_model,
        )

    def _native_unique_id(self) -> str:
        return ImageGenerationTool.kind

    def _resolved_native(self) -> ImageGenerationTool:
        """Get the ImageGenerationTool for the fallback, with capability-level overrides applied."""
        base = self.native if isinstance(self.native, ImageGenerationTool) else ImageGenerationTool()
        overrides = self._image_gen_kwargs()
        if not overrides:
            return base
        return replace(base, **overrides)

    def _default_local(self) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT] | None:
        if self.fallback_model is None:
            return None
        from pydantic_ai.common_tools.image_generation import image_generation_tool

        return image_generation_tool(model=self.fallback_model, native_tool=self._resolved_native())
