from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai.builtin_tools import ImageAspectRatio, ImageGenerationTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model
from pydantic_ai.tools import AgentDepsT, RunContext, Tool
from pydantic_ai.toolsets import AbstractToolset

from .builtin_or_local import BuiltinOrLocalTool

if TYPE_CHECKING:
    from pydantic_ai.common_tools.image_generation import FallbackModel


@dataclass(init=False)
class ImageGeneration(BuiltinOrLocalTool[AgentDepsT]):
    """Image generation capability.

    Uses the model's builtin image generation when available. When the model doesn't
    support it and ``fallback_model`` is provided, falls back to a local tool that
    delegates to a subagent running the specified image-capable model.

    Image generation settings (``quality``, ``size``, etc.) are forwarded to the
    ``ImageGenerationTool`` used by both the builtin and the local fallback subagent.
    When passing a custom ``builtin`` instance, its settings are also used for the
    fallback subagent; capability-level fields override any ``builtin`` instance settings.
    """

    fallback_model: FallbackModel
    """Model to use for image generation when the agent's model doesn't support it natively.

    Must be a model that supports image generation, e.g.:

    * ``'openai-responses:gpt-image-1'`` — OpenAI's dedicated image generation model
    * ``'openai-responses:gpt-4o'`` — OpenAI model with image generation support
    * ``'google-gla:gemini-2.0-flash-preview-image-generation'`` — Google image generation model

    Can be a model name string, ``Model`` instance, or a callable taking ``RunContext``
    that returns a model.
    """

    background: Literal['transparent', 'opaque', 'auto'] | None
    """Background type for the generated image. Forwarded to ``ImageGenerationTool``."""

    input_fidelity: Literal['high', 'low'] | None
    """Input fidelity for matching style/features of input images. Forwarded to ``ImageGenerationTool``."""

    moderation: Literal['auto', 'low'] | None
    """Moderation level for the generated image. Forwarded to ``ImageGenerationTool``."""

    output_compression: int | None
    """Compression level for the output image. Forwarded to ``ImageGenerationTool``."""

    output_format: Literal['png', 'webp', 'jpeg'] | None
    """Output format of the generated image. Forwarded to ``ImageGenerationTool``."""

    partial_images: int | None
    """Number of partial images to generate in streaming mode. Forwarded to ``ImageGenerationTool``."""

    quality: Literal['low', 'medium', 'high', 'auto'] | None
    """Quality of the generated image. Forwarded to ``ImageGenerationTool``."""

    size: Literal['auto', '1024x1024', '1024x1536', '1536x1024', '512', '1K', '2K', '4K'] | None
    """Size of the generated image. Forwarded to ``ImageGenerationTool``."""

    aspect_ratio: ImageAspectRatio | None
    """Aspect ratio to use for generated images. Forwarded to ``ImageGenerationTool``."""

    def __init__(
        self,
        *,
        builtin: ImageGenerationTool
        | Callable[[RunContext[AgentDepsT]], Awaitable[ImageGenerationTool | None] | ImageGenerationTool | None]
        | bool = True,
        local: Tool[AgentDepsT] | Callable[..., Any] | Literal[False] | None = None,
        fallback_model: Model
        | str
        | Callable[[RunContext[AgentDepsT]], Awaitable[Model | str | None] | Model | str | None]
        | None = None,
        background: Literal['transparent', 'opaque', 'auto'] | None = None,
        input_fidelity: Literal['high', 'low'] | None = None,
        moderation: Literal['auto', 'low'] | None = None,
        output_compression: int | None = None,
        output_format: Literal['png', 'webp', 'jpeg'] | None = None,
        partial_images: int | None = None,
        quality: Literal['low', 'medium', 'high', 'auto'] | None = None,
        size: Literal['auto', '1024x1024', '1024x1536', '1536x1024', '512', '1K', '2K', '4K'] | None = None,
        aspect_ratio: ImageAspectRatio | None = None,
    ) -> None:
        if fallback_model is not None and local is not None:
            raise UserError(
                'ImageGeneration: cannot specify both `fallback_model` and `local` — '
                'use `fallback_model` for the default subagent fallback, or `local` for a custom tool'
            )
        self.builtin = builtin
        self.local = local
        self.fallback_model = fallback_model
        self.background = background
        self.input_fidelity = input_fidelity
        self.moderation = moderation
        self.output_compression = output_compression
        self.output_format = output_format
        self.partial_images = partial_images
        self.quality = quality
        self.size = size
        self.aspect_ratio = aspect_ratio
        self.__post_init__()

    _IMAGE_GEN_FIELDS = (
        'background',
        'input_fidelity',
        'moderation',
        'output_compression',
        'output_format',
        'partial_images',
        'quality',
        'size',
        'aspect_ratio',
    )

    def _image_gen_kwargs(self) -> dict[str, Any]:
        """Collect non-None ImageGenerationTool config fields."""
        return {f: v for f in self._IMAGE_GEN_FIELDS if (v := getattr(self, f)) is not None}

    def _default_builtin(self) -> ImageGenerationTool:
        return ImageGenerationTool(**self._image_gen_kwargs())

    def _builtin_unique_id(self) -> str:
        return ImageGenerationTool.kind

    def _resolved_builtin(self) -> ImageGenerationTool:
        """Get the ImageGenerationTool for the fallback, with capability-level overrides applied."""
        base = self.builtin if isinstance(self.builtin, ImageGenerationTool) else ImageGenerationTool()
        overrides = self._image_gen_kwargs()
        if not overrides:
            return base
        return replace(base, **overrides)

    def _default_local(self) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT] | None:
        if self.fallback_model is None:
            return None
        from pydantic_ai.common_tools.image_generation import image_generation_tool

        return image_generation_tool(model=self.fallback_model, builtin=self._resolved_builtin())
