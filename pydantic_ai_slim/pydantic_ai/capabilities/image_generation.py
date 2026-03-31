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
    support it and `fallback_model` is provided, falls back to a local tool that
    delegates to a subagent running the specified image-capable model.

    Image generation settings (`quality`, `size`, etc.) are forwarded to the
    `ImageGenerationTool` used by both the builtin and the local fallback subagent.
    When passing a custom `builtin` instance, its settings are also used for the
    fallback subagent; capability-level fields override any `builtin` instance settings.
    """

    fallback_model: FallbackModel
    """Model to use for image generation when the agent's model doesn't support it natively.

    Must be a model that supports image generation via the `ImageGenerationTool` builtin.
    This requires a conversational model with image generation support, not a dedicated
    image-only API. Examples:

    * `'openai-responses:gpt-5.4'` — OpenAI model with image generation support
    * `'google-gla:gemini-3-pro-image-preview'` — Google image generation model

    Can be a model name string, `Model` instance, or a callable taking `RunContext`
    that returns a model.
    """

    _image_gen_config: ImageGenerationTool
    """The `ImageGenerationTool` config built from the capability-level fields."""

    _image_gen_overrides: dict[str, Any]
    """The non-default image generation kwargs, used for merging with a custom `builtin`."""

    def __init__(
        self,
        *,
        builtin: ImageGenerationTool
        | Callable[[RunContext[AgentDepsT]], Awaitable[ImageGenerationTool | None] | ImageGenerationTool | None]
        | bool = True,
        local: Tool[AgentDepsT] | Callable[..., Any] | Literal[False] | None = None,
        fallback_model: Model
        | str
        | Callable[[RunContext[AgentDepsT]], Awaitable[Model | str] | Model | str]
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

        # Build ImageGenerationTool directly from the init params
        kwargs: dict[str, Any] = {}
        if background is not None:
            kwargs['background'] = background
        if input_fidelity is not None:
            kwargs['input_fidelity'] = input_fidelity
        if moderation is not None:
            kwargs['moderation'] = moderation
        if output_compression is not None:
            kwargs['output_compression'] = output_compression
        if output_format is not None:
            kwargs['output_format'] = output_format
        if partial_images is not None:
            kwargs['partial_images'] = partial_images
        if quality is not None:
            kwargs['quality'] = quality
        if size is not None:
            kwargs['size'] = size
        if aspect_ratio is not None:
            kwargs['aspect_ratio'] = aspect_ratio
        self._image_gen_config = ImageGenerationTool(**kwargs)
        self._image_gen_overrides = kwargs

        self.__post_init__()

    def _default_builtin(self) -> ImageGenerationTool:
        return self._image_gen_config

    def _builtin_unique_id(self) -> str:
        return ImageGenerationTool.kind

    def _resolved_builtin(self) -> ImageGenerationTool:
        """Get the ImageGenerationTool for the fallback, with capability-level overrides applied."""
        base = self.builtin if isinstance(self.builtin, ImageGenerationTool) else ImageGenerationTool()
        if not self._image_gen_overrides:
            return base
        return replace(base, **self._image_gen_overrides)

    def _default_local(self) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT] | None:
        if self.fallback_model is None:
            return None
        from pydantic_ai.common_tools.image_generation import image_generation_tool

        return image_generation_tool(model=self.fallback_model, builtin_tool=self._resolved_builtin())
