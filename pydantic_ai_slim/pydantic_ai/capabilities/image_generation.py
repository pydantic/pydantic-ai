from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai.builtin_tools import ImageAspectRatio, ImageGenerationModelName, ImageGenerationTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.tools import AgentDepsT, RunContext, Tool
from pydantic_ai.toolsets import AbstractToolset

from .builtin_or_local import BuiltinOrLocalTool

if TYPE_CHECKING:
    from pydantic_ai.common_tools.image_generation import ImageGenerationFallbackModel


@dataclass(init=False)
class ImageGeneration(BuiltinOrLocalTool[AgentDepsT]):
    """Image generation capability.

    Uses the model's builtin image generation when available. When the model doesn't
    support it and `fallback_model` is provided, falls back to a local tool that
    delegates to a subagent running the specified image-capable model.

    Image generation settings (`quality`, `size`, etc.) are forwarded to the
    [`ImageGenerationTool`][pydantic_ai.builtin_tools.ImageGenerationTool] used by
    both the builtin and the local fallback subagent. When passing a custom `builtin`
    instance, its settings are also used for the fallback subagent; capability-level
    fields override any `builtin` instance settings.
    """

    fallback_model: ImageGenerationFallbackModel
    """Model to use for image generation when the agent's model doesn't support it natively.

    Must be a model that supports image generation via the
    [`ImageGenerationTool`][pydantic_ai.builtin_tools.ImageGenerationTool] builtin.
    This requires a conversational model with image generation support, not a dedicated
    image-only API. Examples:

    * `'openai-responses:gpt-5.4'` — OpenAI model with image generation support
    * `'google-gla:gemini-3-pro-image-preview'` — Google image generation model

    Can be a model name string, `Model` instance, or a callable taking `RunContext`
    that returns a `Model` instance.
    """

    # Keep these fields in sync with ImageGenerationTool in builtin_tools.py.

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

    Supported by: OpenAI Responses (jpeg/webp, default: 100), Google Vertex AI (jpeg, default: 75).
    """

    output_format: Literal['png', 'webp', 'jpeg'] | None
    """Output format of the generated image.

    Supported by: OpenAI Responses (default: `'png'`), Google Vertex AI.
    """

    quality: Literal['low', 'medium', 'high', 'auto'] | None
    """Quality of the generated image.

    Supported by: OpenAI Responses.
    """

    size: Literal['auto', '1024x1024', '1024x1536', '1536x1024', '512', '1K', '2K', '4K'] | None
    """Size of the generated image.

    Supported by: OpenAI Responses (`'auto'`, `'1024x1024'`, `'1024x1536'`, `'1536x1024'`),
    Google (`'512'`, `'1K'`, `'2K'`, `'4K'`).
    """

    aspect_ratio: ImageAspectRatio | None
    """Aspect ratio for generated images.

    Supported by: Google (Gemini), OpenAI Responses (maps `'1:1'`, `'2:3'`, `'3:2'` to sizes).
    """

    def __init__(
        self,
        *,
        builtin: ImageGenerationTool
        | Callable[[RunContext[AgentDepsT]], Awaitable[ImageGenerationTool | None] | ImageGenerationTool | None]
        | bool = True,
        local: Tool[AgentDepsT] | Callable[..., Any] | Literal[False] | None = None,
        fallback_model: Model
        | KnownModelName
        | str
        | Callable[[RunContext[AgentDepsT]], Awaitable[Model] | Model]
        | None = None,
        action: Literal['generate', 'edit', 'auto'] | None = None,
        background: Literal['transparent', 'opaque', 'auto'] | None = None,
        input_fidelity: Literal['high', 'low'] | None = None,
        moderation: Literal['auto', 'low'] | None = None,
        image_model: ImageGenerationModelName | None = None,
        output_compression: int | None = None,
        output_format: Literal['png', 'webp', 'jpeg'] | None = None,
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
        self.action = action
        self.background = background
        self.input_fidelity = input_fidelity
        self.moderation = moderation
        self.image_model = image_model
        self.output_compression = output_compression
        self.output_format = output_format
        self.quality = quality
        self.size = size
        self.aspect_ratio = aspect_ratio
        self.__post_init__()

    def _image_gen_kwargs(self) -> dict[str, Any]:
        """Collect non-None ImageGenerationTool config fields."""
        kwargs: dict[str, Any] = {}
        if self.action is not None:
            kwargs['action'] = self.action
        if self.background is not None:
            kwargs['background'] = self.background
        if self.input_fidelity is not None:
            kwargs['input_fidelity'] = self.input_fidelity
        if self.moderation is not None:
            kwargs['moderation'] = self.moderation
        if self.image_model is not None:
            kwargs['model'] = self.image_model
        if self.output_compression is not None:
            kwargs['output_compression'] = self.output_compression
        if self.output_format is not None:
            kwargs['output_format'] = self.output_format
        if self.quality is not None:
            kwargs['quality'] = self.quality
        if self.size is not None:
            kwargs['size'] = self.size
        if self.aspect_ratio is not None:
            kwargs['aspect_ratio'] = self.aspect_ratio
        return kwargs

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

        return image_generation_tool(model=self.fallback_model, builtin_tool=self._resolved_builtin())
