from __future__ import annotations

import inspect
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic_ai._utils import install_deprecated_kwarg_alias
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.agent import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import BinaryImage
from pydantic_ai.models import KnownModelName, Model, parse_model_id
from pydantic_ai.native_tools import ImageGenerationTool
from pydantic_ai.tools import RunContext, Tool

ImageGenerationFallbackModelFunc = Callable[
    [RunContext[Any]],
    Awaitable[Model] | Model,
]
"""Callable that resolves a fallback model dynamically per-run."""

ImageGenerationFallbackModel = Model | KnownModelName | str | ImageGenerationFallbackModelFunc | None
"""Type for the fallback model: a model, model name, factory callable, or None."""

__all__ = (
    'ImageGenerationFallbackModel',
    'ImageGenerationFallbackModelFunc',
    'ImageGenerationSubagentTool',
    'image_generation_tool',
)

# Known image-only model names that don't support the conversational Agent loop
# required by the subagent fallback, mapped to suggested LLM alternatives.
_IMAGE_ONLY_MODELS: dict[str, str] = {
    'gpt-image-2': 'openai-responses:gpt-5.5',
    'gpt-image-1.5': 'openai-responses:gpt-5.5',
    'gpt-image-1': 'openai-responses:gpt-5.4',
    'gpt-image-1-mini': 'openai-responses:gpt-5.4',
    'dall-e-3': 'openai-responses:gpt-5.4',
    'dall-e-2': 'openai-responses:gpt-5.4',
    'imagen-3.0-generate-002': 'google-gla:gemini-3-pro-image-preview',
    'imagen-3.0-fast-generate-001': 'google-gla:gemini-3-pro-image-preview',
}


def _check_image_only_model(model: str) -> None:
    """Raise UserError if the model is a known image-only model."""
    _, model_name = parse_model_id(model)
    if suggestion := _IMAGE_ONLY_MODELS.get(model_name):
        raise UserError(
            f'{model_name!r} is a dedicated image generation model that cannot be used as '
            f'`fallback_model` directly. Use a conversational model with image generation '
            f'support instead, e.g. {suggestion!r}.'
        )


@dataclass(kw_only=True)
class ImageGenerationSubagentTool:
    """Local image generation tool that delegates to a subagent.

    Uses a subagent with the specified model and builtin tool configuration
    to generate images when the outer agent's model doesn't support image
    generation natively.
    """

    model: Model | KnownModelName | str | ImageGenerationFallbackModelFunc
    """The model to use for image generation, or a callable that returns one."""

    native_tool: ImageGenerationTool
    """The image generation tool configuration to pass to the subagent."""

    instructions: str = 'Generate an image based on the user prompt. Do not ask clarifying questions.'
    """Instructions for the subagent that generates the image."""

    async def __call__(self, ctx: RunContext[Any], prompt: str) -> BinaryImage:
        """Generate an image using a subagent.

        Args:
            ctx: The run context from the outer agent.
            prompt: A description of the image to generate.
        """
        model = self.model
        if callable(model):
            result = model(ctx)
            if inspect.isawaitable(result):
                result = await result
            model = result

        if isinstance(model, str) and callable(self.model):
            # Only check at call time for dynamically resolved models;
            # static strings are already validated at factory time
            _check_image_only_model(model)

        agent = Agent(
            model,
            output_type=BinaryImage,
            capabilities=[NativeTool(self.native_tool)],
            instructions=self.instructions,
        )
        try:
            result = await agent.run(prompt)
        except UnexpectedModelBehavior as e:
            raise ModelRetry(str(e)) from e
        return result.output

    def __getattr__(self, name: str) -> Any:
        # Deprecated alias for read access to the renamed `builtin_tool` field.
        if name == 'builtin_tool':
            warnings.warn(
                '`ImageGenerationSubagentTool.builtin_tool` is deprecated, use `.native_tool` instead.',
                PydanticAIDeprecationWarning,
                stacklevel=2,
            )
            return self.native_tool
        raise AttributeError(name)


install_deprecated_kwarg_alias(ImageGenerationSubagentTool, old='builtin_tool', new='native_tool')


def image_generation_tool(
    model: Model | KnownModelName | str | ImageGenerationFallbackModelFunc,
    native_tool: ImageGenerationTool | None = None,
    *,
    instructions: str = 'Generate an image based on the user prompt. Do not ask clarifying questions.',
    **_deprecated_kwargs: Any,
) -> Tool[Any]:
    """Creates an image generation tool backed by a subagent.

    Args:
        model: The model to use for image generation (e.g. `'openai-responses:gpt-5.4'`),
            or a callable taking `RunContext` that returns a model.
        native_tool: The image generation tool configuration to pass to the subagent.
        instructions: Instructions for the subagent that generates the image.
    """
    if 'builtin_tool' in _deprecated_kwargs:
        warnings.warn(
            '`image_generation_tool(builtin_tool=...)` is deprecated, use `native_tool=` instead.',
            PydanticAIDeprecationWarning,
            stacklevel=2,
        )
        legacy_native_tool = _deprecated_kwargs.pop('builtin_tool')
        if native_tool is None:
            native_tool = legacy_native_tool
    if _deprecated_kwargs:
        unknown = ', '.join(f'`{k}`' for k in _deprecated_kwargs)
        raise TypeError(f'image_generation_tool() got unexpected keyword arguments: {unknown}')
    if native_tool is None:
        raise TypeError("image_generation_tool() missing required argument: 'native_tool'")
    if isinstance(model, str):
        _check_image_only_model(model)
    return Tool[Any](
        ImageGenerationSubagentTool(model=model, native_tool=native_tool, instructions=instructions).__call__,
        name='generate_image',
        description='Generate an image based on the given prompt.',
    )
