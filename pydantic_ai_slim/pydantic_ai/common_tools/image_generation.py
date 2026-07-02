from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic_ai.agent import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import BinaryImage
from pydantic_ai.models import KnownModelName, Model, parse_model_id
from pydantic_ai.native_tools import ImageGenerationTool
from pydantic_ai.tools import RunContext, Tool

ImageGenerationFallbackModelFunc = Callable[
    [RunContext[Any]],
    Awaitable[Model | KnownModelName | str] | Model | KnownModelName | str,
]
"""Callable that resolves a fallback model dynamically per-run.

May return a `Model` instance or a model name string (e.g. `'openai-responses:gpt-5.4'`);
strings are resolved to a model at call time.
"""

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
    'imagen-3.0-generate-002': 'google:gemini-3-pro-image-preview',
    'imagen-3.0-fast-generate-001': 'google:gemini-3-pro-image-preview',
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

    Uses a subagent with the specified model and native tool configuration
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


def image_generation_tool(
    model: Model | KnownModelName | str | ImageGenerationFallbackModelFunc,
    native_tool: ImageGenerationTool,
    *,
    instructions: str = 'Generate an image based on the user prompt. Do not ask clarifying questions.',
) -> Tool[Any]:
    """Creates an image generation tool backed by a subagent.

    Args:
        model: The model to use for image generation (e.g. `'openai-responses:gpt-5.4'`),
            or a callable taking `RunContext` that returns a model.
        native_tool: The image generation tool configuration to pass to the subagent.
        instructions: Instructions for the subagent that generates the image.
    """
    if isinstance(model, str):
        _check_image_only_model(model)
    return Tool[Any](
        ImageGenerationSubagentTool(model=model, native_tool=native_tool, instructions=instructions).__call__,
        name='generate_image',
        description='Generate an image based on the given prompt.',
    )
