from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai.builtin_tools import ImageGenerationTool
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import BinaryImage
from pydantic_ai.models import Model
from pydantic_ai.tools import RunContext, Tool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pydantic_ai.models import KnownModelName

    FallbackModelFunc = Callable[
        [RunContext[Any]],
        Awaitable[Model | KnownModelName | str | None] | Model | KnownModelName | str | None,
    ]
    """Callable that resolves a fallback model dynamically per-run."""

    FallbackModel = Model | KnownModelName | str | FallbackModelFunc | None
    """Type for the fallback model: a model, model name, factory callable, or None."""

__all__ = ('image_generation_tool',)

# Known image-only model names that don't support the conversational Agent loop
# required by the subagent fallback, mapped to suggested LLM alternatives.
_IMAGE_ONLY_MODELS: dict[str, str] = {
    'gpt-image-1': 'gpt-5.4',
    'dall-e-3': 'gpt-5.4',
    'dall-e-2': 'gpt-5.4',
}


def _check_image_only_model(model: str) -> None:
    """Raise UserError if the model is a known image-only model."""
    from pydantic_ai.models import parse_model_id

    _, model_name = parse_model_id(model)
    if suggestion := _IMAGE_ONLY_MODELS.get(model_name):
        raise UserError(
            f'{model_name!r} is a dedicated image generation model that cannot be used as '
            f'`fallback_model` directly. Use a conversational model with image generation '
            f'support instead, e.g. {suggestion!r}.'
        )


@dataclass
class ImageGenerationLocalTool:
    """Local image generation tool that delegates to a subagent.

    Uses a subagent with the specified model and builtin tool configuration
    to generate images when the outer agent's model doesn't support image
    generation natively.
    """

    model: Model | str | FallbackModelFunc
    """The model to use for image generation, or a callable that returns one."""

    builtin: ImageGenerationTool
    """The image generation tool configuration to pass to the subagent."""

    async def __call__(self, ctx: RunContext[Any], prompt: str) -> BinaryImage:
        """Generate an image using a subagent.

        Args:
            ctx: The run context from the outer agent.
            prompt: A description of the image to generate.
        """
        from pydantic_ai.agent import Agent

        model = self.model
        if callable(model):
            result = model(ctx)
            if inspect.isawaitable(result):
                result = await result
            model = result

        if model is None:
            raise ModelRetry('The fallback model callable returned None; cannot generate an image.')

        if isinstance(model, str) and callable(self.model):
            # Only check at call time for dynamically resolved models;
            # static strings are already validated at factory time
            _check_image_only_model(model)

        agent = Agent(model, output_type=BinaryImage, builtin_tools=[self.builtin])
        try:
            result = await agent.run(prompt)
        except UnexpectedModelBehavior as e:
            raise ModelRetry(str(e)) from e
        return result.output


def image_generation_tool(
    model: Model | str | FallbackModelFunc,
    builtin: ImageGenerationTool,
) -> Tool[Any]:
    """Creates an image generation tool backed by a subagent.

    Args:
        model: The model to use for image generation (e.g. ``'openai-responses:gpt-5.4'``),
            or a callable taking ``RunContext`` that returns a model.
        builtin: The image generation tool configuration to pass to the subagent.
    """
    if isinstance(model, str):
        _check_image_only_model(model)
    return Tool[Any](
        ImageGenerationLocalTool(model=model, builtin=builtin).__call__,
        name='generate_image',
        description='Generate an image based on the given prompt.',
    )
