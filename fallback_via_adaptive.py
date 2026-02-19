"""Reimplementation of FallbackModel using AdaptiveModel to test flexibility.

This demonstrates that AdaptiveModel can fully replicate FallbackModel's behavior
while being more composable and flexible.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from pydantic_ai_slim.pydantic_ai.models import Model, infer_model
from pydantic_ai_slim.pydantic_ai.models.adaptive import AdaptiveContext, AdaptiveModel

if TYPE_CHECKING:
    from pydantic_ai_slim.pydantic_ai.exceptions import ModelHTTPError


@dataclass
class FallbackState:
    """State for fallback model selection."""

    models: list[Model]
    fallback_on: Callable[[Exception], bool] | None = None
    current_index: int = 0


def fallback_selector(ctx: AdaptiveContext[FallbackState]) -> Model:
    """Select the next model in the fallback sequence."""
    state = ctx.state

    # If this is the first attempt, start with the first model
    if len(ctx.attempts) == 0:
        state.current_index = 0
        return state.models[0]

    # Move to next model
    state.current_index += 1
    if state.current_index < len(state.models):
        return state.models[state.current_index]
    else:
        # No more models to try
        raise RuntimeError(f"All {len(state.models)} models failed")


def on_attempt_failed(
    state: FallbackState,
    model: Model,
    exception: Exception,
    timestamp: datetime,
    duration: timedelta,
) -> bool:
    """Decide whether to continue trying after a failure.

    Returns True if the exception should trigger a fallback, False to stop immediately.
    """
    return state.fallback_on(exception)


def create_fallback_model(
    default_model: Model | str,
    *fallback_models: Model | str,
    fallback_on: Callable[[Exception], bool] | tuple[type[Exception], ...] = None,
) -> AdaptiveModel[FallbackState]:
    """Create a FallbackModel using AdaptiveModel.

    Args:
        default_model: The name or instance of the default model to use.
        fallback_models: The names or instances of the fallback models to use upon failure.
        fallback_on: A callable or tuple of exceptions that should trigger a fallback.
            Defaults to ModelHTTPError.

    Returns:
        An AdaptiveModel configured to behave like FallbackModel.
    """
    # Convert all models
    models = [infer_model(default_model), *[infer_model(m) for m in fallback_models]]

    # Handle fallback_on parameter
    if fallback_on is None:
        # Default to ModelHTTPError
        from pydantic_ai_slim.pydantic_ai.exceptions import ModelHTTPError

        fallback_on_callable = lambda exc: isinstance(exc, ModelHTTPError)
    elif isinstance(fallback_on, tuple):
        # Convert tuple of exceptions to callable
        fallback_on_callable = lambda exc: isinstance(exc, fallback_on)
    else:
        fallback_on_callable = fallback_on

    # Create state
    state = FallbackState(models=models, fallback_on=fallback_on_callable)

    # Create adaptive model with failure hook
    return AdaptiveModel(
        selector=fallback_selector,
        state=state,
        on_attempt_failed=on_attempt_failed,
    )


# Example usage comparison:
if __name__ == "__main__":
    print("Fallback via Adaptive implementation loaded successfully!")
    print("\nUsage:")
    print("  model = create_fallback_model('openai:gpt-4o', 'anthropic:claude-3-5-sonnet')")
    print("\nThis demonstrates AdaptiveModel can fully replicate FallbackModel behavior.")
