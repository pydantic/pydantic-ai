"""Per-model capability routing.

Provides [`PerModelCapability`][pydantic_ai.capabilities.PerModelCapability], a capability
that delegates to different implementations based on the model being used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pydantic_ai.exceptions import UserError
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability

if TYPE_CHECKING:
    from pydantic_ai.models import Model


def _unwrap_model(model: Model) -> Model:
    """Unwrap wrapper models to find the underlying provider model."""
    from pydantic_ai.models.wrapper import WrapperModel

    while isinstance(model, WrapperModel):
        model = model.wrapped
    return model


@dataclass
class PerModelCapability(AbstractCapability[AgentDepsT]):
    """A capability that delegates to different implementations based on the model.

    Can be used directly with a `routes` mapping, or subclassed with
    [`get_capability_for_model`][pydantic_ai.capabilities.PerModelCapability.get_capability_for_model]
    for dynamic routing.

    At runtime, `for_run` unwraps any wrapper models, rejects `FallbackModel`,
    and delegates to the matched capability.

    Example using `routes` directly::

        cap = PerModelCapability(routes={
            'openai': my_openai_capability,
            'anthropic': my_anthropic_capability,
        })

    Example subclassing::

        class MyCapability(PerModelCapability):
            def get_capability_for_model(self, model):
                ...  # return provider-specific capability based on model
    """

    routes: dict[type[Model] | str, AbstractCapability[AgentDepsT]] = field(default_factory=dict)  # pyright: ignore[reportUnknownVariableType]
    """Mapping of model class or system string (e.g. `'openai'`, `'anthropic'`) to capability instance."""

    fallback: AbstractCapability[AgentDepsT] | Literal['ignore'] | None = None
    """What to do when the model doesn't match any route.

    - `None`: Raise `UserError` (default, safest).
    - `'ignore'`: Silently return a no-op capability.
    - A capability instance: Use it as the fallback.
    """

    def get_capability_for_model(self, model: Model) -> AbstractCapability[AgentDepsT] | None:
        """Return the provider-specific capability for the given (unwrapped) model.

        Override this to define custom routing logic.
        The default implementation looks up `routes` by model class and system string.

        Returns:
            A capability instance, or `None` to indicate the model is unmatched
            (which will fall through to `fallback` handling).
        """
        # Match by model class
        for key, cap in self.routes.items():
            if isinstance(key, type) and isinstance(model, key):
                return cap

        # Match by system string
        if model.system in self.routes:
            key_str = model.system
            return self.routes[key_str]

        return None

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        model = _unwrap_model(ctx.model)

        if isinstance(model, FallbackModel):
            raise UserError(
                f'{type(self).__name__} is not compatible with FallbackModel '
                f'because it produces provider-specific data that cannot be used across providers'
            )

        cap = self.get_capability_for_model(model)
        if cap is not None:
            return await cap.for_run(ctx)

        # No route matched — check fallback
        if isinstance(self.fallback, AbstractCapability):
            return await self.fallback.for_run(ctx)

        if self.fallback == 'ignore':
            return _NoOpCapability()

        raise UserError(
            f'{type(self).__name__} does not have a route for {model.model_name} '
            f'(system={model.system!r}). Add a route, set fallback, or use fallback="ignore".'
        )

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None


@dataclass
class _NoOpCapability(AbstractCapability[AgentDepsT]):
    """Internal no-op capability returned when fallback='ignore'."""

    pass
