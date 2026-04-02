"""Routed capability base class for model-specific capability routing.

Provides [`RoutedCapability`][pydantic_ai.capabilities.RoutedCapability], a base class
for capabilities that delegate to different implementations based on the model being used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai.exceptions import UserError
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
class RoutedCapability(AbstractCapability[AgentDepsT]):
    """Base class for capabilities that route to model-specific implementations.

    Subclasses override [`get_capability_for_model`][pydantic_ai.capabilities.RoutedCapability.get_capability_for_model]
    to return a provider-specific capability based on the model. At runtime, ``for_run``
    unwraps any wrapper models, rejects ``FallbackModel``, and delegates to the subclass.

    Example::

        class MyCapability(RoutedCapability):
            def get_capability_for_model(self, model):
                if isinstance(model, SomeModel):
                    return SomeModelCapability(...)
                raise UserError(f'MyCapability not supported for {model.model_name}')
    """

    def get_capability_for_model(self, model: Model) -> AbstractCapability[AgentDepsT]:
        """Return the provider-specific capability for the given (unwrapped) model.

        Override this to define the routing logic.
        Raise [`UserError`][pydantic_ai.exceptions.UserError] for unsupported models.
        """
        raise NotImplementedError

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        model = _unwrap_model(ctx.model)

        try:
            from pydantic_ai.models.fallback import FallbackModel

            if isinstance(model, FallbackModel):
                raise UserError(
                    f'{type(self).__name__} is not compatible with FallbackModel '
                    f'because it produces provider-specific data that cannot be used across providers'
                )
        except ImportError:
            pass

        return self.get_capability_for_model(model)

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # pragma: no cover — Subclasses should override if they want serialization support
