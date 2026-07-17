from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

from pydantic_ai._utils import is_async_callable
from pydantic_ai.models import KnownModelName, Model, ModelResolutionContext
from pydantic_ai.tools import AgentDepsT

from .abstract import AbstractCapability

ModelIdResolver = (
    Callable[[ModelResolutionContext[AgentDepsT], str], Model | None]
    | Callable[[ModelResolutionContext[AgentDepsT], str], Awaitable[Model | None]]
)
"""A sync or async model ID resolver."""


@dataclass
class ResolveModelId(AbstractCapability[AgentDepsT]):
    """Resolve model IDs with a user-provided sync or async callable.

    The callable receives a [`ModelResolutionContext`][pydantic_ai.models.ModelResolutionContext]
    followed by the selected model ID. Return `None` to let a later capability or the
    default [`infer_model`][pydantic_ai.models.infer_model] behavior handle the ID.
    """

    resolver: ModelIdResolver[AgentDepsT]

    async def resolve_model_id(
        self,
        ctx: ModelResolutionContext[AgentDepsT],
        *,
        model_id: KnownModelName | str,
    ) -> Model | None:
        if is_async_callable(self.resolver):
            resolver = cast(Callable[[ModelResolutionContext[Any], str], Awaitable[Model | None]], self.resolver)
            return await resolver(ctx, model_id)
        resolver = cast(Callable[[ModelResolutionContext[Any], str], Model | None], self.resolver)
        return resolver(ctx, model_id)

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None
