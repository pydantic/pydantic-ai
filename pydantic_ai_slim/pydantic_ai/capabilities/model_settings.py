from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings, merge_model_settings
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability


@dataclass
class ModelSettingsCapability(AbstractCapability[AgentDepsT]):
    """Capability that provides model settings, either static or dynamic.

    When `settings` is a callable, it receives the [`RunContext`][pydantic_ai.tools.RunContext]
    and is called before each model request, allowing per-step settings.
    """

    settings: ModelSettings | Callable[[RunContext[AgentDepsT]], ModelSettings]

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'ModelSettings'

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> ModelSettingsCapability[Any]:
        """Create from spec. Accepts model settings as kwargs (e.g. max_tokens=4096)."""
        if args:
            return cls(settings=cast(ModelSettings, args[0]))
        return cls(settings=cast(ModelSettings, kwargs))

    def get_model_settings(self) -> ModelSettings | None:
        if callable(self.settings):
            return None
        return self.settings

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        messages: list[ModelMessage],
        model_settings: ModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[list[ModelMessage], ModelSettings, ModelRequestParameters]:
        resolved = self.settings(ctx) if callable(self.settings) else self.settings
        model_settings = merge_model_settings(model_settings, resolved) or resolved
        return messages, model_settings, model_request_parameters
