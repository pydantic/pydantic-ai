from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings, merge_model_settings
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability


@dataclass
class ModelSettingsCapability(AbstractCapability[AgentDepsT]):
    """A capability that applies model settings before each request."""

    settings: ModelSettings

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'ModelSettings'

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> ModelSettingsCapability[Any]:
        """Create from spec. Accepts model settings as kwargs (e.g. max_tokens=4096)."""
        if args:
            return cls(settings=cast(ModelSettings, args[0]))
        return cls(settings=cast(ModelSettings, kwargs))

    # TODO: Restore get_model_settings() method

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        messages: list[ModelMessage],
        model_settings: ModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[list[ModelMessage], ModelSettings, ModelRequestParameters]:
        model_settings = merge_model_settings(model_settings, self.settings) or self.settings
        return messages, model_settings, model_request_parameters
