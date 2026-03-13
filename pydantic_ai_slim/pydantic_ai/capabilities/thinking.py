from __future__ import annotations

from typing import Any, cast

from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT

from .model_settings import ModelSettingsCapability


class Thinking(ModelSettingsCapability[AgentDepsT]):
    """A capability that enables model thinking/reasoning."""

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'Thinking'

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> ModelSettingsCapability[Any]:
        return cls()

    def __init__(self):
        super().__init__(
            cast(
                ModelSettings,
                {
                    'openai_reasoning_effort': 'high',
                    'anthropic_thinking': {'type': 'adaptive'},
                    # etc
                },
            ),
        )
