from __future__ import annotations

from typing import Any, cast

from pydantic_ai.settings import ModelSettings as _ModelSettings
from pydantic_ai.tools import AgentDepsT

from .model_settings import ModelSettings


class Thinking(ModelSettings[AgentDepsT]):
    """Capability that enables model thinking/reasoning."""

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'Thinking'

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> ModelSettings[Any]:
        return cls()

    def __init__(self):
        super().__init__(
            cast(
                _ModelSettings,
                {
                    'openai_reasoning_effort': 'high',
                    'anthropic_thinking': {'type': 'adaptive'},
                    # TODO: Use unified thinking settings from #3894 once merged
                },
            ),
        )
