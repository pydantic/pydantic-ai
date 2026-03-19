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
        # Cast needed because ModelSettings is a TypedDict and we're constructing
        # it from a plain dict with provider-specific keys that aren't in the base type.
        # Providers covered: OpenAI, Anthropic, Google (google.genai SDK), Gemini (direct API)
        super().__init__(
            cast(
                _ModelSettings,
                {
                    'openai_reasoning_effort': 'high',
                    'anthropic_thinking': {'type': 'adaptive'},
                    'google_thinking_config': {'include_thoughts': True},
                    'gemini_thinking_config': {'include_thoughts': True},
                },
            ),
        )
