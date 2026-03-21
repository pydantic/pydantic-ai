from __future__ import annotations

from typing import Any, cast

from pydantic_ai.settings import ModelSettings as _ModelSettings
from pydantic_ai.tools import AgentDepsT

from .model_settings import ModelSettings


class Thinking(ModelSettings[AgentDepsT]):
    """Capability that enables model thinking/reasoning.

    This is a placeholder that hardcodes provider-specific thinking settings.
    It will be replaced by unified thinking settings from #3894, which will
    also allow configurable parameters (e.g. reasoning_effort, budget_tokens).
    """

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'Thinking'

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> Thinking[Any]:
        if args or kwargs:
            raise TypeError(
                'Thinking() does not accept arguments yet — configurable parameters will be available once'
                ' #3894 lands. Use ModelSettings capability for custom thinking settings.'
            )
        return cls()

    def __init__(self):
        # Bypasses the dataclass-generated __init__ to hardcode provider-specific
        # thinking settings. Cast needed because ModelSettings is a TypedDict and
        # these provider-specific keys aren't in the base type.
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
