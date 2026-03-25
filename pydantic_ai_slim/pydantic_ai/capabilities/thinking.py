from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.settings import ModelSettings as _ModelSettings, ThinkingLevel
from pydantic_ai.tools import AgentDepsT

from .abstract import AbstractCapability


@dataclass
class Thinking(AbstractCapability[AgentDepsT]):
    """Enables and configures model thinking/reasoning.

    Uses the unified `thinking` setting in
    [`ModelSettings`][pydantic_ai.settings.ModelSettings] to work portably across providers.
    Provider-specific thinking settings (e.g., `anthropic_thinking`,
    `openai_reasoning_effort`) take precedence when both are set.
    """

    effort: ThinkingLevel = True
    """The thinking effort level.

    - `True`: Enable thinking with the provider's default effort.
    - `False`: Disable thinking (silently ignored on always-on models).
    - `'minimal'`/`'low'`/`'medium'`/`'high'`/`'xhigh'`: Enable thinking at a specific effort level.
    """

    def get_model_settings(self) -> _ModelSettings | None:
        return _ModelSettings(thinking=self.effort)
