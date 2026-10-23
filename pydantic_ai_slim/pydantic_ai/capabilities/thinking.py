from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import Any

from pydantic_ai.settings import ModelSettings, ThinkingLevel

from .abstract import AbstractCapability


@dataclass
class Thinking(AbstractCapability[Any]):
    """Enables and configures model thinking/reasoning.

    Uses the unified `thinking` setting in
    [`ModelSettings`][pydantic_ai.settings.ModelSettings] to work portably across providers.
    Provider-specific thinking settings (e.g., `anthropic_thinking`,
    `openai_reasoning_effort`) take precedence when both are set.

    `summary` sets the unified `thinking_summary` setting, which controls whether the thinking comes back readable.
    """

    effort: ThinkingLevel = True
    """The thinking effort level.

    - `True`: Enable thinking with the provider's default effort.
    - `False`: Disable thinking (silently ignored on always-on models).
    - `'minimal'`/`'low'`/`'medium'`/`'high'`/`'xhigh'`/`'max'`: Enable thinking at a specific effort level.
    """

    _: KW_ONLY

    summary: bool | None = None
    """Whether the model should return readable thinking, as the unified
    [`thinking_summary`][pydantic_ai.settings.ModelSettings.thinking_summary] setting.

    - `True`: Ask for a summary of the model's thinking (or the raw reasoning on open models) wherever the API has a switch for it.
    - `False`: Ask the API not to return thinking content where it can; the model still thinks.
    - `None` (default): Leave `thinking_summary` unset, so summaries are requested where that is safe by default.
    """

    id: str | None = 'thinking'
    """One-off: an agent has a single thinking configuration, so the id is fixed by default.

    Two of them resolve to one via [`combine`][pydantic_ai.capabilities.AbstractCapability.combine],
    which keeps the last. Pass a distinct `id` to keep both, or `id=None` for derived ids.
    """

    def get_model_settings(self) -> ModelSettings | None:
        settings = ModelSettings(thinking=self.effort)
        if self.summary is not None:
            settings['thinking_summary'] = self.summary
        return settings
