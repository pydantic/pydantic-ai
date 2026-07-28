from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai.settings import ModelSettings, ThinkingLevel
from pydantic_ai.tools import RunContext

from .abstract import AbstractCapability

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AgentModelSettings


@dataclass
class Thinking(AbstractCapability[Any]):
    """Enables and configures model thinking/reasoning.

    Uses the unified `thinking` setting in
    [`ModelSettings`][pydantic_ai.settings.ModelSettings] to work portably across providers.
    Provider-specific thinking settings (e.g., `anthropic_thinking`,
    `openai_reasoning_effort`) take precedence when both are set.

    The capability contributes a *default*: it applies only when no earlier
    [settings layer](../agent.md#dynamic-model-settings) has set `thinking`. A value from the
    model's own settings, from `Agent(model_settings=...)`, or from a capability positioned
    outside this one therefore wins. Set
    [`override`][pydantic_ai.capabilities.Thinking.override] to `True` to apply
    [`effort`][pydantic_ai.capabilities.Thinking.effort] regardless.
    """

    effort: ThinkingLevel = True
    """The thinking effort level.

    - `True`: Enable thinking with the provider's default effort.
    - `False`: Disable thinking (silently ignored on always-on models).
    - `'minimal'`/`'low'`/`'medium'`/`'high'`/`'xhigh'`: Enable thinking at a specific effort level.
    """

    override: bool = False
    """Whether to override a `thinking` value set by an earlier settings layer.

    By default the capability yields to an existing value, so it reads as "enable thinking
    unless told otherwise". Set to `True` to always apply
    [`effort`][pydantic_ai.capabilities.Thinking.effort], beating the model's settings,
    `Agent(model_settings=...)`, and any capability outside this one. Per-run
    `model_settings` are merged after all capabilities and win either way.
    """

    def get_model_settings(self) -> AgentModelSettings[Any]:
        if self.override:
            return ModelSettings(thinking=self.effort)

        def resolve(ctx: RunContext[Any]) -> ModelSettings:
            # `ctx.model_settings` holds every layer resolved before this capability,
            # so an existing `thinking` value means someone was more explicit than us.
            if ctx.model_settings is not None and 'thinking' in ctx.model_settings:
                return ModelSettings()
            return ModelSettings(thinking=self.effort)

        return resolve
