from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from pydantic_ai.settings import ModelSettings as _ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability


@dataclass
class ModelSettings(AbstractCapability[AgentDepsT]):
    """Capability that provides model settings, either static or dynamic.

    When `settings` is a callable, it receives the [`RunContext`][pydantic_ai.tools.RunContext]
    and is called before each model request, allowing per-step settings.

    Settings from this capability are merged on top of the agent's top-level `model_settings`
    (same additive pattern as instructions), so capability settings take precedence over
    agent-level defaults but can still be overridden by run-level settings.

    Short name is intentional — passing a dict is enough to get type checking,
    and users rarely need both this and `settings.ModelSettings` in the same scope.
    """

    settings: _ModelSettings | Callable[[RunContext[AgentDepsT]], _ModelSettings]

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> ModelSettings[Any]:
        """Create from spec. Accepts model settings as kwargs (e.g. max_tokens=4096)."""
        if args:
            return cls(settings=cast(_ModelSettings, args[0]))
        return cls(settings=cast(_ModelSettings, kwargs))

    def get_model_settings(self) -> _ModelSettings | Callable[[RunContext[AgentDepsT]], _ModelSettings] | None:
        return self.settings
