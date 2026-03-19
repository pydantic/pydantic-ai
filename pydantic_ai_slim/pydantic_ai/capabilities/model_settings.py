from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from pydantic_ai.settings import ModelSettings as _ModelSettings, merge_model_settings
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability, BeforeModelRequestContext


@dataclass
class ModelSettings(AbstractCapability[AgentDepsT]):
    """Capability that provides model settings, either static or dynamic.

    When `settings` is a callable, it receives the [`RunContext`][pydantic_ai.tools.RunContext]
    and is called before each model request, allowing per-step settings.

    Settings from this capability are merged on top of the agent's top-level `model_settings`
    (same additive pattern as instructions), so capability settings take precedence over
    agent-level defaults but can still be overridden by run-level settings.
    """

    settings: _ModelSettings | Callable[[RunContext[AgentDepsT]], _ModelSettings]

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'ModelSettings'

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> ModelSettings[Any]:
        """Create from spec. Accepts model settings as kwargs (e.g. max_tokens=4096)."""
        if args:
            return cls(settings=cast(_ModelSettings, args[0]))
        return cls(settings=cast(_ModelSettings, kwargs))

    def get_model_settings(self) -> _ModelSettings | None:
        if callable(self.settings):
            return None
        return self.settings

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: BeforeModelRequestContext,
    ) -> BeforeModelRequestContext:
        if callable(self.settings):
            # Dynamic settings need to be resolved and merged per request;
            # static settings are already handled by get_model_settings.
            resolved = self.settings(ctx)
            request_context.model_settings = merge_model_settings(request_context.model_settings, resolved) or resolved
        return request_context
