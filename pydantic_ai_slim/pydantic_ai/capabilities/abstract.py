from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic

from pydantic_ai import _instructions
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, BuiltinToolFunc, RunContext
from pydantic_ai.toolsets import AbstractToolset, ToolsetFunc


@dataclass
class AbstractCapability(ABC, Generic[AgentDepsT]):
    """Abstract base class for agent capabilities."""

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Return the name used for spec serialization (CamelCase class name by default).

        Return None to opt out of spec-based construction.
        """
        return cls.__name__

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> AbstractCapability[Any]:
        """Create from spec arguments. Default: ``cls(*args, **kwargs)``.

        Override when ``__init__`` takes non-serializable types.
        """
        return cls(*args, **kwargs)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        """Return the capability instance to use for this agent run.

        Called once per run, before ``get_*()`` re-extraction and before any hooks fire.
        Override to return a fresh instance for per-run state isolation.
        Default: return ``self`` (shared across runs).
        """
        return self

    def get_instructions(self) -> _instructions.Instructions[AgentDepsT] | None:
        """Return static instructions to include in the system prompt, or None."""
        return None

    def get_model_settings(self) -> ModelSettings | None:
        """Return static model settings to merge into the agent's defaults, or None."""
        return None

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | ToolsetFunc[AgentDepsT] | None:
        """Return a toolset to register with the agent, or None."""
        return None

    def get_builtin_tools(self) -> Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]]:
        """Return builtin tools to register with the agent."""
        return []

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        messages: list[ModelMessage],
        model_settings: ModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[list[ModelMessage], ModelSettings, ModelRequestParameters]:
        """Called before each model request. Can modify messages, settings, and parameters."""
        return messages, model_settings, model_request_parameters

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        response: ModelResponse,
    ) -> ModelResponse:
        """Called after each model response. Can modify the response before further processing."""
        return response
