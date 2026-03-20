from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Sequence
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
class BeforeModelRequestContext:
    """Context passed to and returned from [`AbstractCapability.before_model_request`][pydantic_ai.capabilities.abstract.AbstractCapability.before_model_request].

    Wrapping these parameters in a dataclass instead of a tuple makes the signature
    future-proof: new fields can be added without breaking existing implementations.
    """

    messages: list[ModelMessage]
    model_settings: ModelSettings
    model_request_parameters: ModelRequestParameters


@dataclass
class AbstractCapability(ABC, Generic[AgentDepsT]):
    """Abstract base class for agent capabilities.

    A capability is a reusable, composable unit of agent behavior that can provide
    instructions, model settings, tools, and request/response hooks.

    Lifecycle: capabilities are passed to an [`Agent`][pydantic_ai.Agent] at construction time, where their
    `get_*` methods are called to collect static configuration (instructions, model
    settings, toolsets, builtin tools). Then, on each model request during a run, the
    [`before_model_request`][pydantic_ai.capabilities.AbstractCapability.before_model_request] and
    [`after_model_request`][pydantic_ai.capabilities.AbstractCapability.after_model_request] hooks
    are called to allow dynamic adjustments.

    Built-in capabilities:

    - [`Instructions`][pydantic_ai.capabilities.Instructions] — static or template-based system prompt instructions
    - [`Thinking`][pydantic_ai.capabilities.Thinking] — enables model thinking/reasoning mode
    - [`ModelSettings`][pydantic_ai.capabilities.ModelSettings] — provides extra model settings
    - [`WebSearch`][pydantic_ai.capabilities.WebSearch] — registers the web search builtin tool
    - [`HistoryProcessorCapability`][pydantic_ai.capabilities.HistoryProcessorCapability] — wraps a history processor as a capability
    - [`Toolset`][pydantic_ai.capabilities.Toolset] — registers a toolset with the agent

    Custom capabilities that should work with YAML/JSON specs (via
    [`Agent.from_spec`][pydantic_ai.Agent.from_spec]) can override
    [`get_serialization_name`][pydantic_ai.capabilities.AbstractCapability.get_serialization_name]
    and [`from_spec`][pydantic_ai.capabilities.AbstractCapability.from_spec].
    """

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

    def get_model_settings(self) -> ModelSettings | Callable[[RunContext[AgentDepsT]], ModelSettings] | None:
        """Return model settings to merge into the agent's defaults, or None.

        Return a static `ModelSettings` dict when the settings are known at agent
        construction time and don't change between requests. Return a callable
        that receives [`RunContext`][pydantic_ai.tools.RunContext] when settings
        need to vary per step (e.g. based on `ctx.run_step` or `ctx.deps`).

        When the callable is invoked, `ctx.model_settings` contains the merged
        result of all layers resolved before this capability (model defaults and
        agent-level settings). The returned dict is merged on top of that.
        """
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
        request_context: BeforeModelRequestContext,
    ) -> BeforeModelRequestContext:
        """Called before each model request. Can modify messages, settings, and parameters."""
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        response: ModelResponse,
    ) -> ModelResponse:
        """Called after each model response. Can modify the response before further processing."""
        return response
