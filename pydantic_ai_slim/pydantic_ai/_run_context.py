from __future__ import annotations as _annotations

import dataclasses
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import field
from typing import TYPE_CHECKING, Any, Generic

from opentelemetry.trace import NoOpTracer, Tracer
from typing_extensions import TypeVar

from pydantic_ai._instrumentation import DEFAULT_INSTRUMENTATION_VERSION

from . import _utils, messages as _messages

if TYPE_CHECKING:
    from .agent import Agent
    from .capabilities.abstract import AbstractCapability
    from .models import Model
    from .result import RunUsage
    from .settings import ModelSettings
    from .tool_manager import ToolManager
    from .tools import ToolDefinition


# TODO (v2): Change the default for all typevars like this from `None` to `object`
AgentDepsT = TypeVar('AgentDepsT', default=None, contravariant=True)
"""Type variable for agent dependencies."""

RunContextAgentDepsT = TypeVar('RunContextAgentDepsT', default=None, covariant=True)
"""Type variable for the agent dependencies in `RunContext`."""


@dataclasses.dataclass(repr=False, kw_only=True)
class RunContext(Generic[RunContextAgentDepsT]):
    """Information about the current call."""

    deps: RunContextAgentDepsT
    """Dependencies for the agent."""
    model: Model
    """The model used in this run."""
    usage: RunUsage
    """LLM usage associated with the run."""
    agent: Agent[RunContextAgentDepsT, Any] | None = field(default=None, repr=False)
    """The agent running this context, or `None` if not set."""
    prompt: str | Sequence[_messages.UserContent] | None = None
    """The original user prompt passed to the run."""
    messages: list[_messages.ModelMessage] = field(default_factory=list[_messages.ModelMessage])
    """Messages exchanged in the conversation so far."""
    validation_context: Any = None
    """Pydantic [validation context](https://docs.pydantic.dev/latest/concepts/validators/#validation-context) for tool args and run outputs."""
    tracer: Tracer = field(default_factory=NoOpTracer)
    """The tracer to use for tracing the run."""
    trace_include_content: bool = False
    """Whether to include the content of the messages in the trace."""
    instrumentation_version: int = DEFAULT_INSTRUMENTATION_VERSION
    """Instrumentation settings version, if instrumentation is enabled."""
    retries: dict[str, int] = field(default_factory=dict[str, int])
    """Number of retries for each tool so far."""
    tool_call_id: str | None = None
    """The ID of the tool call."""
    tool_name: str | None = None
    """Name of the tool being called."""
    retry: int = 0
    """Number of retries so far.

    For tool calls, this is the number of retries of the specific tool.
    For output validation, this is the number of output validation retries.
    """
    max_retries: int = 0
    """The maximum number of retries allowed.

    For tool calls, this is the maximum retries for the specific tool.
    For output validation, this is the maximum output validation retries.
    """
    run_step: int = 0
    """The current step in the run."""
    tool_call_approved: bool = False
    """Whether a tool call that required approval has now been approved."""
    tool_call_metadata: Any = None
    """Metadata from `DeferredToolResults.metadata[tool_call_id]`, available when `tool_call_approved=True`."""
    partial_output: bool = False
    """Whether the output passed to an output validator is partial."""
    run_id: str | None = None
    """"Unique identifier for the agent run."""
    conversation_id: str | None = None
    """Unique identifier for the conversation this run belongs to.

    A conversation spans potentially multiple agent runs that share message history.
    Resolved at the start of `Agent.run` (etc.) from the explicit `conversation_id`
    argument, the most recent `conversation_id` on `message_history`, or a fresh UUID7.
    """
    metadata: dict[str, Any] | None = None
    """Metadata associated with this agent run, if configured."""
    model_settings: ModelSettings | None = None
    """The resolved model settings for the current run step.

    Populated before each model request, after all model settings layers
    (model defaults, agent-level, capability, and run-level) have been merged.
    Available in model request hooks (`before_model_request`, `wrap_model_request`,
    `after_model_request`). Currently `None` in tool hooks, output validators,
    and during agent construction.
    """

    tool_manager: ToolManager[RunContextAgentDepsT] | None = None
    """The tool manager for the current run step.

    Provides access to tool validation and execution, including tracing and
    capability hooks. Useful for toolsets that need to dispatch tool calls
    programmatically (e.g. code execution sandboxes).

    Not available in `TemporalRunContext` — it is not serializable across
    Temporal activity boundaries.
    """

    capabilities: dict[str, AbstractCapability[RunContextAgentDepsT]] = field(default_factory=lambda: {})
    """All capabilities registered for the current run, including deferred ones."""

    loaded_capability_ids: set[str] = field(default_factory=set[str])
    """IDs of the deferred capabilities the model has explicitly loaded via `load_capability`.

    The capability-side mirror of `discovered_tool_names`: the runtime-revealed subset.
    Seeded during run preparation from message history (`parse_loaded_capabilities`); the
    `load_capability` tool body adds to it for in-step loads. Use `available_capability_ids`
    for the full set of currently-active capabilities (auto/always-on plus these).
    """

    capability_loaded: bool | None = None
    """Whether the capability whose hook is currently running is loaded.

    This is `None` outside capability hook dispatch, where there is no current capability.
    """

    discovered_tool_names: set[str] = field(default_factory=set[str])
    """Names of deferred tools revealed via tool-search return parts in the message history.

    The tool-side mirror of `loaded_capability_ids`: the runtime-revealed subset that
    `ToolSearchToolset.get_tools` reads to decide which deferred tools to make visible this
    turn. Populated during run preparation from message history. Use `available_tool_names`
    for the full set of currently-callable tools (always-visible plus these).
    """

    @property
    def last_attempt(self) -> bool:
        """Whether this is the last attempt at running this tool before an error is raised."""
        return self.retry == self.max_retries

    @property
    def available_capability_ids(self) -> set[str]:
        """IDs of the capabilities whose contributions are live to the model right now.

        The capability-side mirror of `available_tool_names`: `available = auto/always ∪
        runtime-revealed`. Here that's the non-deferred capabilities (`defer_loading` not
        `True`) plus the deferred ones the model has loaded (`loaded_capability_ids`), so
        `available_capability_ids - loaded_capability_ids` is the auto/always-on subset.

        Distinct from `capabilities`, the full registry (including deferred ones not yet
        loaded). See `loaded_capability_ids` for the runtime-revealed subset.
        """
        return {
            id for id, cap in self.capabilities.items() if cap.defer_loading is not True
        } | self.loaded_capability_ids

    @property
    def available_tool_names(self) -> set[str]:
        """Names of tools callable by the model on the current turn (non-deferred visible + discovered + loaded-capability).

        The subset of `tools` keys whose definitions are not deferred; the tool-side mirror
        of `available_capability_ids`: `available = auto/always ∪ runtime-revealed`, so
        `available_tool_names - discovered_tool_names` is the always-visible subset.
        """
        return {name for name, tool_def in self.tools.items() if not tool_def.defer_loading}

    @property
    def tools(self) -> dict[str, ToolDefinition]:
        """All tool definitions present this turn, keyed by name (includes still-deferred ones). Index `available_tool_names` into this for the callable subset."""
        if self.tool_manager is None or self.tool_manager.tools is None:
            return {}
        return {name: tool.tool_def for name, tool in self.tool_manager.tools.items()}

    __repr__ = _utils.dataclasses_no_defaults_repr


_CURRENT_RUN_CONTEXT: ContextVar[RunContext[Any] | None] = ContextVar(
    'pydantic_ai.current_run_context',
    default=None,
)
"""Context variable storing the current [`RunContext`][pydantic_ai.tools.RunContext]."""


def get_current_run_context() -> RunContext[Any] | None:
    """Get the current run context, if one is set.

    Returns:
        The current [`RunContext`][pydantic_ai.tools.RunContext], or `None` if not in an agent run.
    """
    return _CURRENT_RUN_CONTEXT.get()


@contextmanager
def set_current_run_context(run_context: RunContext[Any]) -> Iterator[None]:
    """Context manager to set the current run context.

    Args:
        run_context: The run context to set as current.

    Yields:
        None
    """
    token = _CURRENT_RUN_CONTEXT.set(run_context)
    try:
        yield
    finally:
        _CURRENT_RUN_CONTEXT.reset(token)
