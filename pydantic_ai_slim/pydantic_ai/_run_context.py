from __future__ import annotations as _annotations

import dataclasses
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import field
from typing import TYPE_CHECKING, Any, Generic

from opentelemetry.trace import NoOpTracer, Tracer
from typing_extensions import TypeVar

from pydantic_ai._instrumentation import DEFAULT_INSTRUMENTATION_VERSION

from . import _utils, messages as _messages
from ._enqueue import EnqueueContent, PendingMessage, PendingMessagePriority
from .exceptions import UserError

if TYPE_CHECKING:
    from .agent import Agent
    from .capabilities.abstract import AbstractCapability
    from .models import Model
    from .settings import ModelSettings
    from .tool_manager import ToolManager
    from .tools import ToolDefinition
    from .usage import RunUsage, UsageLimits

AgentDepsT = TypeVar('AgentDepsT', default=object, contravariant=True)
"""Type variable for agent dependencies."""

RunContextAgentDepsT = TypeVar('RunContextAgentDepsT', default=object, covariant=True)
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
    usage_limits: UsageLimits | None = None
    """The [`UsageLimits`][pydantic_ai.usage.UsageLimits] enforced for this run.

    During a run this is always set: if no limits were passed, the run enforces the default
    [`UsageLimits()`][pydantic_ai.usage.UsageLimits] (e.g. `request_limit=50`). It is only `None` on a
    bare/synthetic `RunContext` that isn't backed by a run.

    This reflects the limits the run is already enforcing, so tools and capabilities can disclose or
    adapt to the run's budget (e.g. a budget-disclosure capability) without having to be configured
    with a duplicate copy. Combine it with [`usage`][pydantic_ai.tools.RunContext.usage] to compute
    how much budget remains. Treat it as read-only: it is the live object the run enforces against, so
    mutating a field here *would* change what the run enforces on subsequent requests.
    """
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
    pending_messages: list[PendingMessage] | None = field(default=None, repr=False)
    """Queue read and mutated by the internal `PendingMessageDrainCapability`.

    Set to the run's live queue during an agent run; `None` in synthetic contexts that aren't
    backed by a running agent (e.g. the `RunContext` built by `Agent.system_prompt_parts`), where
    [`enqueue`][pydantic_ai.tools.RunContext.enqueue] would have nowhere to drain to and so raises.
    Managed by the framework: read it if useful, but use [`enqueue`][pydantic_ai.tools.RunContext.enqueue]
    to add messages rather than mutating it directly.
    """

    _event_stream_buffer: list[_messages.AgentStreamEvent] | None = field(default=None, repr=False)
    """Private implementation detail — not part of the public API; do not read or write.

    The run's shared event buffer (the same list held by `GraphAgentState`). Framework code appends
    events to it via [`_emit_event`][pydantic_ai._run_context.RunContext._emit_event]; the agent graph
    drains it into the agent event stream so consumers (`event_stream_handler`, `agent.run_stream_events`,
    `agent.iter` streaming) observe them. `None` in synthetic contexts not backed by a running agent.
    A public API for emitting custom events is intentionally not exposed yet.
    """

    _mcp_tool_defs_cache: dict[str, dict[str, ToolDefinition]] = field(default_factory=lambda: {}, repr=False)
    """Private implementation detail — not part of the public API; do not read or write.

    Per-run cache of MCP tool definitions, keyed by toolset `id`, read and written only by the
    durable-execution MCP toolset wrappers (Temporal/DBOS) so a toolset's tool definitions are
    fetched at most once per run rather than before every model request. It lives on the run —
    recreated for each agent run and reconstructed identically on durable replay/recovery — not on
    the process-shared toolset instance, so whether a wrapper schedules its `get_tools` activity/step
    depends only on the run's own history and stays replay-deterministic.
    """

    tool_manager: ToolManager[RunContextAgentDepsT] | None = None
    """The tool manager for the current run step.

    Provides access to tool validation and execution, including tracing and
    capability hooks. Useful for toolsets that need to dispatch tool calls
    programmatically (e.g. code execution sandboxes).

    Not available in `TemporalRunContext` — it is not serializable across
    Temporal activity boundaries.
    """

    root_capability: AbstractCapability[RunContextAgentDepsT] | None = None
    """The effective root capability for this run.

    Reflects the merged capability chain (agent-level + per-run extras) that
    is driving model requests, hooks, and toolsets for the current run.
    Capability implementations can use this to validate per-run additions
    (e.g. detect runtime-added capabilities that require worker registration).

    Not part of the Temporal activity-boundary serialization (capabilities
    don't round-trip), but populated on the activity side from the bound
    agent's `root_capability`.
    """

    capabilities: dict[str, AbstractCapability[RunContextAgentDepsT]] = field(default_factory=lambda: {})
    """All capabilities registered for the current run, including deferred ones."""

    loaded_capability_ids: set[str] = field(default_factory=set[str])
    """IDs of the deferred capabilities the model has explicitly loaded via the `load_capability` tool.

    The capability-side mirror of `discovered_tool_names`: the runtime-revealed subset.
    Seeded during run preparation from message history (`parse_loaded_capabilities`); the
    `load_capability` tool body adds to it for in-step loads. Use `available_capability_ids`
    for the full set of currently-active capabilities (auto/always-on plus these).
    Managed by the framework: safe to read, but don't mutate it directly.
    """

    capability_loaded: bool | None = None
    """Whether the capability whose hook or callback is currently running is loaded.

    This is `None` outside capability dispatch, where there is no current capability.
    """

    discovered_tool_names: set[str] = field(default_factory=set[str])
    """Names of deferred tools revealed via tool-search return parts in the message history.

    The tool-side mirror of `loaded_capability_ids`: the runtime-revealed subset that
    `ToolSearchToolset.get_tools` reads to decide which deferred tools to make visible this
    turn. Populated during run preparation from message history. Use `available_tool_names`
    for the full set of currently-callable tools (always-visible plus these).
    Managed by the framework: safe to read, but don't mutate it directly.
    """

    @property
    def last_attempt(self) -> bool:
        """Whether this is the last attempt at running this tool before an error is raised."""
        return self.retry == self.max_retries

    def _emit_event(self, event: _messages.AgentStreamEvent) -> None:
        """Append an event to the run's event buffer for the agent graph to drain into the event stream.

        Private framework plumbing — not public API. Only valid during an agent run, where the buffer
        is set (`_event_stream_buffer is not None`).
        """
        assert self._event_stream_buffer is not None, 'events are only emitted during an agent run, which has a buffer'
        self._event_stream_buffer.append(event)

    @property
    def available_capability_ids(self) -> set[str]:
        """IDs of the capabilities whose contributions are live to the model right now.

        The capability-side mirror of `available_tool_names`: `available = auto/always ∪
        runtime-revealed`. Here that's the non-deferred capabilities (`defer_loading` not
        `True`) plus the deferred ones the model has loaded (`loaded_capability_ids`), so
        `available_capability_ids - loaded_capability_ids` is the auto/always-on subset.

        Distinct from `capabilities`, the full registry (including deferred ones not yet
        loaded). See `loaded_capability_ids` for the runtime-revealed subset.

        Reliable from `before_run` onwards: the `capabilities` registry is seeded once at
        run start, and `loaded_capability_ids` is refreshed from history before each model
        request, so the loaded subset grows across steps as the model loads capabilities.
        Because it grows step by step, where you read it in the
        [hook order](../hooks.md#hook-ordering) determines what you see — e.g. a capability
        loaded during one step is not reflected until the next step's hooks.
        """
        return {
            id for id, cap in self.capabilities.items() if cap.defer_loading is not True
        } | self.loaded_capability_ids

    @property
    def available_tool_names(self) -> set[str]:
        """Names of function tools the model can call on the current turn.

        The visible subset of [`tools`][pydantic_ai.tools.RunContext.tools]: always-visible
        tools, tools revealed via [tool search](../tools-advanced.md#tool-search), and tools
        owned by loaded deferred capabilities.

        Only fully populated once the turn's tools have been resolved during model-request
        preparation, so it is reliable in model-request hooks (`before_model_request`,
        `wrap_model_request`, `after_model_request`) and tool hooks. In earlier hooks like
        `before_run` it falls back to `discovered_tool_names` (reconstructed from history).
        See [hook ordering](../hooks.md#hook-ordering) for how timing affects what you see.
        """
        if self.tool_manager is None or self.tool_manager.tools is None:
            return set[str]() | self.discovered_tool_names
        # Local import avoids a module-level cycle: `native_tools._tool_search` imports
        # `RunContext` for tool-search strategy callables.
        from .native_tools._tool_search import ToolSearchTool

        tools = self.tools
        # "Always available" = not search-managed AND not deferred. We deliberately keep the
        # `not defer_loading` check rather than relying on `with_native is None` alone: depending
        # on hook timing, a deferred tool can be read here before the tool-search toolset has
        # stamped `with_native='tool-search'` on it, so `with_native is None` by itself would leak
        # a still-hidden tool. Gating on `defer_loading` keeps it hidden until it's genuinely revealed.
        always_available = {
            name
            for name, tool_def in tools.items()
            if tool_def.with_native != ToolSearchTool.kind and not tool_def.defer_loading
        }
        runtime_revealed = self.discovered_tool_names & set(tools)
        loaded_capability_tools = {
            name
            for name, tool_def in tools.items()
            if tool_def.capability_id is not None and tool_def.capability_id in self.loaded_capability_ids
        }
        return always_available | runtime_revealed | loaded_capability_tools

    @property
    def tools(self) -> dict[str, ToolDefinition]:
        """All tool definitions present this turn, keyed by name (includes still-deferred ones). Index `available_tool_names` into this for the callable subset."""
        if self.tool_manager is None or self.tool_manager.tools is None:
            return {}
        return {name: tool.tool_def for name, tool in self.tool_manager.tools.items()}

    def enqueue(
        self,
        *content: EnqueueContent,
        priority: PendingMessagePriority = 'asap',
    ) -> str | None:
        """Enqueue content to be injected into the conversation.

        Safe to call from anywhere a `RunContext` is available — async tools,
        sync tools (auto-wrapped in a thread executor by Pydantic AI), and
        capability hooks. The drain only iterates the queue between graph nodes
        (in `before_model_request` and `after_node_run`), never concurrently
        with the tool body, so `list.append` from a worker thread doesn't race
        the drain.

        Args:
            *content: One or more [`EnqueueContent`][pydantic_ai.run.EnqueueContent] items.
                Adjacent [`UserContent`][pydantic_ai.messages.UserContent] (a `str` or multi-modal
                content like an [`ImageUrl`][pydantic_ai.messages.ImageUrl]) is gathered into one
                [`UserPromptPart`][pydantic_ai.messages.UserPromptPart], and each
                [`ModelRequestPart`][pydantic_ai.messages.ModelRequestPart] (e.g. a
                [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart]) is coalesced with adjacent
                part-style items into one [`ModelRequest`][pydantic_ai.messages.ModelRequest]; a complete
                [`ModelRequest`][pydantic_ai.messages.ModelRequest] or
                [`ModelResponse`][pydantic_ai.messages.ModelResponse] is kept as its own message. The
                assembled sequence must end in a request. Calling with no positional args is a no-op.
            priority: When to deliver:
                `'asap'` (default) — at the earliest opportunity (next model request,
                    or a redirect if the agent would otherwise end).
                `'when_idle'` — only when the agent would otherwise end, after `'asap'` messages.

        Returns:
            The `enqueue_id` of the queued message, echoed on the
            [`EnqueuedMessagesEvent`][pydantic_ai.messages.EnqueuedMessagesEvent] emitted when it's
            delivered, or `None` when there was nothing to enqueue (an empty call).

        Raises:
            UserError: If this `RunContext` isn't backed by a running agent's queue (e.g. the
                synthetic context from `Agent.system_prompt_parts`), since there'd be nowhere
                to deliver the message.
        """
        if self.pending_messages is None:
            raise UserError(
                '`enqueue` is only available during an agent run (from tools, capability hooks, or '
                '`AgentRun.enqueue`). This `RunContext` has no pending-message queue to drain.'
            )
        pending = PendingMessage.from_content(*content, priority=priority)
        if pending is None:
            return None
        self.pending_messages.append(pending)
        return pending.enqueue_id

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
def set_current_run_context(run_context: RunContext[Any]) -> Generator[None]:
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
