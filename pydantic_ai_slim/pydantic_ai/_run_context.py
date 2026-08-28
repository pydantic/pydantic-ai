from __future__ import annotations as _annotations

import dataclasses
import sys
import warnings
from collections.abc import Awaitable, Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import field
from functools import wraps
from typing import TYPE_CHECKING, Any, Generic

from opentelemetry.trace import NoOpTracer, Tracer
from typing_extensions import TypeVar, deprecated

from pydantic_ai._instrumentation import DEFAULT_INSTRUMENTATION_VERSION

from . import _utils, messages as _messages
from ._enqueue import EnqueueContent, PendingMessage, PendingMessagePriority
from ._warnings import PydanticAIDeprecationWarning
from .exceptions import UserError

if TYPE_CHECKING:
    from ._cancel import RunCancellation
    from .agent import Agent
    from .capabilities.abstract import AbstractCapability
    from .models import AbstractModel
    from .realtime import RealtimeModelSettings, RealtimeSession
    from .settings import ModelSettings
    from .tool_manager import ToolManager
    from .tools import ToolDefinition
    from .usage import RunUsage, UsageLimits

AgentDepsT = TypeVar('AgentDepsT', default=object, contravariant=True)
"""Type variable for agent dependencies."""

RunContextAgentDepsT = TypeVar('RunContextAgentDepsT', default=object, covariant=True)
"""Type variable for the agent dependencies in `RunContext`."""


@dataclasses.dataclass(frozen=True)
class AnchoredEvidence:
    """Reveal and load evidence the provider that served a response could still see.

    `RunContext.discovered_tool_names` and `loaded_capability_ids` are cut at any `CompactionPart`,
    because the consumer that matters for them is the *next* request, whose provider isn't knowable
    when history is parsed. A call the model already made is a different question with a different
    answer: the response records which provider served it, so a boundary that provider would have
    skipped on the wire — another provider's, or one whose payload it doesn't render — hid nothing
    from it. This holds what those parts of history still evidence.

    Additive, never a replacement: the sets it widens are shared mutable run state that tool
    execution writes in-step reveals into, so the widened view has to be a separate object.
    """

    discovered_tool_names: frozenset[str] = frozenset()
    """Deferred tools revealed inside the anchored window but not in `discovered_tool_names`."""

    loaded_capability_ids: frozenset[str] = frozenset()
    """Capabilities loaded inside the anchored window but not in `loaded_capability_ids`."""


@dataclasses.dataclass(repr=False, kw_only=True)
class RunContext(Generic[RunContextAgentDepsT]):
    """Information about the current call."""

    deps: RunContextAgentDepsT
    """Dependencies for the agent."""
    model: AbstractModel
    """The active model, which is a `RealtimeModel` during a realtime session."""
    usage: RunUsage
    """LLM usage associated with the run."""
    _model_id: str | None = field(default=None, repr=False)
    """The model selection token used to resolve `model`, for internal durable transport."""
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
    model_settings: ModelSettings | RealtimeModelSettings | None = None
    """The resolved model settings for the current run step.

    Populated before each model request, after all model settings layers
    (model defaults, agent-level, capability, and run-level) have been merged.
    Available in model request hooks (`before_model_request`, `wrap_model_request`,
    `after_model_request`). Currently `None` in tool hooks, output validators,
    and during agent construction.

    During a realtime session this holds the merged
    [`RealtimeModelSettings`][pydantic_ai.realtime.RealtimeModelSettings] the session was opened
    with, for the whole session (realtime settings are fixed at connect time).
    """
    pending_messages: list[PendingMessage] | None = field(default=None, repr=False)
    """Queue read and mutated by the internal `PendingMessageDrainCapability`.

    Set to the run's live queue during an agent run; `None` in synthetic contexts that aren't
    backed by a running agent (e.g. the `RunContext` built by `Agent.system_prompt_parts`), where
    [`enqueue`][pydantic_ai.tools.RunContext.enqueue] would have nowhere to drain to and so raises.
    Managed by the framework: read it if useful, but use [`enqueue`][pydantic_ai.tools.RunContext.enqueue]
    to add messages rather than mutating it directly.
    """

    _cancellation: RunCancellation | None = field(default=None, repr=False)
    """Private implementation detail — not part of the public API; do not read or write.

    The run's cancellation controller, used by [`cancel`][pydantic_ai.tools.RunContext.cancel].
    Holds a live task reference, so it is runtime-only: `None` in synthetic contexts that aren't
    backed by a running agent, and not available across durable-execution serialization boundaries
    (e.g. inside a Temporal activity).
    """

    _event_stream_buffer: list[_messages.AgentStreamEvent] | None = field(default=None, repr=False)
    """Private implementation detail — not part of the public API; do not read or write.

    The run's shared event buffer (the same list held by `GraphAgentState`). Framework code appends
    events to it via [`_emit_event`][pydantic_ai._run_context.RunContext._emit_event]; the agent graph
    drains it into the agent event stream so consumers (`event_stream_handler`, `agent.run_stream_events`,
    `agent.iter` streaming) observe them. `None` in synthetic contexts not backed by a running agent.
    A public API for emitting custom events is intentionally not exposed yet.
    """

    _durable_operations: dict[tuple[str, str], Callable[..., Awaitable[Any]]] | None = field(default=None, repr=False)
    """Per-run durable capability operation dispatchers, for internal use only."""

    _run_capabilities_by_id: dict[str, AbstractCapability[Any]] | None = field(default=None, repr=False)
    """Per-run capability instances used for durable recovery, for internal use only."""

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

    realtime_session: RealtimeSession | None = field(default=None, repr=False)
    """The [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] this run is, once it is connected.

    `None` in classic runs, and during the parts of a realtime run that precede the connection:
    `before_run`, `wrap_run` before `handler()` starts the session, and instruction resolution.
    Use [`realtime`][pydantic_ai.tools.RunContext.realtime] to detect a realtime run in those
    stages. Tools and hooks that run during the live session can use it to e.g.
    [`interrupt()`][pydantic_ai.realtime.RealtimeSession.interrupt] playback or
    [`send()`][pydantic_ai.realtime.RealtimeSession.send] follow-up content.
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
    Derived from message history (`parse_loaded_capabilities`) before each request, so a capability
    loaded during a step appears from the *next* one — the same step that first carries its
    instructions to the model, and therefore the first on which its tools can be called. Use
    `active_capability_ids` for the full set of currently-active capabilities (auto/always-on
    plus these). Managed by the framework: safe to read, but don't mutate it directly.
    """

    capability_active: bool | None = None
    """Whether the capability whose hook or callback is currently running is active right now.

    *Active*, not *available* and not *loaded*: see
    [`active_capability_ids`][pydantic_ai.tools.RunContext.active_capability_ids] for why
    capabilities use "active" while tools use "available".

    An always-on capability is active for the whole run, so this reads `True` inside its hooks
    although nothing ever loaded it. A deferred capability has to be loaded before it becomes
    active, and its hooks are skipped until then — so it reads `True` there too. What it answers is
    "may this capability act now?", not "was it selected?"; for the latter, look an id up in
    [`loaded_capability_ids`][pydantic_ai.tools.RunContext.loaded_capability_ids].

    This is `None` outside capability dispatch, where there is no current capability.
    """

    discovered_tool_names: set[str] = field(default_factory=set[str])
    """Names of deferred function tools named by durable message history.

    Raw evidence, not a verdict: it collects every name tool-search returns and
    `ToolAvailabilityDeltaPart`s mention — including deltas from any tool's `ToolReturn.tools` and
    from `load_capability` — without checking that the tool still exists or that its owner is
    loaded. Read by `is_tool_available` and the reveal builders, which apply those checks.
    Populated during run preparation from message history. Use `available_tool_names` for the full
    set of currently-callable tools (always-visible plus these).
    Managed by the framework: safe to read, but don't mutate it directly.
    """

    _anchored_evidence: AnchoredEvidence = field(default_factory=lambda: AnchoredEvidence(), repr=False)
    """Evidence the serving provider could still see that the conservative window dropped.

    Set at tool-call dispatch and read only by `is_tool_available`. Private because the sets above
    stay the answer for everything that feeds a *future* request, whose provider isn't knowable yet;
    this one is the answer for a call the model has already made, where it is. See `AnchoredEvidence`.
    """

    @property
    def model_id(self) -> str | None:
        """The identifier from which the run's active `model` was resolved.

        This is `None` when the model was passed as an instance instead of being resolved from an
        identifier. The property is read-only; Pydantic AI manages the selection token internally.
        """
        return self._model_id

    @property
    @deprecated(
        '`capability_loaded` is deprecated, use `capability_active` instead: the value is `True` for an '
        'always-on capability that was never loaded.',
        category=PydanticAIDeprecationWarning,
    )
    def capability_loaded(self) -> bool | None:
        """Whether the capability whose hook or callback is currently running is active right now.

        Deprecated: use [`capability_active`][pydantic_ai.tools.RunContext.capability_active]. This
        never meant "loaded" — it is `True` for an always-on capability nothing ever loaded.
        """
        return self.capability_active

    @capability_loaded.setter
    @deprecated(
        '`capability_loaded` is deprecated, use `capability_active` instead: the value is `True` for an '
        'always-on capability that was never loaded.',
        category=PydanticAIDeprecationWarning,
    )
    def capability_loaded(self, value: bool | None) -> None:
        # A plain dataclass field until this rename, so assignment used to work; a read-only property
        # would turn that into an `AttributeError` at runtime rather than a deprecation.
        self.capability_active = value

    @property
    def realtime(self) -> bool:
        """Whether this run is a realtime session, i.e. `model` is the connected `RealtimeModel`.

        Reliable from `before_run` through session close, including instruction resolution — unlike
        [`realtime_session`][pydantic_ai.tools.RunContext.realtime_session], which is only set once
        the session is connected. The class is looked up through `sys.modules` rather than imported:
        if the realtime package was never imported, no realtime model can exist, and a classic run
        should not pay for (or cycle into) that import.
        """
        realtime = sys.modules.get('pydantic_ai.realtime')
        return realtime is not None and isinstance(self.model, realtime.RealtimeModel)

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
    def active_capability_ids(self) -> set[str]:
        """IDs of the capabilities whose contributions are live to the model right now.

        *Active*, deliberately not *available*: a capability is not something the model calls, so
        "available" would read as "offered in the catalog, there for the loading" — which is the
        opposite set, the deferred ones that are **not** yet contributing. Active means the
        capability's instructions, tools, settings and hooks are in force on this step:
        non-deferred capabilities (`defer_loading` not `True`) plus the deferred ones the model has
        loaded, so `active_capability_ids - loaded_capability_ids` is the auto/always-on subset.

        Tools keep the word *available* because for them there is only one question — may the model
        call this now? — and no catalog sense to collide with. So `is_tool_available` reads "revealed,
        and its owning capability is active".

        Two axes, deliberately not mixed. *Configuration* is set once by the author: a capability is
        either **deferred** (`defer_loading=True`) or **always-on**. *Runtime* is derived per step:
        **loaded** records what the model asked for, **active** what is in force. So "always-on" is
        the antonym of "deferred", never of "active" — an always-on capability is always active, and
        a deferred one becomes active once loaded.

        Distinct from `capabilities`, the full registry (including deferred ones not yet
        loaded). See `loaded_capability_ids` for the subset the model explicitly loaded.

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
    @deprecated(
        '`available_capability_ids` is deprecated, use `active_capability_ids` instead: for a '
        'capability, "available" reads as "there for the loading", which is the opposite set.',
        category=PydanticAIDeprecationWarning,
    )
    def available_capability_ids(self) -> set[str]:
        """IDs of the capabilities whose contributions are live to the model right now.

        Deprecated: use [`active_capability_ids`][pydantic_ai.tools.RunContext.active_capability_ids].
        """
        return self.active_capability_ids

    @property
    def _deferred_capability_ids(self) -> set[str]:
        """IDs of the capabilities configured to load on demand.

        Private, and read only by `is_tool_available`, which needs the *configured* shape rather
        than the runtime one: `loaded_capability_ids` records what history says was loaded, which
        can name a capability that has since been reconfigured as always-on. Overridden in
        `TemporalRunContext` with the snapshot serialized at activity dispatch, since the
        `capabilities` registry this reads does not cross that boundary.
        """
        return {id for id, cap in self.capabilities.items() if cap.defer_loading is True}

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
        return {name for name, tool_def in self.tools.items() if self.is_tool_available(tool_def)}

    def is_tool_available(self, tool: str | ToolDefinition) -> bool:
        """Whether a function tool is currently available to the model.

        Pass a [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] when checking a definition
        held by a toolset, especially inside `get_tools`. This form evaluates the definition's
        own fields against the reveal state recorded in history, so it remains
        reliable when a wrapping toolset has removed the definition from the resolved tool set.

        Pass a tool name where [`tools`][pydantic_ai.tools.RunContext.tools] is reliable, such as
        model-request hooks or ordinary tool execution. The name form looks up the current definition
        in `tools`; when live tool state is unavailable (including inside a Temporal activity), it
        falls back to `available_tool_names`. An unknown name returns `False`. See
        [`available_tool_names`][pydantic_ai.tools.RunContext.available_tool_names] for the timing
        caveat, and [`ModelRequestParameters.revealed_tool_names`][pydantic_ai.models.ModelRequestParameters.revealed_tool_names]
        for the reveal state sent through the model-request pipeline.
        """
        if isinstance(tool, str):
            if self.tool_manager is None or self.tool_manager.tools is None:
                # Same live-state condition as `available_tool_names`: mid-`get_tools` the
                # manager exists but its tool set isn't resolved yet, so fall back to history.
                return tool in self.available_tool_names
            tool_def = self.tools.get(tool)
            if tool_def is None:
                return False
        else:
            tool_def = tool

        # Local import avoids a module-level cycle: `native_tools._tool_search` imports
        # `RunContext` for tool-search strategy callables.
        from .native_tools._tool_search import ToolSearchTool

        # "Always available" deliberately checks `defer_loading`, not only `with_native`: a deferred
        # definition can be observed before tool search stamps `with_native='tool-search'` on it.
        if tool_def.with_native != ToolSearchTool.kind and not tool_def.defer_loading:
            return True
        capability_id = tool_def.capability_id
        # Loading a deferred capability discloses its tools as a bundle — the load exchange carries
        # the instructions *and* the schemas — so for its own tools the load already is the reveal.
        # Demanding a separate reveal marker on top would strand a tool permanently: history
        # processing can drop the reveal while keeping the load, and from there the model has no way
        # back, because a capability-owned tool is not in the search corpus and reloading an
        # already-active capability is refused.
        #
        # Both halves are load-bearing. The capability must still be *configured* deferred, not just
        # named by a load record in history: a capability that has since been reconfigured as
        # always-on never announced its tools as a bundle, so a stale record must not reveal them.
        evidence = self._anchored_evidence
        if (
            capability_id is not None
            and capability_id in self._deferred_capability_ids
            and capability_id in self.loaded_capability_ids | evidence.loaded_capability_ids
        ):
            return capability_id in self.active_capability_ids | evidence.loaded_capability_ids
        if tool_def.name not in self.discovered_tool_names | evidence.discovered_tool_names:
            return False
        # A run holds to load, then reveal, then call. `discovered_tool_names` is raw history
        # evidence and only answers the middle step, so it can name a tool whose capability was
        # never loaded — a history no real run produces, and one that would skip the instructions
        # written to be read first. Checking the owner here keeps this predicate in step with what
        # `ToolManager` will run, so "available" means one thing everywhere it is asked.
        return capability_id is None or capability_id in (self.active_capability_ids | evidence.loaded_capability_ids)

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
                    or a redirect if the agent would otherwise end). In a realtime session, an active
                    assistant response is allowed to finish before the content is sent; otherwise it
                    is sent immediately.
                `'when_idle'` — only when the agent would otherwise end, after `'asap'` messages.
                    In a realtime session, this means after the next response completes.

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

    def cancel(self) -> None:
        """Cancel the agent run this context belongs to.

        Safe to call from anywhere a `RunContext` is available — tools, `event_stream_handler`s,
        and capability hooks. This *requests* cancellation: it returns normally, and the calling
        code keeps running until its next `await`, where the cancellation is delivered — so the
        caller can still do cleanup, but its return value (e.g. a tool's result) is discarded. The
        run then stops what it is doing (the in-flight model request is torn down, sibling tool
        tasks are cancelled and drained, a suspended server-side job is best-effort cancelled) and
        ends with [`RunCancelled`][pydantic_ai.exceptions.RunCancelled], preserving everything that
        completed before the cancellation took effect in message history. Idempotent; a no-op once
        the run has finished. Cancellation is terminal: capability hooks may observe it and clean
        up, but cannot recover the run to success. Cancellation cannot forcibly stop synchronous
        code running in a worker thread; it may continue and perform side effects, although its
        result is discarded.

        Raises:
            UserError: If this `RunContext` isn't backed by a running agent (e.g. the synthetic
                context from `Agent.system_prompt_parts`, or across a durable-execution
                serialization boundary such as a Temporal activity).
        """
        # Read via `__dict__` because `TemporalRunContext.__getattribute__` raises a
        # serialize-it-yourself `UserError` for absent fields, which would be misleading here:
        # the controller holds a live task reference and can never cross an activity boundary.
        cancellation: RunCancellation | None = self.__dict__.get('_cancellation')
        if cancellation is None:
            raise UserError(
                '`cancel` is only available during an agent run (from tools, event stream handlers, '
                'or capability hooks) in the same process as the run itself. '
                'This `RunContext` has no run to cancel.'
            )
        cancellation.cancel()

    __repr__ = _utils.dataclasses_no_defaults_repr


_run_context_init = RunContext.__init__


@wraps(_run_context_init)
def _run_context_init_with_capability_loaded(
    self: RunContext[Any], *, capability_loaded: bool | None = None, **kwargs: Any
) -> None:
    if capability_loaded is not None:
        warnings.warn(
            '`capability_loaded` is deprecated, use `capability_active` instead: the value is `True` for an '
            'always-on capability that was never loaded.',
            PydanticAIDeprecationWarning,
            stacklevel=2,
        )
        kwargs.setdefault('capability_active', capability_loaded)
    _run_context_init(self, **kwargs)


# Wrapping the generated `__init__` rather than keeping an `InitVar` field: on Python 3.13+
# `dataclasses.replace()` round-trips every init-only variable through `getattr`, which would fire
# the deprecation warning on each of the run's internal `replace(ctx, ...)` calls. A non-field
# keyword is invisible to `replace()`, and `@wraps` keeps `inspect.signature` resolving to the real
# one. `TemporalRunContext` defines its own `__init__` and is unaffected either way.
RunContext.__init__ = _run_context_init_with_capability_loaded


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
