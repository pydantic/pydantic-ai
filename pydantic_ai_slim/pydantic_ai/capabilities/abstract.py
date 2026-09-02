from __future__ import annotations

from abc import ABC
from collections import Counter
from collections.abc import AsyncIterable, Awaitable, Callable, Sequence
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, TypeAlias
from weakref import WeakValueDictionary

from pydantic import ValidationError
from typing_extensions import deprecated

from pydantic_ai import _utils
from pydantic_ai._instructions import (
    AgentInstruction,
    AgentInstructions,
    SourcedInstruction,
    normalize_instructions,
    sourced_instruction,
)
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import (
    AgentStreamEvent,
    CapabilityInstructionSource,
    ModelResponse,
    ToolCallPart,
)
from pydantic_ai.tools import (
    AgentDepsT,
    AgentNativeTool,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    SystemPromptFunc,
    ToolDefinition,
)
from pydantic_ai.toolsets import AbstractToolset, AgentToolset

if TYPE_CHECKING:
    from pydantic_ai import _agent_graph
    from pydantic_ai.agent.abstract import AbstractAgent, AgentModelSettings
    from pydantic_ai.capabilities.prefix_tools import PrefixTools
    from pydantic_ai.models import (
        KnownModelName,
        Model,
        ModelRequestContext,
        ModelResolutionContext,
        ModelSelectionContext,
    )
    from pydantic_ai.output import OutputContext
    from pydantic_ai.result import FinalResult
    from pydantic_ai.run import AgentRunResult
    from pydantic_graph import End

# --- Handler type aliases for use in hook method signatures ---
# These make it easier to write correct type annotations when subclassing AbstractCapability.

AgentNode: TypeAlias = '_agent_graph.AgentNode[AgentDepsT, Any]'
"""Type alias for an agent graph node (`UserPromptNode`, `ModelRequestNode`, `CallToolsNode`)."""

NodeResult: TypeAlias = '_agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]'
"""Type alias for the result of executing an agent graph node: either the next node or `End`."""

WrapRunHandler: TypeAlias = 'Callable[[], Awaitable[AgentRunResult[Any]]]'
"""Handler type for [`wrap_run`][pydantic_ai.capabilities.AbstractCapability.wrap_run]."""

WrapNodeRunHandler: TypeAlias = 'Callable[[_agent_graph.AgentNode[AgentDepsT, Any]], Awaitable[_agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]]]'
"""Handler type for [`wrap_node_run`][pydantic_ai.capabilities.AbstractCapability.wrap_node_run]."""

WrapModelRequestHandler: TypeAlias = 'Callable[[ModelRequestContext], Awaitable[ModelResponse]]'
"""Handler type for [`wrap_model_request`][pydantic_ai.capabilities.AbstractCapability.wrap_model_request]."""

ModelSelection: TypeAlias = 'Model | KnownModelName | str'
"""A concrete model selection, before model ID resolution."""

ModelSelector: TypeAlias = 'Callable[[ModelSelectionContext[AgentDepsT]], ModelSelection | Awaitable[ModelSelection]]'
"""A sync or async per-step model selector."""

AgentModel: TypeAlias = 'ModelSelection | ModelSelector[AgentDepsT]'
"""A static model selection or a callable evaluated for every request step."""

RawToolArgs: TypeAlias = str | dict[str, Any]
"""Type alias for raw (pre-validation) tool arguments."""

ValidatedToolArgs: TypeAlias = dict[str, Any]
"""Type alias for validated tool arguments."""

WrapToolValidateHandler: TypeAlias = Callable[[RawToolArgs], Awaitable[ValidatedToolArgs]]
"""Handler type for [`wrap_tool_validate`][pydantic_ai.capabilities.AbstractCapability.wrap_tool_validate]."""

WrapToolExecuteHandler: TypeAlias = Callable[[ValidatedToolArgs], Awaitable[Any]]
"""Handler type for [`wrap_tool_execute`][pydantic_ai.capabilities.AbstractCapability.wrap_tool_execute]."""

RawOutput: TypeAlias = str | dict[str, Any]
"""Type alias for raw output data (text or tool args)."""

DurableOperationDispatcher: TypeAlias = Callable[
    [RunContext[object], tuple[object, ...], dict[str, object]], Awaitable[object]
]

WrapOutputValidateHandler: TypeAlias = Callable[[RawOutput], Awaitable[Any]]
"""Handler type for wrap_output_validate."""

WrapOutputProcessHandler: TypeAlias = Callable[[Any], Awaitable[Any]]
"""Handler type for wrap_output_process."""

CapabilityPosition = Literal['outermost', 'innermost']
"""Position tier for a capability in the middleware chain.

- `'outermost'`: in the outermost tier, before all non-outermost capabilities.
  Multiple capabilities can declare `'outermost'`; original list order breaks ties
  within the tier, and `wraps`/`wrapped_by` edges refine order further.
- `'innermost'`: in the innermost tier, after all non-innermost capabilities.
  Same tie-breaking rules apply.
"""

CapabilityRef: TypeAlias = 'type[AbstractCapability[Any]] | AbstractCapability[Any]'
"""Reference to a capability — either a type (matches all instances of that type) or a specific instance (matches by identity)."""


CapabilityDescription = str | SystemPromptFunc[AgentDepsT]
"""Capability description: a static string, or a function (sync/async, with or without
[`RunContext`][pydantic_ai.tools.RunContext]) that returns one.

For dynamic descriptions, return a callable from
[`get_description`][pydantic_ai.capabilities.AbstractCapability.get_description] rather than
having the method itself take `RunContext`.
"""


@dataclass
class CapabilityOrdering:
    """Ordering constraints for a capability within a combined capability chain.

    Capabilities follow middleware semantics: the first capability in the list is the
    **outermost** layer, wrapping all others. Declare ordering constraints via
    [`get_ordering`][pydantic_ai.capabilities.AbstractCapability.get_ordering]
    to control a capability's position in the chain regardless of how the user lists them.

    When a [`CombinedCapability`][pydantic_ai.capabilities.CombinedCapability] is
    constructed, it topologically sorts its children to satisfy these constraints,
    preserving user-provided order as a tiebreaker.
    """

    position: CapabilityPosition | None = None
    """Fixed position in the chain, or `None` for user-provided order."""

    wraps: Sequence[CapabilityRef] = ()
    """This capability wraps around (is outside of) these capabilities in the middleware chain.

    Each entry can be a capability **type** (matches all instances of that type via `issubclass`)
    or a specific capability **instance** (matches by identity via `is`).

    Note: instance refs use identity (`is`) matching, so if a capability's
    [`for_run`][pydantic_ai.capabilities.AbstractCapability.for_run] returns a
    new instance, refs to the original will no longer match. Use type refs
    when the target capability uses per-run state isolation.
    """

    wrapped_by: Sequence[CapabilityRef] = ()
    """This capability is wrapped by (is inside of) these capabilities in the middleware chain.

    Each entry can be a capability **type** (matches all instances of that type via `issubclass`)
    or a specific capability **instance** (matches by identity via `is`).

    Note: instance refs use identity (`is`) matching, so if a capability's
    [`for_run`][pydantic_ai.capabilities.AbstractCapability.for_run] returns a
    new instance, refs to the original will no longer match. Use type refs
    when the target capability uses per-run state isolation.
    """

    requires: Sequence[type[AbstractCapability[Any]]] = ()
    """These types must be present in the chain (no ordering implied)."""


class _DurableOperationBindings:
    """Agent-identity bindings that do not retain unhashable agent instances."""

    def __init__(self) -> None:
        self._agents: WeakValueDictionary[int, AbstractAgent[Any, Any]] = WeakValueDictionary()
        self._bindings: dict[int, dict[str, DurableOperationDispatcher]] = {}

    def get(
        self, agent: AbstractAgent[Any, Any], default: dict[str, DurableOperationDispatcher]
    ) -> dict[str, DurableOperationDispatcher]:
        self._prune()
        agent_id = id(agent)
        return self._bindings.get(agent_id, default) if self._agents.get(agent_id) is agent else default

    def setdefault(self, agent: AbstractAgent[Any, Any]) -> dict[str, DurableOperationDispatcher]:
        self._prune()
        agent_id = id(agent)
        if self._agents.get(agent_id) is not agent:
            self._agents[agent_id] = agent
            self._bindings[agent_id] = {}
        return self._bindings[agent_id]

    def __len__(self) -> int:
        self._prune()
        return len(self._bindings)

    def _prune(self) -> None:
        live_ids = set(self._agents)
        self._bindings = {agent_id: bindings for agent_id, bindings in self._bindings.items() if agent_id in live_ids}


@dataclass(init=False)
class AbstractCapability(ABC, Generic[AgentDepsT]):
    """Abstract base class for agent capabilities.

    A capability is a reusable, composable unit of agent behavior that can provide
    instructions, model settings, tools, and request/response hooks.

    Lifecycle: capabilities are passed to an [`Agent`][pydantic_ai.Agent] at construction time, where
    most `get_*` methods are called to collect static configuration (instructions, model
    settings, toolsets, native tools). When [`for_run`][pydantic_ai.capabilities.AbstractCapability.for_run]
    returns a replacement instance, that configuration is re-extracted from the replacement at run
    setup. The exception is
    [`get_wrapper_toolset`][pydantic_ai.capabilities.AbstractCapability.get_wrapper_toolset],
    which is always called per-run during toolset assembly. Then, on each model request during a
    run, the [`before_model_request`][pydantic_ai.capabilities.AbstractCapability.before_model_request]
    and [`after_model_request`][pydantic_ai.capabilities.AbstractCapability.after_model_request]
    hooks are called to allow dynamic adjustments.

    See the [capabilities documentation](../capabilities/overview.md) for built-in capabilities.

    [`get_serialization_name`][pydantic_ai.capabilities.AbstractCapability.get_serialization_name]
    and [`from_spec`][pydantic_ai.capabilities.AbstractCapability.from_spec] support
    YAML/JSON specs (via `Agent.from_spec`); they have
    sensible defaults and typically don't need to be overridden.
    """

    def _get_durable_operation_bindings(
        self,
    ) -> _DurableOperationBindings:
        bindings = self.__dict__.get('_pydantic_ai_durable_operation_bindings')
        if not isinstance(bindings, _DurableOperationBindings):
            bindings = _DurableOperationBindings()
            object.__setattr__(self, '_pydantic_ai_durable_operation_bindings', bindings)
        return bindings

    _safe_at_runtime: ClassVar[bool] = False
    """Whether this capability can be added per-run when a durability capability is bound.

    Internal, in-tree only. [`Instrumentation`][pydantic_ai.capabilities.Instrumentation]
    is the only built-in capability that sets this to `True`; the bundled `durable_exec`
    integrations read it to allow `Instrumentation` to attach per-run despite the
    blanket restriction on runtime capability additions.

    A first-class extension point that derives this from a capability's overridden
    hooks (so third-party capabilities don't need to set a flag manually) is tracked
    in [#5477](https://github.com/pydantic/pydantic-ai/issues/5477).
    """

    _: KW_ONLY

    id: str | None = None
    """Optional identifier used to reference this capability within a run.

    Must be unique within a run, not per instance: it identifies the capability across the
    run — including the fresh instance a [`for_run`][pydantic_ai.capabilities.AbstractCapability.for_run]
    override may return — rather than a specific object.

    Required when `defer_loading=True`. If omitted for an always-on
    capability, the run derives a local id from the class name.
    """

    description: str | None = None
    """Description of the capability."""

    defer_loading: bool = False
    """If True, model-facing tools and instructions are hidden until the model explicitly
    loads the capability via the `load_capability` tool.

    Model settings and lifecycle hooks are registered during run setup, but only
    apply or fire once the capability is loaded.

    Requires a stable [`id`][pydantic_ai.capabilities.AbstractCapability.id] so
    message history can identify the capability. A
    [`description`][pydantic_ai.capabilities.AbstractCapability.description] or
    [`get_description`][pydantic_ai.capabilities.AbstractCapability.get_description]
    override is optional and only adds routing context to the load catalog.
    """

    def apply(self, visitor: Callable[[AbstractCapability[AgentDepsT]], None]) -> None:
        """Run a visitor function on all leaf capabilities in this tree.

        For a single capability, calls the visitor on itself.
        Overridden by [`CombinedCapability`][pydantic_ai.capabilities.CombinedCapability]
        to recursively visit all child capabilities.
        """
        visitor(self)

    @property
    @deprecated(
        '`has_wrap_node_run` is deprecated: `wrap_node_run` now runs under every way of driving a run, '
        'so there is nothing left to test for.',
        category=PydanticAIDeprecationWarning,
    )
    def has_wrap_node_run(self) -> bool:
        """Whether this capability (or any sub-capability) overrides wrap_node_run.

        Deprecated: `wrap_node_run` runs under every way of driving a run, so there is nothing left to test for.
        """
        return self._has_wrap_node_run

    @property
    def _has_wrap_node_run(self) -> bool:
        return type(self).wrap_node_run is not AbstractCapability.wrap_node_run

    @property
    def _has_on_node_run_error(self) -> bool:
        return type(self).on_node_run_error is not AbstractCapability.on_node_run_error

    @property
    def _has_wrap_model_request(self) -> bool:
        return type(self).wrap_model_request is not AbstractCapability.wrap_model_request

    @property
    def _has_on_model_request_error(self) -> bool:
        return type(self).on_model_request_error is not AbstractCapability.on_model_request_error

    @property
    def has_wrap_run_event_stream(self) -> bool:
        """Whether this capability (or any sub-capability) overrides wrap_run_event_stream."""
        return type(self).wrap_run_event_stream is not AbstractCapability.wrap_run_event_stream

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Return the name used for spec serialization (CamelCase class name by default).

        Return None to opt out of spec-based construction.
        """
        return cls.__name__

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> AbstractCapability[Any]:
        """Create from spec arguments. Default: `cls(*args, **kwargs)`.

        Override when `__init__` takes non-serializable types.
        """
        return cls(*args, **kwargs)

    def get_ordering(self) -> CapabilityOrdering | None:
        """Return ordering constraints for this capability, or `None` for default behavior.

        Override to declare a fixed position (`'outermost'` / `'innermost'`),
        relative ordering (`wraps` / `wrapped_by` other capability types or instances),
        or dependency requirements (`requires`).

        [`CombinedCapability`][pydantic_ai.capabilities.CombinedCapability] uses
        these to topologically sort its children at construction time.
        """
        return None

    def for_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> AbstractCapability[AgentDepsT]:
        """Return the capability instance to use with an agent.

        Called after the agent's own configuration is available and before capability
        contributions are extracted. Constructor capabilities are bound once during agent
        construction; static run capabilities are bound once per run. Override this to inspect
        the agent and return an agent-bound copy. The default returns `self`.

        A [`CapabilityFunc`][pydantic_ai.capabilities.CapabilityFunc] result is also bound before
        its own [`for_run`][pydantic_ai.capabilities.AbstractCapability.for_run] hook. A specialized
        run-bound value returned by an ordinary capability's `for_run()` is not bound again.

        Capabilities in the `innermost` ordering tier (see
        [`get_ordering`][pydantic_ai.capabilities.AbstractCapability.get_ordering]), i.e. durability
        capabilities, bind in a second phase, after the other capabilities' contributed toolsets have
        been extracted, so `agent.toolsets` is complete when their `for_agent` wraps it. The flip side
        is that `innermost` capabilities can't contribute toolsets of their own.
        """
        return self

    def _prepare_run_context(self, ctx: RunContext[AgentDepsT]) -> None:
        """Install private per-run state before any capability lifecycle hook runs.

        Durable dispatch tables must be available before every capability's `before_run`, because
        one capability may call another capability's durable operation from its hook. This setup
        therefore cannot be implemented as a `before_run` hook itself. It stays private pending the
        capability surface decisions tracked in #5477.
        """

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        """Return the capability instance to use for this agent run.

        Called once per run, before `get_*()` re-extraction and before any hooks fire.
        Override to return a fresh instance for per-run state isolation.
        Under durable execution, worker processes re-derive this instance from the deserialized
        run context, so all per-run state must be derivable from `ctx`.
        Default: return `self` (shared across runs).
        """
        return self

    def _validate_runtime_capabilities(
        self, ctx: RunContext[AgentDepsT], capabilities: Sequence[AbstractCapability[AgentDepsT]]
    ) -> None:
        """Validate capabilities contributed specifically for this run.

        Deliberately private: whether this becomes part of the public runtime extension
        surface (and in what shape) will be decided as part of
        [#5477](https://github.com/pydantic/pydantic-ai/issues/5477).
        """

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Return instructions to include in the system prompt, or None.

        Return static instruction text, a dynamic instruction callable, or a sequence
        containing either. For dynamic per-request behavior, return a callable that receives
        [`RunContext`][pydantic_ai.tools.RunContext] or a
        `TemplateStr` — not a dynamic string.

        When [`defer_loading`][pydantic_ai.capabilities.AbstractCapability.defer_loading] is
        True, these instructions are resolved only after the model calls the
        `load_capability` tool for this capability.
        """
        return None

    def _collect_instructions(self) -> list[SourcedInstruction[AgentDepsT]]:
        """Return this capability's instructions, each paired with the id to address it by.

        The agent uses this instead of [`get_instructions`][pydantic_ai.capabilities.AbstractCapability.get_instructions]
        so a capability with an [`id`][pydantic_ai.capabilities.AbstractCapability.id] gets its own
        [`InstructionPart`][pydantic_ai.messages.InstructionPart]s rather than being folded into the
        agent's — computed contributions included, since addressing the capability means addressing
        everything it tells the model. Container capabilities override this to keep each leaf's
        contribution attributed; every other capability inherits this default, so overriding
        `get_instructions` is enough.
        """
        return self._collect_own_instructions()

    def _collect_own_instructions(self) -> list[SourcedInstruction[AgentDepsT]]:
        """Collect this capability's public contribution without container recursion."""
        return [
            self._attribute_instruction(instruction) for instruction in normalize_instructions(self.get_instructions())
        ]

    def _attribute_instruction(self, instruction: AgentInstruction[AgentDepsT]) -> SourcedInstruction[AgentDepsT]:
        """Attribute one instruction recipe to this capability."""
        return sourced_instruction(instruction, CapabilityInstructionSource(self.id) if self.id is not None else None)

    def _attribute_container_instructions(
        self,
        authored: Sequence[AgentInstruction[AgentDepsT]],
        relayed: Sequence[SourcedInstruction[AgentDepsT]],
    ) -> list[SourcedInstruction[AgentDepsT]]:
        """Attribute what an overriding container returned, keeping what it merely passed along.

        Public container overrides return bare recipes, so identity is the only information that
        connects a relayed recipe to the child that authored it. An object appearing under more
        than one child is deliberately not connected: equal interned strings can be the same
        object, and leaving their keys unidentified is safer than assigning either child at random.
        """
        occurrences = Counter(id(sourced.instruction) for sourced in relayed)
        relayed_by_identity = {
            id(sourced.instruction): sourced for sourced in relayed if occurrences[id(sourced.instruction)] == 1
        }
        # `relayed` keeps every recipe alive for the whole call, so these identities stay meaningful.
        return [
            self._attribute_instruction(instruction)
            if (sourced := relayed_by_identity.get(id(instruction))) is None
            else sourced
            for instruction in authored
        ]

    def get_description(self) -> CapabilityDescription[AgentDepsT] | None:
        """Return a human-readable description of this capability, or None.

        Surfaced to the model in the catalog shown with the `load_capability` tool when
        [`defer_loading`][pydantic_ai.capabilities.AbstractCapability.defer_loading] is True.

        Return a static description string or a callable that receives
        [`RunContext`][pydantic_ai.tools.RunContext] (or no arguments) when the deferred
        capability catalog is rendered. Default: return the static `description` field.
        """
        return self.description

    def get_model_settings(self) -> AgentModelSettings[AgentDepsT] | None:
        """Return model settings to merge into the agent's defaults, or None.

        Return a static `ModelSettings` dict when the settings don't change between
        requests. Return a callable that receives [`RunContext`][pydantic_ai.tools.RunContext]
        when settings need to vary per step (e.g. based on `ctx.run_step` or `ctx.deps`).

        When the callable is invoked, `ctx.model_settings` contains the merged
        result of all layers resolved before this capability (model defaults and
        agent-level settings). The returned dict is merged on top of that.

        When [`defer_loading`][pydantic_ai.capabilities.AbstractCapability.defer_loading] is
        True, these settings are registered up front but merge as an empty dict until the
        model calls the `load_capability` tool for this capability.
        """
        return None

    def get_model(self) -> AgentModel[AgentDepsT] | None:
        """Return a static model, a per-step model selector, or `None` to make no selection.

        A selector receives
        [`ModelSelectionContext`][pydantic_ai.models.ModelSelectionContext] and may be
        synchronous or asynchronous. Static selections are resolved once per run; selectors
        are evaluated before each new logical model request step. When several capabilities
        contribute a model, the last non-`None` selection wins. This differs from
        [`resolve_model_id()`][pydantic_ai.capabilities.AbstractCapability.resolve_model_id],
        where the first resolver to return a model wins.

        See [Selecting the model](../capabilities/custom.md#selecting-the-model) for precedence,
        bootstrap, and deferred-capability semantics.
        """
        return None

    @property
    def has_resolve_model_id(self) -> bool:
        """Whether this capability or a wrapped capability overrides `resolve_model_id`."""
        return type(self).resolve_model_id is not AbstractCapability.resolve_model_id

    async def resolve_model_id(
        self,
        ctx: ModelResolutionContext[AgentDepsT],
        *,
        model_id: KnownModelName | str,
    ) -> Model | None:
        """Resolve a model ID, or return `None` to defer.

        Capabilities are tried in user-supplied order. When every capability returns `None`, the ID
        is passed to [`infer_model`][pydantic_ai.models.infer_model]. The context provides
        the agent and actual run dependencies, so resolution can configure tenant-specific
        providers or look up models in a registry.
        """
        return None

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Return a toolset to register with the agent, or None."""
        return None

    def get_native_tools(self) -> Sequence[AgentNativeTool[AgentDepsT]]:
        """Return native tools to register with the agent."""
        return []

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Wrap the agent's assembled toolset, or return None to leave it unchanged.

        Called per-run with the combined non-output toolset (after the
        [`prepare_tools`][pydantic_ai.capabilities.AbstractCapability.prepare_tools] hook
        has already wrapped it). Output tools are added separately and are not included.

        Unlike value-contribution methods such as
        [`get_instructions`][pydantic_ai.capabilities.AbstractCapability.get_instructions],
        this receives the already assembled toolset and is called each run (after
        [`for_run`][pydantic_ai.capabilities.AbstractCapability.for_run]).
        When multiple capabilities provide wrappers, they follow middleware semantics:
        the first capability in the list wraps outermost (matching `wrap_*` hooks).

        Use this to apply cross-cutting toolset wrappers like
        [`PreparedToolset`][pydantic_ai.toolsets.PreparedToolset],
        [`FilteredToolset`][pydantic_ai.toolsets.FilteredToolset],
        or custom [`WrapperToolset`][pydantic_ai.toolsets.WrapperToolset] subclasses.
        """
        return None

    # --- Tool preparation hooks ---

    async def prepare_tools(
        self,
        ctx: RunContext[AgentDepsT],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Filter or modify function tool definitions for this step.

        Receives **function** tools only. For [output tools][pydantic_ai.output.ToolOutput],
        override
        [`prepare_output_tools`][pydantic_ai.capabilities.AbstractCapability.prepare_output_tools]
        — it runs separately, with `ctx.retry`/`ctx.max_retries` reflecting the **output**
        retry budget instead of the function-tool budget.

        Return a filtered or modified list. The result flows into both the model's request
        parameters and `ToolManager.tools`, so filtering also blocks tool execution.

        On a deferred capability this runs only once the capability is loaded, and then receives
        every function tool, as an always-on capability does. There is nothing to govern
        before that: an unloaded capability's tools are neither advertised to the model nor
        callable, so no filtering here could change what the model can reach.
        """
        return tool_defs

    async def prepare_output_tools(
        self,
        ctx: RunContext[AgentDepsT],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Filter or modify output tool definitions for this step.

        Receives only [output tools][pydantic_ai.output.ToolOutput]. `ctx.retry` and
        `ctx.max_retries` reflect the **output** retry budget (agent-level
        `max_output_retries`), matching the output hook lifecycle.

        Return a filtered or modified list. The result flows into both the model's request
        parameters and `ToolManager.tools`, so filtering also blocks tool execution.
        """
        return tool_defs

    # --- Run lifecycle hooks ---

    async def before_run(
        self,
        ctx: RunContext[AgentDepsT],
    ) -> None:
        """Called before the agent run starts. Observe-only; use `wrap_run` for modification.

        A realtime session is a run. ContextVars set here are ambient in its instruction
        resolution, pump and tool tasks, and the caller's `async with` block.
        """

    async def after_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        """Called after the agent run produces a result. Can modify the result.

        Not called when the run ends without a result (e.g. a cancellation that nothing
        recovered from). It IS called when a result was produced while a cancellation was
        pending or absorbed upstream — but before the backstop's cancellation re-check, so the
        cancellation still propagates after this hook returns and the run still ends cancelled.
        Put cancellation-safe cleanup in [`wrap_run`][pydantic_ai.capabilities.AbstractCapability.wrap_run]
        (a `try`/`finally` around `handler()`), which does observe the `CancelledError`.

        For a realtime session, the result is produced when the session closes; a transformed result
        becomes `session.result` before the caller leaves the `async with` boundary.
        """
        return result

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        """Wraps the entire agent run. `handler()` executes the run.

        If `handler()` raises and this method catches the exception and
        returns a result instead, the error is suppressed and the recovery
        result is used.

        If this method does not call `handler()` (short-circuit), the run
        is skipped and the returned result is used directly.

        Note: if the caller cancels the run (e.g. by breaking out of an
        `iter()` loop), this method receives an `asyncio.CancelledError`.
        Implementations that hold resources should handle cleanup accordingly. Cancellation is
        terminal: the hook may observe it and clean up, but cannot recover the run to success.

        A realtime session is a run: `handler()` resolves when the session closes. ContextVars set
        before calling it are ambient in instruction resolution, pumps, tool tasks, and the caller's
        block. Downward ContextVar propagation is one-way; keep bidirectional per-run state on the
        `for_run` copy's instance attributes. Suppression and result transformation apply at the
        session's `async with` boundary, after the caller may have observed events in real time.
        """
        return await handler()

    async def on_run_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        error: BaseException,
    ) -> AgentRunResult[Any]:
        """Called when the agent run fails with an exception.

        This is the error counterpart to
        [`after_run`][pydantic_ai.capabilities.AbstractCapability.after_run]:
        while `after_run` is called on success, `on_run_error` is called on
        failure (after [`wrap_run`][pydantic_ai.capabilities.AbstractCapability.wrap_run]
        has had its chance to recover).

        **Raise** the original `error` (or a different exception) to propagate it.
        **Return** an [`AgentRunResult`][pydantic_ai.run.AgentRunResult] to suppress
        the error and recover the run.

        Cancellation is terminal: the hook may observe it and clean up, but cannot recover the
        run to success.

        Not called for `GeneratorExit` or `KeyboardInterrupt`.

        For a realtime session, returning a recovery result sets `session.result` and suppresses the
        error at the caller's `async with` boundary, after events may already have been observed.
        """
        raise error

    # --- Node run lifecycle hooks ---

    async def before_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
    ) -> AgentNode[AgentDepsT]:
        """Called before each graph node executes. Can observe or replace the node."""
        return node

    async def after_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        result: NodeResult[AgentDepsT],
    ) -> NodeResult[AgentDepsT]:
        """Called after each graph node succeeds. Can modify the result (next node or `End`).

        Not called for a node interrupted by cancellation — including a cancellation the node
        itself absorbed and completed through, which the framework re-asserts at the node
        boundary: cancellation skips downstream hooks. Put cancellation-safe cleanup in
        [`wrap_node_run`][pydantic_ai.capabilities.AbstractCapability.wrap_node_run]
        (a `try`/`finally` around `handler()`), which does observe the `CancelledError`.
        (A hook that catches the `CancelledError` *and* calls `Task.uncancel()` takes over the
        cancellation bookkeeping for that boundary, so this hook does fire for that node —
        the run itself still ends cancelled at the next boundary.)
        """
        return result

    async def wrap_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        handler: WrapNodeRunHandler[AgentDepsT],
    ) -> NodeResult[AgentDepsT]:
        """Wraps execution of each agent graph node (run step).

        Called for every node in the agent graph (`UserPromptNode`,
        `ModelRequestNode`, `CallToolsNode`).  `handler(node)` executes
        the node and returns the next node (or `End`).

        Override to inspect or modify nodes before execution, inspect or modify
        the returned next node, call `handler` multiple times (retry), or
        return a different node to redirect graph progression.

        Note: this hook fires however the run is driven -- [`agent.run()`][pydantic_ai.agent.AbstractAgent.run],
        [`agent.run_stream()`][pydantic_ai.agent.AbstractAgent.run_stream], an
        [`agent.iter()`][pydantic_ai.agent.Agent.iter] run advanced with
        [`agent_run.next()`][pydantic_ai.run.AgentRun.next], and a bare `async for node in agent_run:`
        loop, which advances through `next()` too. The one exception is the final
        [`ModelRequestNode`][pydantic_ai.agent.ModelRequestNode] under `run_stream()`, which hands back
        the result mid-stream and so only fires `before_node_run`.

        When using `agent.run()` with `event_stream_handler`, the handler wraps both
        streaming and graph advancement (i.e. the model call happens inside the wrapper).
        When using `agent.run_stream()`, the handler wraps only graph advancement — streaming
        happens before the wrapper because `run_stream()` must yield the stream to the caller
        while the stream context is still open, which cannot happen from inside a callback.

        A cancelled run delivers `asyncio.CancelledError` through `handler()`. Cancellation is
        terminal: the hook may observe it and clean up, but cannot recover the run to success —
        even a returned `End` result is discarded once a cancellation is pending.
        """
        return await handler(node)

    async def on_node_run_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        error: Exception,
    ) -> NodeResult[AgentDepsT]:
        """Called when a graph node fails with an exception.

        This is the error counterpart to
        [`after_node_run`][pydantic_ai.capabilities.AbstractCapability.after_node_run].

        **Raise** the original `error` (or a different exception) to propagate it.
        **Return** a next node or `End` to recover and continue the graph.

        Useful for recovering from
        [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior]
        by redirecting to a different node (e.g. retry with different model settings).
        """
        raise error

    # --- Event stream hook ---

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        """Wrap a run or realtime session's consumer-facing event stream.

        For classic runs, the wrapper is applied where each node's stream is produced, so it fires
        however the run is driven — including under [`agent.iter()`][pydantic_ai.agent.Agent.iter]
        and when the caller streams a node itself with `node.stream()`. For realtime sessions, it
        wraps the `async for event in session` view. A wrapper must yield events appropriate for the
        stream it wraps.

        Transformations affect only what the stream consumer sees. They never change realtime
        session history, tool execution, or the classic run's accumulated response and output.

        Note: when this method is overridden (or [`Hooks.on.event`][pydantic_ai.capabilities.hooks.Hooks.on]
        / [`Hooks.on.run_event_stream`][pydantic_ai.capabilities.hooks.Hooks.on] are registered),
        `agent.run()` and [`AgentRun.next()`][pydantic_ai.run.AgentRun.next] automatically enable
        streaming mode so this hook fires even without an explicit `event_stream_handler`.
        """
        try:
            async for event in stream:
                yield event
        finally:
            await _utils.aclose_if_supported(stream)

    # --- Model request lifecycle hooks ---

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Called before each model request. Can modify messages, settings, and parameters.

        [`model_request_parameters.instruction_parts`][pydantic_ai.models.ModelRequestParameters.instruction_parts]
        is the source of truth for the instructions: rewriting them here changes what the model
        receives, and the request recorded in message history is re-rendered from them afterwards.
        Assigning to a [`ModelRequest.instructions`][pydantic_ai.messages.ModelRequest] in
        `request_context.messages` is not propagated the other way, so it does not reach the model.
        """
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Called after each model response. Can modify the response before further processing.

        Raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to reject the response and
        ask the model to try again. The original response is still appended to message history
        so the model can see what it said. Retries count against the output side of the agent's retry budget.
        """
        return response

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Wraps the model request. handler() calls the model.

        Raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to skip `on_model_request_error`
        and directly retry the model request with a retry prompt. If the handler was called,
        the model response is preserved in history for context (same as `after_model_request`).
        """
        return await handler(request_context)

    async def on_model_request_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        error: Exception,
    ) -> ModelResponse:
        """Called when a model request fails with an exception.

        This is the error counterpart to
        [`after_model_request`][pydantic_ai.capabilities.AbstractCapability.after_model_request].

        **Raise** the original `error` (or a different exception) to propagate it.
        **Return** a [`ModelResponse`][pydantic_ai.messages.ModelResponse] to suppress
        the error and use the response as if the model call succeeded.
        **Raise** [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to retry the model request
        with a retry prompt instead of recovering or propagating.

        Not called for [`SkipModelRequest`][pydantic_ai.exceptions.SkipModelRequest]
        or [`ModelRetry`][pydantic_ai.exceptions.ModelRetry].
        """
        raise error

    # --- Tool validate lifecycle hooks ---

    async def before_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: RawToolArgs,
    ) -> RawToolArgs:
        """Modify raw args before validation.

        Raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to skip validation and
        ask the model to redo the tool call.

        A tool call can only be deferred once its arguments have been validated, so raising
        [`CallDeferred`][pydantic_ai.exceptions.CallDeferred] or
        [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] here is a `UserError`. Defer
        from [`after_tool_validate`][pydantic_ai.capabilities.AbstractCapability.after_tool_validate],
        a tool's `args_validator`, or
        [`before_tool_execute`][pydantic_ai.capabilities.AbstractCapability.before_tool_execute].
        """
        return args

    async def after_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
    ) -> ValidatedToolArgs:
        """Modify validated args. Called only on successful validation.

        Raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to reject the validated args
        and ask the model to redo the tool call.

        The arguments are valid by this point, so raising
        [`CallDeferred`][pydantic_ai.exceptions.CallDeferred] or
        [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] here defers the call — the tool
        isn't executed, and the deferral joins the run's
        [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] with the validated arguments.

        This hook also runs when the tool's `args_validator` (or `wrap_tool_validate`) already
        deferred the call, so it stays a reliable gate on validated arguments: rejecting here wins
        over that deferral, deferring here replaces it, and the args returned here are the ones the
        deferred call carries.
        """
        return args

    async def wrap_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: RawToolArgs,
        handler: WrapToolValidateHandler,
    ) -> ValidatedToolArgs:
        """Wraps tool argument validation. handler() runs the validation.

        Deferring with [`CallDeferred`][pydantic_ai.exceptions.CallDeferred] or
        [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] is allowed *after* `handler()`
        has returned, when the arguments are known to be valid; raising one before that is a
        `UserError`.
        """
        return await handler(args)

    async def on_tool_validate_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: RawToolArgs,
        error: ValidationError | ModelRetry,
    ) -> ValidatedToolArgs:
        """Called when tool argument validation fails.

        This is the error counterpart to
        [`after_tool_validate`][pydantic_ai.capabilities.AbstractCapability.after_tool_validate].
        Fires for `ValidationError` (schema mismatch) and
        [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] (custom validator rejection).

        **Raise** the original `error` (or a different exception) to propagate it.
        **Return** validated args to suppress the error and continue as if validation passed.

        Not called for [`SkipToolValidation`][pydantic_ai.exceptions.SkipToolValidation], or when a
        tool's `args_validator` raises [`CallDeferred`][pydantic_ai.exceptions.CallDeferred] or
        [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] — those are control flow, not
        errors, and the call is deferred instead of executed.

        Raising a deferral *from this hook* is a `UserError`: it only runs because validation failed,
        so there are no valid arguments to show whoever would resolve the deferral.
        """
        raise error

    # --- Tool execute lifecycle hooks ---

    async def before_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
    ) -> ValidatedToolArgs:
        """Modify validated args before execution.

        Raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to skip execution and
        ask the model to redo the tool call.

        This is the hook to defer from: raising
        [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] or
        [`CallDeferred`][pydantic_ai.exceptions.CallDeferred] here defers the call *before* the tool
        function runs, so nothing happens until it's resolved.
        """
        return args

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: Any,
    ) -> Any:
        """Modify result after execution.

        Raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to reject the tool result
        and ask the model to redo the tool call.

        Deferring from here is accepted but rarely what you want: the tool function has already run,
        so its side effects happened and `result` is discarded. Defer from
        [`before_tool_execute`][pydantic_ai.capabilities.AbstractCapability.before_tool_execute]
        instead.
        """
        return result

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        """Wraps tool execution. handler() runs the tool.

        Defer before calling `handler()`: a deferral raised after it has returned is accepted, but
        the tool function already ran and its result is discarded. Defer from
        [`before_tool_execute`][pydantic_ai.capabilities.AbstractCapability.before_tool_execute]
        instead.
        """
        return await handler(args)

    async def on_tool_execute_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        error: Exception,
    ) -> Any:
        """Called when tool execution fails with an exception.

        This is the error counterpart to
        [`after_tool_execute`][pydantic_ai.capabilities.AbstractCapability.after_tool_execute].

        **Raise** the original `error` (or a different exception) to propagate it.
        **Return** any value to suppress the error and use it as the tool result.
        **Raise** [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to ask the model to
        redo the tool call instead of recovering or propagating.

        Not called for control flow exceptions
        ([`SkipToolExecution`][pydantic_ai.exceptions.SkipToolExecution],
        [`CallDeferred`][pydantic_ai.exceptions.CallDeferred],
        [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired]),
        retry signals ([`ToolRetryError`][pydantic_ai.exceptions.ToolRetryError]
        from [`ModelRetry`][pydantic_ai.exceptions.ModelRetry]), or failure signals
        ([`ToolFailedError`][pydantic_ai.exceptions.ToolFailedError]
        from [`ToolFailed`][pydantic_ai.exceptions.ToolFailed]).
        Use [`wrap_tool_execute`][pydantic_ai.capabilities.AbstractCapability.wrap_tool_execute]
        to intercept retries or failures.
        """
        raise error

    # --- Output validate lifecycle hooks ---

    async def before_output_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: RawOutput,
    ) -> RawOutput:
        """Modify raw model output before validation/parsing.

        The primary hook for pre-parse repair and normalization of model output.
        Fires only for structured output that requires parsing: prompted, native,
        tool, and union output. Does **not** fire for plain text or image output.

        For structured text output, `output` is the raw text string from the model.
        For tool output, `output` is the raw tool arguments (string or dict).

        Raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to skip validation and
        ask the model to try again with a custom message.

        During streaming, this hook fires on every partial validation attempt as well as
        the final result. Check `ctx.partial_output` to distinguish and avoid expensive
        work on partial results.
        """
        return output

    async def after_output_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
    ) -> Any:
        """Modify validated output after successful parsing. Called only on success.

        `output` is the **semantic value** the model was asked to produce — e.g., a
        `MyModel` instance for `output_type=MyModel`, or `42` for `output_type=int`, or
        the input to a single-arg output function. For multi-arg output functions, this
        is the `dict` of arguments (the genuine multi-value input).

        Note: this differs from *tool* hooks (`after_tool_validate`), which always see
        `dict[str, Any]` — tool args follow the schema contract. Output hooks see the
        semantic output value, regardless of how it's internally represented during
        validation.

        Raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to reject the validated
        output and ask the model to try again.
        """
        return output

    async def wrap_output_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: RawOutput,
        handler: WrapOutputValidateHandler,
    ) -> Any:
        """Wraps output validation. handler(output) performs the validation.

        [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] from within the handler goes to
        [`on_output_validate_error`][pydantic_ai.capabilities.AbstractCapability.on_output_validate_error].
        `ModelRetry` raised directly (not from the handler) bypasses the error hook.
        """
        return await handler(output)

    async def on_output_validate_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: RawOutput,
        error: ValidationError | ModelRetry,
    ) -> Any:
        """Called when output validation fails.

        This is the error counterpart to
        [`after_output_validate`][pydantic_ai.capabilities.AbstractCapability.after_output_validate].

        **Raise** the original `error` (or a different exception) to propagate it.
        **Return** validated output to suppress the error and continue.
        """
        raise error

    # --- Output process lifecycle hooks ---

    async def before_output_process(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
    ) -> Any:
        """Modify validated output before processing (extraction, output function call).

        `output` is the **semantic value** — e.g., a `MyModel` instance or `42`, matching
        `after_output_validate`. For multi-arg output functions, it's the `dict` of args.
        See [`after_output_validate`][pydantic_ai.capabilities.AbstractCapability.after_output_validate]
        for a full explanation of the semantic-value contract.

        Raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to skip processing and
        ask the model to try again.
        """
        return output

    async def after_output_process(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
    ) -> Any:
        """Modify result after output processing.

        Raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to reject the result
        and ask the model to try again.
        """
        return output

    async def wrap_output_process(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
        handler: WrapOutputProcessHandler,
    ) -> Any:
        """Wraps output processing. handler(output) runs extraction + output function call.

        [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] bypasses
        [`on_output_process_error`][pydantic_ai.capabilities.AbstractCapability.on_output_process_error]
        (treated as control flow, not an error).

        During streaming, this fires only when partial validation succeeds, and on the
        final result. Check `ctx.partial_output` to skip expensive work on partial results.
        """
        return await handler(output)

    async def on_output_process_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
        error: Exception,
    ) -> Any:
        """Called when output processing fails with an exception.

        This is the error counterpart to
        [`after_output_process`][pydantic_ai.capabilities.AbstractCapability.after_output_process].

        **Raise** the original `error` (or a different exception) to propagate it.
        **Return** any value to suppress the error and use it as the output.

        Not called for retry signals ([`ToolRetryError`][pydantic_ai.exceptions.ToolRetryError]
        from [`ModelRetry`][pydantic_ai.exceptions.ModelRetry]).
        """
        raise error

    # --- Deferred tool call hooks ---

    async def handle_deferred_tool_calls(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        requests: DeferredToolRequests,
    ) -> DeferredToolResults | None:
        """Handle deferred tool calls (approval-required or externally-executed) inline during an agent run.

        Called by `ToolManager` when:

        - a tool raises [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] or
          [`CallDeferred`][pydantic_ai.exceptions.CallDeferred] during execution, or
        - the model calls a tool registered with `requires_approval=True` (see
          [Human-in-the-Loop Tool Approval](../deferred-tools.md#human-in-the-loop-tool-approval))
          or a tool backed by [external execution](../deferred-tools.md#external-tool-execution).

        Uses accumulation dispatch: each capability in the chain receives remaining
        unresolved requests and can resolve some or all of them. Results are merged
        and unresolved calls are passed to the next capability.

        **Return** a [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults] to resolve
        some or all calls.
        **Return** `None` to leave all calls unresolved.
        """
        return None

    # --- Convenience methods ---

    def prefix_tools(self, prefix: str) -> PrefixTools[AgentDepsT]:
        """Returns a new capability that wraps this one and prefixes its tool names.

        Only this capability's tools are prefixed; other agent tools are unaffected.
        """
        from .prefix_tools import PrefixTools

        return PrefixTools(wrapped=self, prefix=prefix)


def leaf_capabilities(capability: AbstractCapability[AgentDepsT]) -> list[AbstractCapability[AgentDepsT]]:
    """Collect the leaf capabilities in a capability tree, in application order."""
    leaves: list[AbstractCapability[AgentDepsT]] = []
    capability.apply(leaves.append)
    return leaves
