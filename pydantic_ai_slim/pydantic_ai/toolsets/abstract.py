from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol

from pydantic_core import SchemaValidator
from typing_extensions import Self

from .._instructions import normalize_toolset_instruction_parts
from .._run_context import AgentDepsT, RunContext
from .._utils import gather
from ..messages import InstructionPart, ToolsetInstructionSource
from ..tools import ToolDefinition, ToolsPrepareFunc
from ._instruction_collection import InstructionContribution, make_contribution

if TYPE_CHECKING:
    from .approval_required import ApprovalRequiredToolset
    from .deferred_loading import DeferredLoadingToolset
    from .filtered import FilteredToolset
    from .include_return_schemas import IncludeReturnSchemasToolset
    from .prefixed import PrefixedToolset
    from .prepared import PreparedToolset
    from .renamed import RenamedToolset
    from .set_metadata import SetMetadataToolset


AGENT_TOOLSET_ID = '<agent>'
"""The [`id`][pydantic_ai.toolsets.AbstractToolset.id] of the function toolset an agent builds for its own tools."""

OUTPUT_TOOLSET_ID = '<output>'
"""The [`id`][pydantic_ai.toolsets.AbstractToolset.id] of the toolset an agent builds for its output tools."""


class SchemaValidatorProt(Protocol):
    """Protocol for a Pydantic Core `SchemaValidator` or `PluggableSchemaValidator` (which is private but API-compatible)."""

    def validate_json(
        self,
        input: str | bytes | bytearray,
        *,
        allow_partial: bool | Literal['off', 'on', 'trailing-strings'] = False,
        **kwargs: Any,
    ) -> Any: ...

    def validate_python(
        self, input: Any, *, allow_partial: bool | Literal['off', 'on', 'trailing-strings'] = False, **kwargs: Any
    ) -> Any: ...


@dataclass(kw_only=True)
class ToolsetTool(Generic[AgentDepsT]):
    """Definition of a tool available on a toolset.

    This is a wrapper around a plain tool definition that includes information about:

    - the toolset that provided it, for use in error messages
    - the maximum number of retries to attempt if the tool call fails
    - the validator for the tool's arguments
    """

    toolset: AbstractToolset[AgentDepsT]
    """The toolset that provided this tool, for use in error messages."""
    tool_def: ToolDefinition
    """The tool definition for this tool, including the name, description, and parameters."""
    max_retries: int
    """The maximum number of retries to attempt if the tool call fails."""
    args_validator: SchemaValidator | SchemaValidatorProt
    """The Pydantic Core validator for the tool's arguments.

    For example, a [`pydantic.TypeAdapter(...).validator`](https://docs.pydantic.dev/latest/concepts/type_adapter/) or [`pydantic_core.SchemaValidator`](https://docs.pydantic.dev/latest/api/pydantic_core/#pydantic_core.SchemaValidator).
    """
    args_validator_func: Callable[..., Any] | None = None
    """Custom args validator function that runs after schema validation but before tool execution.

    Called on every tool call, receiving the schema-validated arguments as keyword args.
    The function should have the same typed parameters as the tool function,
    with `RunContext` as the first argument.
    Raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to ask the model to correct the arguments and
    try again, or [`ToolFailed`][pydantic_ai.exceptions.ToolFailed] to report a terminal failure the model
    should adapt to instead of retrying. Return `None` on success.
    """


class AbstractToolset(ABC, Generic[AgentDepsT]):
    """A toolset is a collection of tools that can be used by an agent.

    It is responsible for:

    - Listing the tools it contains
    - Validating the arguments of the tools
    - Calling the tools

    See [toolset docs](../toolsets.md) for more information.
    """

    @property
    @abstractmethod
    def id(self) -> str | None:
        """An ID for the toolset that is unique among all toolsets registered with the same agent.

        If you're implementing a concrete implementation that users can instantiate more than once, you should let them optionally pass a custom ID to the constructor and return that here.

        A toolset needs to have an ID in order to be used in a durable execution environment like Temporal, in which case the ID will be used to identify the toolset's activities within the workflow.

        IDs wrapped in angle brackets (`'<agent>'` for an agent's own function toolset, `'<output>'` for
        its output tools) name a role the framework fills on the user's behalf rather than a registered
        toolset. Don't return one from your own toolset.
        """
        raise NotImplementedError()

    @property
    def label(self) -> str:
        """The name of the toolset for use in error messages."""
        label = self.__class__.__name__
        if self.id:  # pragma: no branch
            label += f' {self.id!r}'
        return label

    @property
    def tool_name_conflict_hint(self) -> str:
        """A hint for how to avoid name conflicts with other toolsets for use in error messages."""
        return 'Rename the tool or wrap the toolset in a `PrefixedToolset` to avoid name conflicts.'

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Return the toolset to use for this agent run.

        Called once per run, before `__aenter__`. Override this to return a fresh instance
        for per-run state isolation. Default: return `self` (shared across runs).
        """
        return self

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Return the toolset to use for this run step.

        Called at the start of each run step. Override this to return a modified
        instance for per-step state transitions. If returning a new instance,
        you are responsible for managing any lifecycle transitions (exiting old
        inner toolsets, entering new ones). Default: return `self` (no per-step changes).
        """
        return self

    async def __aenter__(self) -> Self:
        """Enter the toolset context.

        This is where you can set up network connections in a concrete implementation.
        """
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        """Exit the toolset context.

        This is where you can tear down network connections in a concrete implementation.
        """
        return None

    async def get_instructions(
        self, ctx: RunContext[AgentDepsT]
    ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
        r"""Return instructions for how to use this toolset's tools.

        Override this method to provide instructions that help the agent understand
        how to use the tools in this toolset effectively.

        Simple implementations can return a plain `str`; advanced implementations can return
        [`InstructionPart`][pydantic_ai.messages.InstructionPart] objects to indicate whether
        each instruction part is static or dynamic for caching purposes.

        Args:
            ctx: The run context for this agent run.

        Returns:
            Instruction string, `InstructionPart`, list of either, or `None`.
            Plain `str` values are treated as dynamic instructions by default.
        """
        return None

    async def _collect_instruction_contributions(
        self, ctx: RunContext[AgentDepsT]
    ) -> list[InstructionContribution[AgentDepsT]]:
        """Collect contributions once, preserving the toolset that authored every relayed part.

        A toolset that only passes its children along is walked; anything that speaks for itself is
        asked once. That covers a leaf and a container whose subclass took `get_instructions` over
        with the same path, because the difference between them is only what they own: a returned
        key owned below is relayed unchanged and stays attributed to its owner, and everything else
        is the caller's own text, resolved against the caller's own key.
        """
        if not self._authors_own_instructions():
            return await self._collect_child_instruction_contributions(ctx)

        result = await self.get_instructions(ctx)
        sources_by_key = self._instruction_sources_by_key()
        contributions: list[InstructionContribution[AgentDepsT]] = []
        for part in normalize_toolset_instruction_parts(result):
            # A key names the toolset that owns it, so a part arriving under one below this
            # container is being relayed and stays attributed there. Everything else this container
            # wrote itself, and is resolved against its own key like any other author's.
            owner = (
                sources_by_key.get(part.id.source)
                if part.id is not None and isinstance(part.id.source, ToolsetInstructionSource)
                else None
            )
            contributions.append(make_contribution(owner if owner is not None else self, part))
        return contributions

    async def _collect_child_instruction_contributions(
        self, ctx: RunContext[AgentDepsT]
    ) -> list[InstructionContribution[AgentDepsT]]:
        """Gather child contributions without re-entering a container's public override check."""
        child_contributions = await gather(
            *(child._collect_instruction_contributions(ctx) for child in self._instruction_children())
        )
        return [contribution for contributions in child_contributions for contribution in contributions]

    def _instruction_children(self) -> Sequence[AbstractToolset[AgentDepsT]]:
        """The toolsets whose instruction contributions this one passes along."""
        return ()

    def _authors_own_instructions(self) -> bool:
        """Whether `get_instructions` speaks for this toolset rather than aggregating its children.

        True here because a toolset with nothing below it can only be speaking for itself. A
        container overrides this to answer for the case that actually varies: whether a subclass has
        taken the method over, or it is still the inherited implementation that just relays.
        """
        return True

    def _instruction_source(self) -> ToolsetInstructionSource | None:
        """Read this toolset's source without validating an id that contributes no instructions."""
        if self.id is None or ':' in self.id:
            return None
        return ToolsetInstructionSource(self.id)

    def _instruction_sources_by_key(self) -> dict[ToolsetInstructionSource, AbstractToolset[AgentDepsT]]:
        """Map every source key at or below this toolset to the toolset that owns it.

        Children are inserted first and the container is inserted last without overwriting them, so
        a child remains the owner when a malformed tree repeats its key at a container boundary.
        The duplicate contribution check reports the ambiguity if both sources actually contribute.
        """
        sources: dict[ToolsetInstructionSource, AbstractToolset[AgentDepsT]] = {}
        for child in self._instruction_children():
            for source_id, source in child._instruction_sources_by_key().items():
                sources.setdefault(source_id, source)
        if source := self._instruction_source():
            sources.setdefault(source, self)
        return sources

    @abstractmethod
    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """The tools that are available in this toolset."""
        raise NotImplementedError()

    @abstractmethod
    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        """Call a tool with the given arguments.

        Args:
            name: The name of the tool to call.
            tool_args: The arguments to pass to the tool.
            ctx: The run context.
            tool: The tool definition returned by [`get_tools`][pydantic_ai.toolsets.AbstractToolset.get_tools] that was called.
        """
        raise NotImplementedError()

    def apply(self, visitor: Callable[[AbstractToolset[AgentDepsT]], None]) -> None:
        """Run a visitor function on all "leaf" toolsets (i.e. those that implement their own tool listing and calling)."""
        visitor(self)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        """Run a visitor function on all "leaf" toolsets (i.e. those that implement their own tool listing and calling) and replace them in the hierarchy with the result of the function."""
        return visitor(self)

    def filtered(
        self, filter_func: Callable[[RunContext[AgentDepsT], ToolDefinition], bool | Awaitable[bool]]
    ) -> FilteredToolset[AgentDepsT]:
        """Returns a new toolset that filters this toolset's tools using a filter function that takes the agent context and the tool definition.

        See [toolset docs](../toolsets.md#filtering-tools) for more information.
        """
        from .filtered import FilteredToolset

        return FilteredToolset(self, filter_func)

    def prefixed(self, prefix: str) -> PrefixedToolset[AgentDepsT]:
        """Returns a new toolset that prefixes the names of this toolset's tools.

        See [toolset docs](../toolsets.md#prefixing-tool-names) for more information.
        """
        from .prefixed import PrefixedToolset

        return PrefixedToolset(self, prefix)

    def prepared(self, prepare_func: ToolsPrepareFunc[AgentDepsT]) -> PreparedToolset[AgentDepsT]:
        """Returns a new toolset that prepares this toolset's tools using a prepare function that takes the agent context and the original tool definitions.

        See [toolset docs](../toolsets.md#preparing-tool-definitions) for more information.
        """
        from .prepared import PreparedToolset

        return PreparedToolset(self, prepare_func)

    def renamed(self, name_map: dict[str, str]) -> RenamedToolset[AgentDepsT]:
        """Returns a new toolset that renames this toolset's tools using a dictionary mapping new names to original names.

        See [toolset docs](../toolsets.md#renaming-tools) for more information.
        """
        from .renamed import RenamedToolset

        return RenamedToolset(self, name_map)

    def approval_required(
        self,
        approval_required_func: Callable[[RunContext[AgentDepsT], ToolDefinition, dict[str, Any]], bool] = (
            lambda ctx, tool_def, tool_args: True
        ),
    ) -> ApprovalRequiredToolset[AgentDepsT]:
        """Returns a new toolset that requires (some) calls to tools it contains to be approved.

        See [toolset docs](../toolsets.md#requiring-tool-approval) for more information.
        """
        from .approval_required import ApprovalRequiredToolset

        return ApprovalRequiredToolset(self, approval_required_func)

    def defer_loading(self, tool_names: Sequence[str] | None = None) -> DeferredLoadingToolset[AgentDepsT]:
        """Returns a new toolset that marks tools for deferred loading, hiding them until revealed.

        Tool search, `load_capability` and another tool's `ToolReturn.tools` all reveal.

        See [toolset docs](../toolsets.md#deferred-loading) for more information.

        Args:
            tool_names: Optional sequence of tool names to mark for deferred loading.
                If `None`, all tools are marked for deferred loading.
        """
        from .deferred_loading import DeferredLoadingToolset

        return DeferredLoadingToolset(self, tool_names=frozenset(tool_names) if tool_names is not None else None)

    def include_return_schemas(self) -> IncludeReturnSchemasToolset[AgentDepsT]:
        """Returns a new toolset that sets `include_return_schema=True` on all tools.

        This causes the model to receive return type information for the tools
        in this toolset. For models that natively support return schemas (e.g.
        Google Gemini), the schema is passed as a structured field. For other
        models, it is injected into the tool description as JSON text.

        This is the toolset-level equivalent of the
        [`IncludeToolReturnSchemas`][pydantic_ai.capabilities.IncludeToolReturnSchemas]
        capability, which can be used to enable return schemas across all
        toolsets or a subset matched by a
        [`ToolSelector`][pydantic_ai.tools.ToolSelector].
        """
        from .include_return_schemas import IncludeReturnSchemasToolset

        return IncludeReturnSchemasToolset(self)

    def with_metadata(self, **metadata: Any) -> SetMetadataToolset[AgentDepsT]:
        """Returns a new toolset that merges the given metadata onto all tools."""
        from .set_metadata import SetMetadataToolset

        return SetMetadataToolset(self, metadata)
