from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, Protocol

from pydantic_core import SchemaValidator
from typing_extensions import Self

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition, ToolsPrepareFunc

if TYPE_CHECKING:
    from .filtered import FilteredToolset
    from .prefixed import PrefixedToolset
    from .prepared import PreparedToolset
    from .renamed import RenamedToolset


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


@dataclass
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


class AbstractToolset(ABC, Generic[AgentDepsT]):
    """A toolset is a collection of tools that can be used by an agent.

    It is responsible for:

    - Listing the tools it contains
    - Validating the arguments of the tools
    - Calling the tools

    See [toolset docs](../toolsets.md) for more information.
    """

    @property
    def name(self) -> str:
        """The name of the toolset for use in error messages."""
        return self.__class__.__name__.replace('Toolset', ' toolset')

    @property
    def tool_name_conflict_hint(self) -> str:
        """A hint for how to avoid name conflicts with other toolsets for use in error messages."""
        return 'Rename the tool or wrap the toolset in a `PrefixedToolset` to avoid name conflicts.'

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
        """Run a visitor function on all concrete toolsets that are not wrappers (i.e. they implement their own tool listing and calling)."""
        visitor(self)

    def filtered(
        self, filter_func: Callable[[RunContext[AgentDepsT], ToolDefinition], bool]
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
