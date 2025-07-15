from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic

from pydantic_core import SchemaValidator
from typing_extensions import Self

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition


class AbstractToolset(ABC, Generic[AgentDepsT]):
    """A toolset is a collection of tools that can be used by an agent.

    It is responsible for:

    - Listing the tools it contains
    - Validating the arguments of the tools
    - Calling the tools
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

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """TODO: Docstring."""
        return self

    @abstractmethod
    async def list_tool_defs(self, ctx: RunContext[AgentDepsT]) -> list[ToolDefinition]:
        """The tool definitions that are available in this toolset."""
        raise NotImplementedError()

    @abstractmethod
    def max_retries_for_tool(self, name: str) -> int:
        """The maximum number of retries for a given tool during an agent run."""
        raise NotImplementedError()

    @abstractmethod
    def get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        """Get the Pydantic Core schema validator for a given tool."""
        raise NotImplementedError()

    @abstractmethod
    async def call_tool(self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any]) -> Any:
        """Call a tool with the given arguments.

        Args:
            ctx: The run context.
            name: The name of the tool to call.
            tool_args: The arguments to pass to the tool.
        """
        raise NotImplementedError()

    def accept(self, visitor: Callable[[AbstractToolset[AgentDepsT]], Any]) -> Any:
        """Run a visitor function on all toolsets that implement tool listing and calling themselves instead of delegating to other toolsets."""
        return visitor(self)
