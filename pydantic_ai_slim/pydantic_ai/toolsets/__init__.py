from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, Literal

from pydantic_core import SchemaValidator
from typing_extensions import Self

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition

if TYPE_CHECKING:
    from ..models import Model
    from ._run import RunToolset


class AbstractToolset(ABC, Generic[AgentDepsT]):
    """A toolset is a collection of tools that can be used by an agent.

    It is responsible for:
    - Listing the tools it contains
    - Validating the arguments of the tools
    - Calling the tools
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace('Toolset', ' toolset')

    @property
    def tool_name_conflict_hint(self) -> str:
        return 'Consider renaming the tool or wrapping the toolset in a `PrefixedToolset` to avoid name conflicts.'

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> bool | None:
        return None

    @abstractmethod
    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def tool_defs(self) -> list[ToolDefinition]:
        raise NotImplementedError()

    @property
    def tool_names(self) -> list[str]:
        return [tool_def.name for tool_def in self.tool_defs]

    def get_tool_def(self, name: str) -> ToolDefinition | None:
        return next((tool_def for tool_def in self.tool_defs if tool_def.name == name), None)

    @abstractmethod
    def _get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        raise NotImplementedError()

    def validate_tool_args(
        self, ctx: RunContext[AgentDepsT], name: str, args: str | dict[str, Any] | None, allow_partial: bool = False
    ) -> dict[str, Any]:
        pyd_allow_partial: Literal['off', 'trailing-strings'] = 'trailing-strings' if allow_partial else 'off'
        validator = self._get_tool_args_validator(ctx, name)
        if isinstance(args, str):
            return validator.validate_json(args or '{}', allow_partial=pyd_allow_partial)
        else:
            return validator.validate_python(args or {}, allow_partial=pyd_allow_partial)

    @abstractmethod
    def _max_retries_for_tool(self, name: str) -> int:
        raise NotImplementedError()

    @abstractmethod
    async def call_tool(
        self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        raise NotImplementedError()

    @contextmanager
    def override_sampling_model(self, model: Model) -> Iterator[None]:
        yield
