from __future__ import annotations as _annotations

from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar, Union

from typing_extensions import Concatenate, ParamSpec, TypeAlias

if TYPE_CHECKING:
    from .result import ResultData
else:
    ResultData = Any

__all__ = (
    'AgentDeps',
    'RunContext',
    'ResultValidatorFunc',
    'SystemPromptFunc',
    'ToolReturnValue',
    'ToolContextFunc',
    'ToolPlainFunc',
    'ToolParams',
    'JsonData',
)

AgentDeps = TypeVar('AgentDeps')
"""Type variable for agent dependencies."""


@dataclass
class RunContext(Generic[AgentDeps]):
    """Information about the current call."""

    deps: AgentDeps
    """Dependencies for the agent."""
    retry: int
    """Number of retries so far."""
    tool_name: str | None
    """Name of the tool being called."""


ToolParams = ParamSpec('ToolParams')
"""Retrieval function param spec."""

SystemPromptFunc = Union[
    Callable[[RunContext[AgentDeps]], str],
    Callable[[RunContext[AgentDeps]], Awaitable[str]],
    Callable[[], str],
    Callable[[], Awaitable[str]],
]
"""A function that may or maybe not take `RunContext` as an argument, and may or may not be async.

Usage `SystemPromptFunc[AgentDeps]`.
"""

ResultValidatorFunc = Union[
    Callable[[RunContext[AgentDeps], ResultData], ResultData],
    Callable[[RunContext[AgentDeps], ResultData], Awaitable[ResultData]],
    Callable[[ResultData], ResultData],
    Callable[[ResultData], Awaitable[ResultData]],
]
"""
A function that always takes `ResultData` and returns `ResultData`,
but may or maybe not take `CallInfo` as a first argument, and may or may not be async.

Usage `ResultValidator[AgentDeps, ResultData]`.
"""

JsonData: TypeAlias = 'None | str | int | float | Sequence[JsonData] | Mapping[str, JsonData]'
"""Type representing any JSON data."""

ToolReturnValue = Union[JsonData, Awaitable[JsonData]]
"""Return value of a tool function."""
ToolContextFunc = Callable[Concatenate[RunContext[AgentDeps], ToolParams], ToolReturnValue]
"""A tool function that takes `RunContext` as the first argument.

Usage `ToolContextFunc[AgentDeps, ToolParams]`.
"""
ToolPlainFunc = Callable[ToolParams, ToolReturnValue]
"""A tool function that does not take `RunContext` as the first argument.

Usage `ToolPlainFunc[ToolParams]`.
"""
