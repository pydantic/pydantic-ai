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
    'CallContext',
    'ResultValidatorFunc',
    'SystemPromptFunc',
    'RetrieverReturnValue',
    'RetrieverContextFunc',
    'RetrieverPlainFunc',
    'RetrieverParams',
    'JsonData',
)

AgentDeps = TypeVar('AgentDeps')
"""Type variable for agent dependencies."""


@dataclass
class CallContext(Generic[AgentDeps]):
    """Information about the current call."""

    deps: AgentDeps
    """Dependencies for the agent."""
    retry: int
    """Number of retries so far."""
    tool_name: str | None
    """Name of the tool being called."""


RetrieverParams = ParamSpec('RetrieverParams')
"""Retrieval function param spec."""

SystemPromptFunc = Union[
    Callable[[CallContext[AgentDeps]], str],
    Callable[[CallContext[AgentDeps]], Awaitable[str]],
    Callable[[], str],
    Callable[[], Awaitable[str]],
]
"""A function that may or maybe not take `CallContext` as an argument, and may or may not be async.

Usage `SystemPromptFunc[AgentDeps]`.
"""

ResultValidatorFunc = Union[
    Callable[[CallContext[AgentDeps], ResultData], ResultData],
    Callable[[CallContext[AgentDeps], ResultData], Awaitable[ResultData]],
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

RetrieverReturnValue = Union[JsonData, Awaitable[JsonData]]
"""Return value of a retriever function."""
RetrieverContextFunc = Callable[Concatenate[CallContext[AgentDeps], RetrieverParams], RetrieverReturnValue]
"""A retriever function that takes `CallContext` as the first argument.

Usage `RetrieverContextFunc[AgentDeps, RetrieverParams]`.
"""
RetrieverPlainFunc = Callable[RetrieverParams, RetrieverReturnValue]
"""A retriever function that does not take `CallContext` as the first argument.

Usage `RetrieverPlainFunc[RetrieverParams]`.
"""
