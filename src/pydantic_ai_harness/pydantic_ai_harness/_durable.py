"""Durable-execution helpers: carry a tool's `ModelRetry` out of a durable operation as data, and detect a durable run.

A capability routes a tool's I/O through a `durable_operation` so that durable execution records
the result instead of repeating the I/O on recovery. What a durable operation records is what it
returns, and a `ModelRetry` is raised, so an operation that can ask the model to retry returns a
`RetryRequest` instead, and the tool raises it again with `raise_retry`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolReturn
from pydantic_ai.tools import RunContext

T = TypeVar('T')


@dataclass
class RetryRequest:
    """A `ModelRetry` raised inside a durable operation, returned as the operation's result."""

    message: str
    """The message to return to the model."""


ToolOperation = Callable[[str], Awaitable[ToolReturn[str] | RetryRequest]]
"""A durable operation that makes one tool's request for its one argument and renders the tool's result."""


async def retry_as_result(result: Awaitable[T]) -> T | RetryRequest:
    """Await `result`, returning a `ModelRetry` it raises as a `RetryRequest`."""
    try:
        return await result
    except ModelRetry as retry:
        return RetryRequest(retry.message)


def raise_retry(result: T | RetryRequest) -> T:
    """Return `result`, or raise the `ModelRetry` a `RetryRequest` stands for."""
    if isinstance(result, RetryRequest):
        raise ModelRetry(result.message)
    return result


@runtime_checkable
class _Durability(Protocol):
    """The part of the durable-execution capabilities' shared base this check needs."""

    in_durable_context: bool


# `_monty_exec.in_temporal_workflow` asks Temporal directly, because only Temporal replays
# `run_code`. This one covers every engine.
def in_durable_context(ctx: RunContext[Any]) -> bool:
    """Whether this run executes inside a durable workflow or flow, without importing the optional extras."""
    return any(
        any(base.__module__.startswith('pydantic_ai.durable_exec') for base in type(capability).__mro__)
        and isinstance(capability, _Durability)
        and capability.in_durable_context
        for capability in ctx.capabilities.values()
    )
