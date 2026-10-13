"""Carry a tool's `ModelRetry` out of a capability's durable operation as data.

A capability routes a tool's I/O through a `durable_operation` so that durable execution records
the result instead of repeating the I/O on recovery. What a durable operation records is what it
returns, and a `ModelRetry` is raised, so an operation that can ask the model to retry returns a
`RetryRequest` instead, and the tool raises it again with `raise_retry`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolReturn

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
