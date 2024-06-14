from dataclasses import dataclass
from typing import AsyncIterable, Generic, Iterable, TypeVar


@dataclass
class Cost:
    """Cost of a run."""

    total_cost: int


@dataclass
class History:
    """History of an entire conversation."""

    messages: list[str]

    def as_json(self) -> str:
        """Return the history as a JSON string."""
        return str(self.messages)


ResponseType = TypeVar('ResponseType')


@dataclass
class RunResult(Generic[ResponseType]):
    """Result of a run."""

    response: ResponseType
    history: History
    cost: Cost


@dataclass
class RunStreamResult(Generic[ResponseType]):
    """Result of streamed run."""

    history: History
    cost: Cost
    _streamed: str = ''

    def stream(self) -> Iterable[str]:
        """Iterate through the result."""
        raise NotImplementedError()

    def response(self) -> ResponseType:
        """Access the combined result - basically the chunks yielded by `stream` concatenated together and validated."""
        raise NotImplementedError()


@dataclass
class AsyncRunStreamResult(Generic[ResponseType]):
    """Result of an async streamed run."""

    history: History
    cost: Cost
    _streamed: str = ''

    async def stream(self) -> AsyncIterable[str]:
        """Iterate through the result."""
        raise NotImplementedError()

    async def response(self) -> ResponseType:
        """Access the combined result - basically the chunks yielded by `stream` concatenated together and validated."""
        raise NotImplementedError()
