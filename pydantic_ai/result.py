from dataclasses import dataclass
from typing import Generic, TypeVar, Iterable, AsyncIterable


@dataclass
class Cost:
    total_cost: int


@dataclass
class History:
    messages: list[str]

    def as_json(self) -> str:
        return str(self.messages)


ResponseType = TypeVar('ResponseType')


@dataclass
class RunResult(Generic[ResponseType]):
    response: ResponseType
    history: History
    cost: Cost


@dataclass
class RunStreamResult(Generic[ResponseType]):
    history: History
    cost: Cost
    _streamed: str = ''

    def stream(self) -> Iterable[str]:
        raise NotImplementedError()

    def response(self) -> ResponseType:
        raise NotImplementedError()


@dataclass
class AsyncRunStreamResult(Generic[ResponseType]):
    history: History
    cost: Cost
    _streamed: str = ''

    async def stream(self) -> AsyncIterable[str]:
        raise NotImplementedError()

    async def response(self) -> ResponseType:
        raise NotImplementedError()
