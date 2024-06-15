from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterable, Generic, Literal, TypedDict, TypeVar

import pydantic_core


@dataclass
class Cost:
    """Cost of a run."""

    total_cost: int


class SystemPrompt(TypedDict):
    role: Literal['system']
    content: str


class UserPrompt(TypedDict):
    role: Literal['user']
    timestamp: datetime
    content: str


class FunctionResponse(TypedDict):
    role: Literal['function-response']
    timestamp: datetime
    function_id: str
    function_name: str
    response: str


class FunctionValidationError(TypedDict):
    role: Literal['function-validation-error']
    timestamp: datetime
    function_id: str
    function_name: str
    errors: list[pydantic_core.ErrorDetails]


class LLMResponse(TypedDict):
    role: Literal['llm-response']
    timestamp: datetime
    content: str


class FunctionCall(TypedDict):
    function_id: str
    function_name: str
    arguments: str


class LLMFunctionCalls(TypedDict):
    role: Literal['llm-function-calls']
    timestamp: datetime
    calls: list[FunctionCall]


# TODO FunctionRunError?
LLMMessage = LLMResponse | LLMFunctionCalls
Message = SystemPrompt | UserPrompt | FunctionResponse | FunctionValidationError | LLMMessage


ResponseType = TypeVar('ResponseType')


@dataclass
class RunResult(Generic[ResponseType]):
    """Result of a run."""

    response: ResponseType
    message_history: list[Message]
    cost: Cost

    def message_history_json(self) -> str:
        """Return the history of messages as a JSON string."""
        return pydantic_core.to_json(self.message_history).decode()


@dataclass
class RunStreamResult(Generic[ResponseType]):
    """Result of an async streamed run."""

    # history: History
    cost: Cost
    _streamed: str = ''

    async def stream(self) -> AsyncIterable[str]:
        """Iterate through the result."""
        raise NotImplementedError()

    async def response(self) -> ResponseType:
        """Access the combined result - basically the chunks yielded by `stream` concatenated together and validated."""
        raise NotImplementedError()
