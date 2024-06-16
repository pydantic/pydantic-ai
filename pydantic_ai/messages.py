from datetime import datetime
from typing import Annotated, Literal, TypedDict

import pydantic
import pydantic_core


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
    content: str


class FunctionValidationError(TypedDict):
    role: Literal['function-validation-error']
    timestamp: datetime
    function_id: str
    function_name: str
    errors: list[pydantic_core.ErrorDetails]


class PlainResponseForbidden(TypedDict):
    role: Literal['plain-response-forbidden']
    timestamp: datetime


def plain_response_forbidden() -> PlainResponseForbidden:
    return PlainResponseForbidden(role='plain-response-forbidden', timestamp=datetime.now())


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
Message = SystemPrompt | UserPrompt | FunctionResponse | FunctionValidationError | PlainResponseForbidden | LLMMessage

MessagesTypeAdapter = pydantic.TypeAdapter(list[Annotated[Message, pydantic.Field(discriminator='role')]])
