from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Literal

import pydantic
import pydantic_core


@dataclass
@dataclass
class SystemPrompt:
    content: str
    role: Literal['system'] = 'system'


@dataclass
class UserPrompt:
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    role: Literal['user'] = 'user'


@dataclass
class FunctionResponse:
    function_id: str
    function_name: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    role: Literal['function-response'] = 'function-response'


@dataclass
class FunctionValidationError:
    function_id: str
    function_name: str
    errors: list[pydantic_core.ErrorDetails]
    timestamp: datetime = field(default_factory=datetime.now)
    role: Literal['function-validation-error'] = 'function-validation-error'


@dataclass
class PlainResponseForbidden:
    timestamp: datetime = field(default_factory=datetime.now)
    role: Literal['plain-response-forbidden'] = 'plain-response-forbidden'


@dataclass
class LLMResponse:
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    role: Literal['llm-response'] = 'llm-response'


@dataclass
class FunctionCall:
    function_id: str
    function_name: str
    arguments: str


@dataclass
class LLMFunctionCalls:
    calls: list[FunctionCall]
    timestamp: datetime = field(default_factory=datetime.now)
    role: Literal['llm-function-calls'] = 'llm-function-calls'


LLMMessage = LLMResponse | LLMFunctionCalls
Message = SystemPrompt | UserPrompt | FunctionResponse | FunctionValidationError | PlainResponseForbidden | LLMMessage

MessagesTypeAdapter = pydantic.TypeAdapter(list[Annotated[Message, pydantic.Field(discriminator='role')]])
