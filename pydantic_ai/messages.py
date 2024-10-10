from __future__ import annotations as _annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Literal, Union

import pydantic
import pydantic_core


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
class FunctionReturn:
    function_id: str
    function_name: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    role: Literal['function-return'] = 'function-return'

    def llm_response(self) -> str:
        # return f'Response from calling {m.function_name}: {m.content}'
        return self.content


@dataclass
class FunctionRetry:
    function_id: str
    function_name: str
    content: list[pydantic_core.ErrorDetails] | str
    timestamp: datetime = field(default_factory=datetime.now)
    role: Literal['function-retry'] = 'function-retry'

    def llm_response(self) -> str:
        if isinstance(self.content, str):
            description = self.content
        else:
            description = f'{len(self.content)} validation errors: {json.dumps(self.content, indent=2)}'
        return f'{description}\n\nFix the errors and try again.'


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


LLMMessage = Union[LLMResponse, LLMFunctionCalls]
Message = Union[SystemPrompt, UserPrompt, FunctionReturn, FunctionRetry, PlainResponseForbidden, LLMMessage]

MessagesTypeAdapter = pydantic.TypeAdapter(list[Annotated[Message, pydantic.Field(discriminator='role')]])
