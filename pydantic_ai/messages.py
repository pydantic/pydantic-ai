from __future__ import annotations as _annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, Literal, Union

import pydantic
import pydantic_core

from . import _pydantic


@dataclass
class SystemPrompt:
    content: str
    role: Literal['system'] = 'system'


@dataclass
class UserPrompt:
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    role: Literal['user'] = 'user'


return_value_object = _pydantic.LazyTypeAdapter(dict[str, Any])


@dataclass
class ToolReturn:
    tool_name: str
    content: str | dict[str, Any]
    tool_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    role: Literal['tool-return'] = 'tool-return'

    def model_response_str(self) -> str:
        if isinstance(self.content, str):
            return self.content
        else:
            content = return_value_object.validate_python(self.content)
            return return_value_object.dump_json(content).decode()

    def model_response_object(self) -> dict[str, Any]:
        if isinstance(self.content, str):
            return {'return_value': self.content}
        else:
            return return_value_object.validate_python(self.content)


@dataclass
class RetryPrompt:
    content: list[pydantic_core.ErrorDetails] | str
    tool_name: str | None = None
    tool_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    role: Literal['retry-prompt'] = 'retry-prompt'

    def model_response(self) -> str:
        if isinstance(self.content, str):
            description = self.content
        else:
            description = f'{len(self.content)} validation errors: {json.dumps(self.content, indent=2)}'
        return f'{description}\n\nFix the errors and try again.'


@dataclass
class LLMResponse:
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    role: Literal['llm-response'] = 'llm-response'


@dataclass
class ArgsJson:
    args_json: str


@dataclass
class ArgsObject:
    args_object: dict[str, Any]


@dataclass
class ToolCall:
    """Either a retriever/tool call or structured response from the agent."""

    tool_name: str
    args: ArgsJson | ArgsObject
    tool_id: str | None = None

    @classmethod
    def from_json(cls, tool_name: str, args_json: str, tool_id: str | None = None) -> ToolCall:
        return cls(tool_name, ArgsJson(args_json), tool_id)

    @classmethod
    def from_object(cls, tool_name: str, args_object: dict[str, Any]) -> ToolCall:
        return cls(tool_name, ArgsObject(args_object))


@dataclass
class LLMToolCalls:
    calls: list[ToolCall]
    timestamp: datetime = field(default_factory=datetime.now)
    role: Literal['llm-tool-calls'] = 'llm-tool-calls'


LLMMessage = Union[LLMResponse, LLMToolCalls]
Message = Union[SystemPrompt, UserPrompt, ToolReturn, RetryPrompt, LLMMessage]

MessagesTypeAdapter = pydantic.TypeAdapter(list[Annotated[Message, pydantic.Field(discriminator='role')]])
