from __future__ import annotations as _annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, Literal, Union

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
class ToolReturn:
    tool_name: str
    content: str
    tool_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    role: Literal['tool-return'] = 'tool-return'

    def llm_response(self) -> str:
        return self.content


@dataclass
class ToolRetry:
    tool_name: str
    content: list[pydantic_core.ErrorDetails] | str
    tool_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    role: Literal['tool-retry'] = 'tool-retry'

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

    @staticmethod
    def llm_response() -> str:
        return 'Plain text responses are not allowed, please call one of the functions instead.'


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
    """
    Either a retriever/tool call or structured response from the agent.
    """

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
Message = Union[SystemPrompt, UserPrompt, ToolReturn, ToolRetry, PlainResponseForbidden, LLMMessage]

MessagesTypeAdapter = pydantic.TypeAdapter(list[Annotated[Message, pydantic.Field(discriminator='role')]])
