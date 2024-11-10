from __future__ import annotations as _annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, Literal, Union

import pydantic
import pydantic_core

from . import _pydantic
from ._utils import now_utc as _now_utc


@dataclass
class SystemPrompt:
    """A system prompt, generally written by the application developer."""

    content: str
    """The content of the prompt."""
    role: Literal['system'] = 'system'
    """Message type identifier, this type is available on all message as a discriminator."""


@dataclass
class UserPrompt:
    """A user prompt, generally written by the end user."""

    content: str
    """The content of the prompt."""
    timestamp: datetime = field(default_factory=_now_utc)
    """The timestamp of the prompt."""
    role: Literal['user'] = 'user'
    """Message type identifier, this type is available on all message as a discriminator."""


tool_return_value_object = _pydantic.LazyTypeAdapter(dict[str, Any])


@dataclass
class ToolReturn:
    """A tool return message, this encodes the result of running a retriever."""

    tool_name: str
    """The name of the "tool" was called."""
    content: str | dict[str, Any]
    """The return value."""
    tool_id: str | None = None
    """Optional tool identifier, this is used by some models including OpenAI."""
    timestamp: datetime = field(default_factory=_now_utc)
    """The timestamp, when the tool returned."""
    role: Literal['tool-return'] = 'tool-return'
    """Message type identifier, this type is available on all message as a discriminator."""

    def model_response_str(self) -> str:
        if isinstance(self.content, str):
            return self.content
        else:
            content = tool_return_value_object.validate_python(self.content)
            return tool_return_value_object.dump_json(content).decode()

    def model_response_object(self) -> dict[str, Any]:
        if isinstance(self.content, str):
            return {'return_value': self.content}
        else:
            return tool_return_value_object.validate_python(self.content)


@dataclass
class RetryPrompt:
    """A message sent when running a retriever failed, result validation failed, or no tool could be found to call."""

    content: list[pydantic_core.ErrorDetails] | str
    """Details of why and how the model should retry.

    If the retry was triggered by a [ValidationError][pydantic_core.ValidationError], this will be a list of
    error details.
    """
    tool_name: str | None = None
    """The name of the tool that was called, if any."""
    tool_id: str | None = None
    """The tool identifier, if any."""
    timestamp: datetime = field(default_factory=_now_utc)
    """The timestamp, when the retry was triggered."""
    role: Literal['retry-prompt'] = 'retry-prompt'
    """Message type identifier, this type is available on all message as a discriminator."""

    def model_response(self) -> str:
        if isinstance(self.content, str):
            description = self.content
        else:
            description = f'{len(self.content)} validation errors: {json.dumps(self.content, indent=2)}'
        return f'{description}\n\nFix the errors and try again.'


@dataclass
class ModelTextResponse:
    """A plain text response from a model."""

    content: str
    """The text content of the response."""
    timestamp: datetime = field(default_factory=_now_utc)
    """The timestamp of the response.

    If the model provides a timestamp in the response (as OpenAI does) that will be used.
    """
    role: Literal['model-text-response'] = 'model-text-response'
    """Message type identifier, this type is available on all message as a discriminator."""


@dataclass
class ArgsJson:
    args_json: str
    """A JSON string of arguments."""


@dataclass
class ArgsObject:
    args_object: dict[str, Any]
    """A python dictionary of arguments."""


@dataclass
class ToolCall:
    """Either a retriever/tool call from the agent."""

    tool_name: str
    """The name of the tool to call."""
    args: ArgsJson | ArgsObject
    """The arguments to pass to the tool.

    Either as JSON or a Python dictionary depending on how data was returned.
    """
    tool_id: str | None = None
    """Optional tool identifier, this is used by some models including OpenAI."""

    @classmethod
    def from_json(cls, tool_name: str, args_json: str, tool_id: str | None = None) -> ToolCall:
        return cls(tool_name, ArgsJson(args_json), tool_id)

    @classmethod
    def from_object(cls, tool_name: str, args_object: dict[str, Any]) -> ToolCall:
        return cls(tool_name, ArgsObject(args_object))

    def has_content(self) -> bool:
        if isinstance(self.args, ArgsObject):
            return any(self.args.args_object.values())
        else:
            return bool(self.args.args_json)


@dataclass
class ModelStructuredResponse:
    """A structured response from a model."""

    calls: list[ToolCall]
    """The tool calls being made."""
    timestamp: datetime = field(default_factory=_now_utc)
    """The timestamp of the response.

    If the model provides a timestamp in the response (as OpenAI does) that will be used.
    """
    role: Literal['model-structured-response'] = 'model-structured-response'
    """Message type identifier, this type is available on all message as a discriminator."""


ModelAnyResponse = Union[ModelTextResponse, ModelStructuredResponse]
"""Any response from a model."""
Message = Union[SystemPrompt, UserPrompt, ToolReturn, RetryPrompt, ModelAnyResponse]
"""Any message type."""

MessagesTypeAdapter = _pydantic.LazyTypeAdapter(list[Annotated[Message, pydantic.Field(discriminator='role')]])
