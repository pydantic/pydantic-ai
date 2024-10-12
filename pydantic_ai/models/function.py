from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ..messages import LLMMessage, Message
from . import AbstractToolDefinition, AgentModel, Model

if TYPE_CHECKING:
    from .._utils import ObjectJsonSchema


class FunctionDef(Protocol):
    def __call__(
        self, messages: list[Message], allow_plain_response: bool, tools: dict[str, ToolDescription], /
    ) -> LLMMessage: ...


@dataclass
class ToolDescription:
    name: str
    description: str
    json_schema: ObjectJsonSchema


@dataclass
class FunctionModel(Model):
    # NOTE: Avoid test discovery by pytest.
    __test__ = False

    function: FunctionDef

    def agent_model(self, allow_plain_response: bool, tools: list[AbstractToolDefinition]) -> AgentModel:
        return FunctionAgentModel(
            self.function,
            allow_plain_response,
            {r.name: ToolDescription(r.name, r.description, r.json_schema) for r in tools},
        )


@dataclass
class FunctionAgentModel(AgentModel):
    __test__ = False

    function: FunctionDef
    allow_plain_response: bool
    tools: dict[str, ToolDescription]

    async def request(self, messages: list[Message]) -> LLMMessage:
        return self.function(messages, self.allow_plain_response, self.tools)
