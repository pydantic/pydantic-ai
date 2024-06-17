from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ..messages import LLMMessage, Message
from . import AbstractToolDefinition, AgentModel, Model

if TYPE_CHECKING:
    from .._utils import ObjectJsonSchema


class FunctionDef(Protocol):
    def __call__(self, messages: list[Message], allow_plain_message: bool, tools: dict[str, Tool], /) -> LLMMessage: ...


@dataclass
class Tool:
    name: str
    description: str
    json_schema: ObjectJsonSchema


@dataclass
class FunctionModel(Model):
    # NOTE: Avoid test discovery by pytest.
    __test__ = False

    function: FunctionDef

    def agent_model(self, allow_plain_message: bool, tools: list[AbstractToolDefinition]) -> AgentModel:
        return TestAgentModel(
            self.function, allow_plain_message, {t.name: Tool(t.name, t.description, t.json_schema) for t in tools}
        )


@dataclass
class TestAgentModel(AgentModel):
    function: FunctionDef
    allow_plain_message: bool
    tools: dict[str, Tool]

    async def request(self, messages: list[Message]) -> LLMMessage:
        return self.function(messages, self.allow_plain_message, self.tools)
