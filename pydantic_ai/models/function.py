from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ..messages import LLMMessage, Message
from . import AbstractRetrieverDefinition, AgentModel, Model

if TYPE_CHECKING:
    from .._utils import ObjectJsonSchema


class FunctionDef(Protocol):
    def __call__(
        self, messages: list[Message], allow_plain_message: bool, retrievers: dict[str, RetrieverDescription], /
    ) -> LLMMessage: ...


@dataclass
class RetrieverDescription:
    name: str
    description: str
    json_schema: ObjectJsonSchema


@dataclass
class FunctionModel(Model):
    # NOTE: Avoid test discovery by pytest.
    __test__ = False

    function: FunctionDef

    def agent_model(self, allow_plain_message: bool, retrievers: list[AbstractRetrieverDefinition]) -> AgentModel:
        return TestAgentModel(
            self.function,
            allow_plain_message,
            {r.name: RetrieverDescription(r.name, r.description, r.json_schema) for r in retrievers},
        )


@dataclass
class TestAgentModel(AgentModel):
    function: FunctionDef
    allow_plain_message: bool
    retrievers: dict[str, RetrieverDescription]

    async def request(self, messages: list[Message]) -> LLMMessage:
        return self.function(messages, self.allow_plain_message, self.retrievers)
