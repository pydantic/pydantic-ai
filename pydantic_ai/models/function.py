from __future__ import annotations as _annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from typing_extensions import TypeAlias

from ..messages import LLMMessage, Message
from . import AbstractToolDefinition, AgentModel, Model

if TYPE_CHECKING:
    from .._utils import ObjectJsonSchema


@dataclass(frozen=True)
class AgentInfo:
    """Information about an agent passed to a function."""

    retrievers: Mapping[str, AbstractToolDefinition]
    allow_text_result: bool
    result_tool: AbstractToolDefinition | None


FunctionDef: TypeAlias = Callable[[list[Message], AgentInfo], LLMMessage]


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

    def agent_model(
        self,
        retrievers: Mapping[str, AbstractToolDefinition],
        allow_text_result: bool,
        result_tool: AbstractToolDefinition | None,
    ) -> AgentModel:
        return FunctionAgentModel(self.function, AgentInfo(retrievers, allow_text_result, result_tool))

    def name(self) -> str:
        return f'function:{self.function.__name__}'


@dataclass
class FunctionAgentModel(AgentModel):
    function: FunctionDef
    agent_info: AgentInfo

    async def request(self, messages: list[Message]) -> LLMMessage:
        return self.function(messages, self.agent_info)
