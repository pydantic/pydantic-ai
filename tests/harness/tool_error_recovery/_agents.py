"""Agent builders shared by the behavior tests."""

from __future__ import annotations

from typing import Any

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


def tool_returns(messages: list[Any]) -> int:
    return sum(1 for m in messages for p in m.parts if isinstance(p, ToolReturnPart))


def call_then_echo(max_calls: int = 1) -> FunctionModel:
    """Model that calls `do_thing` up to `max_calls` times, then echoes the last tool result."""

    def fn(messages: list[Any], info: AgentInfo) -> ModelResponse:
        for m in reversed(messages):
            for p in m.parts:
                if isinstance(p, ToolReturnPart):
                    if tool_returns(messages) >= max_calls:
                        return ModelResponse(parts=[TextPart(content=f'RESULT:{p.content}')])
                    break
        return ModelResponse(parts=[ToolCallPart(tool_name='do_thing', args={})])

    return FunctionModel(fn)


def build(cap: AbstractCapability[Any], tool_fn: Any, *, max_calls: int = 1) -> Agent[None, str]:
    agent: Agent[None, str] = Agent(call_then_echo(max_calls), capabilities=[cap])
    agent.tool_plain(name='do_thing')(tool_fn)
    return agent
