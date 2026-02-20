"""Starlette server for AI SDK E2E integration testing.

Takes an agent name as a CLI argument and serves it at /api/chat.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaThinkingCalls, DeltaThinkingPart, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

# --- Agents ---

text_agent = Agent(model=TestModel(custom_output_text='Hello, world!'), output_type=str)


async def _thinking_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaThinkingCalls | str]:
    yield {0: DeltaThinkingPart(content='Let me think about this... The answer is clear.')}
    yield 'The answer is 42.'


thinking_agent = Agent(model=FunctionModel(stream_function=_thinking_stream), output_type=str)

tool_agent: Agent[None, str] = Agent(model=TestModel(), output_type=str)


@tool_agent.tool_plain
def get_weather(city: str) -> str:
    return f'Sunny in {city}'


approval_agent: Agent[None, str | DeferredToolRequests] = Agent(
    model=TestModel(),
    output_type=[str, DeferredToolRequests],
)


@approval_agent.tool_plain(requires_approval=True)
def delete_file(path: str) -> str:
    return f'Deleted {path}'


multi_tool_agent: Agent[None, str] = Agent(model=TestModel(), output_type=str)


@multi_tool_agent.tool_plain
def lookup_weather(city: str) -> str:
    return f'Rainy in {city}'


@multi_tool_agent.tool_plain
def lookup_time(timezone: str) -> str:
    return f'12:00 {timezone}'


AGENTS: dict[str, Agent[None, ...]] = {
    'text': text_agent,
    'thinking': thinking_agent,
    'tool': tool_agent,
    'tool_approval': approval_agent,
    'multi_tool': multi_tool_agent,
}


def create_app(agent: Agent[None, ...]) -> Starlette:
    async def chat_endpoint(request: Request) -> Response:
        return await VercelAIAdapter.dispatch_request(request, agent=agent, sdk_version=6)

    return Starlette(routes=[Route('/api/chat', chat_endpoint, methods=['POST'])])


if __name__ == '__main__':
    import uvicorn

    agent_name = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    agent = AGENTS[agent_name]
    uvicorn.run(create_app(agent), host='127.0.0.1', port=port, log_level='warning')
