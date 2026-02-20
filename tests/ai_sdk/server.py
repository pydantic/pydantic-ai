"""Starlette server for AI SDK E2E integration testing.

Provides multiple endpoints exercising different response shapes:
text, thinking, tool calls (with and without approval), and combinations.
Each agent uses TestModel which naturally calls tools and produces responses.
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

# --- Text-only ---

text_agent = Agent(model=TestModel(custom_output_text='Hello, world!'), output_type=str)


# --- Thinking + text (requires FunctionModel since TestModel doesn't produce thinking) ---


async def thinking_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaThinkingCalls | str]:
    yield {0: DeltaThinkingPart(content='Let me think about this... The answer is clear.')}
    yield 'The answer is 42.'


thinking_agent = Agent(model=FunctionModel(stream_function=thinking_stream), output_type=str)


# --- Tool (no approval) ---

tool_agent: Agent[None, str] = Agent(model=TestModel(), output_type=str)


@tool_agent.tool_plain
def get_weather(city: str) -> str:
    return f'Sunny in {city}'


# --- Tool with approval ---

approval_agent: Agent[None, str | DeferredToolRequests] = Agent(
    model=TestModel(),
    output_type=[str, DeferredToolRequests],
)


@approval_agent.tool_plain(requires_approval=True)
def delete_file(path: str) -> str:
    return f'Deleted {path}'


# --- Multiple tools ---

multi_tool_agent: Agent[None, str] = Agent(model=TestModel(), output_type=str)


@multi_tool_agent.tool_plain
def lookup_weather(city: str) -> str:
    return f'Rainy in {city}'


@multi_tool_agent.tool_plain
def lookup_time(timezone: str) -> str:
    return f'12:00 {timezone}'


# --- Routes ---


async def text_endpoint(request: Request) -> Response:
    return await VercelAIAdapter.dispatch_request(request, agent=text_agent, sdk_version=6)


async def thinking_endpoint(request: Request) -> Response:
    return await VercelAIAdapter.dispatch_request(request, agent=thinking_agent, sdk_version=6)


async def tool_endpoint(request: Request) -> Response:
    return await VercelAIAdapter.dispatch_request(request, agent=tool_agent, sdk_version=6)


async def approval_endpoint(request: Request) -> Response:
    return await VercelAIAdapter.dispatch_request(request, agent=approval_agent, sdk_version=6)


async def multi_tool_endpoint(request: Request) -> Response:
    return await VercelAIAdapter.dispatch_request(request, agent=multi_tool_agent, sdk_version=6)


app = Starlette(
    routes=[
        Route('/api/chat/text', text_endpoint, methods=['POST']),
        Route('/api/chat/thinking', thinking_endpoint, methods=['POST']),
        Route('/api/chat/tool', tool_endpoint, methods=['POST']),
        Route('/api/chat/approval', approval_endpoint, methods=['POST']),
        Route('/api/chat/multi-tool', multi_tool_endpoint, methods=['POST']),
    ]
)


if __name__ == '__main__':
    import uvicorn

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    uvicorn.run(app, host='127.0.0.1', port=port, log_level='warning')
