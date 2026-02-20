"""Minimal Starlette server for AI SDK integration testing."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai.ui.vercel_ai import VercelAIAdapter


def _has_tool_return(messages: list[ModelMessage]) -> bool:
    return any(
        isinstance(part, ToolReturnPart)
        for msg in messages
        if isinstance(msg, ModelRequest)
        for part in msg.parts
    )


async def stream_function(messages: list[ModelMessage], agent_info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
    if _has_tool_return(messages):
        yield 'Done.'
    else:
        yield {
            0: DeltaToolCall(
                name='delete_file',
                json_args='{"path": "test.txt"}',
                tool_call_id='delete_1',
            )
        }


agent: Agent[None, str | DeferredToolRequests] = Agent(
    model=FunctionModel(stream_function=stream_function),
    output_type=[str, DeferredToolRequests],
)


@agent.tool_plain(requires_approval=True)
def delete_file(path: str) -> str:
    return f'Deleted {path}'


async def chat_endpoint(request: Request) -> Response:
    return await VercelAIAdapter.dispatch_request(request, agent=agent, sdk_version=6)


app = Starlette(routes=[Route('/api/chat', chat_endpoint, methods=['POST'])])


if __name__ == '__main__':
    import uvicorn

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    uvicorn.run(app, host='127.0.0.1', port=port, log_level='warning')
