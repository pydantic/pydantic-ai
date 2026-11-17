"""An in-process MCP server that counts the protocol methods a durable run reaches it with.

Round trips are the only way to see what a durable run costs an MCP server, and counting them
server-side is what attributes them correctly: the MCP SDK issues its own `tools/list` before a
`tools/call` whenever the session's output-schema cache is cold, so a count taken on our side of the
wire would credit that listing to the engine.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext

from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.messages import ModelMessage, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


def counting_mcp_server(*, instructions: str | None = None) -> tuple[FastMCP[None], dict[str, int]]:
    """A server exposing one `echo` tool, alongside the counts of the methods it is called with."""
    counts: dict[str, int] = {}
    server: FastMCP[None] = FastMCP('counting', instructions=instructions)

    @server.tool
    def echo(text: str) -> str:
        return f'echo: {text}'

    class Counter(Middleware):
        async def on_message(self, context: MiddlewareContext[Any], call_next: Any) -> Any:
            method = context.method or '<unknown>'
            counts[method] = counts.get(method, 0) + 1
            return await call_next(context)

    server.add_middleware(Counter())
    return server, counts


def two_echo_calls_model() -> FunctionModel:
    """A model that calls `echo` in each of its first two requests, then answers: three requests."""
    step = 0

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal step
        step += 1
        if step <= 2:
            return ModelResponse(parts=[ToolCallPart('echo', {'text': f'hi {step}'})])
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(model)
