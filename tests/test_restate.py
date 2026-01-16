from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

try:
    from restate import TerminalError
except ImportError:  # pragma: lax no cover
    pytest.skip('restate not installed', allow_module_level=True)

from pydantic import TypeAdapter

from pydantic_ai import Agent, RunContext
from pydantic_ai.durable_exec.restate import RestateAgent
from pydantic_ai.durable_exec.restate._model import RestateModelWrapper
from pydantic_ai.durable_exec.restate._serde import PydanticTypeAdapter
from pydantic_ai.durable_exec.restate._toolset import RestateContextRunToolSet
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import RunUsage


class FakeRestateContext:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run_typed(self, name: str, fn: Any, options: Any = None, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(name)
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


def _run_ctx() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(call_tools=[]), usage=RunUsage())


def test_pydantic_type_adapter_round_trip():
    serde = PydanticTypeAdapter(int)
    assert serde.deserialize(b'') is None
    assert serde.serialize(None) == b''

    buf = serde.serialize(123)
    assert serde.deserialize(buf) == 123


@pytest.mark.anyio
async def test_restate_context_run_toolset_success_and_error_mapping():
    fake_ctx = FakeRestateContext()
    toolset = FunctionToolset()

    async def ok(ctx: RunContext[None]) -> str:
        return 'ok'

    async def retry(ctx: RunContext[None]) -> None:
        raise ModelRetry('nope')

    async def deferred(ctx: RunContext[None]) -> None:
        raise CallDeferred()

    async def approval(ctx: RunContext[None]) -> None:
        raise ApprovalRequired()

    async def user_error(ctx: RunContext[None]) -> None:
        raise UserError('bad')

    toolset.add_function(ok, takes_ctx=True)
    toolset.add_function(retry, takes_ctx=True)
    toolset.add_function(deferred, takes_ctx=True)
    toolset.add_function(approval, takes_ctx=True)
    toolset.add_function(user_error, takes_ctx=True)

    ctx = _run_ctx()
    tools = await toolset.get_tools(ctx)

    wrapped = RestateContextRunToolSet(toolset, fake_ctx)

    assert await wrapped.call_tool('ok', {}, ctx, tools['ok']) == 'ok'
    assert 'Calling ok' in fake_ctx.calls

    with pytest.raises(ModelRetry, match='nope'):
        await wrapped.call_tool('retry', {}, ctx, tools['retry'])

    with pytest.raises(CallDeferred):
        await wrapped.call_tool('deferred', {}, ctx, tools['deferred'])

    with pytest.raises(ApprovalRequired):
        await wrapped.call_tool('approval', {}, ctx, tools['approval'])

    with pytest.raises(TerminalError, match='bad'):
        await wrapped.call_tool('user_error', {}, ctx, tools['user_error'])


@pytest.mark.anyio
async def test_restate_model_wrapper_request_and_request_stream_errors():
    fake_ctx = FakeRestateContext()
    model = TestModel(call_tools=[], custom_output_text='hi')
    wrapped = RestateModelWrapper(model, fake_ctx)

    mrp = ModelRequestParameters()
    res = await wrapped.request([], None, mrp)
    assert fake_ctx.calls == ['Model call']
    assert res.parts

    with pytest.raises(UserError, match='requires a `run_context`'):
        async with wrapped.request_stream([], None, mrp, run_context=None):
            pass

    ctx = _run_ctx()
    with pytest.raises(UserError, match='requires an `event_stream_handler`'):
        async with wrapped.request_stream([], None, mrp, run_context=ctx):
            pass


@pytest.mark.anyio
async def test_restate_agent_wraps_model_and_tool_calls():
    fake_ctx = FakeRestateContext()
    agent = Agent(TestModel())

    @agent.tool
    async def plus_one(ctx: RunContext, x: int) -> int:
        return x + 1

    restate_agent = RestateAgent(agent, fake_ctx)
    result = await restate_agent.run('go')

    assert result.output
    assert fake_ctx.calls.count('Model call') >= 2
    assert 'Calling plus_one' in fake_ctx.calls


@pytest.mark.anyio
async def test_restate_agent_disable_auto_wrapping_tools():
    fake_ctx = FakeRestateContext()
    agent = Agent(TestModel())

    @agent.tool
    async def plus_one(ctx: RunContext, x: int) -> int:
        return x + 1

    restate_agent = RestateAgent(agent, fake_ctx, disable_auto_wrapping_tools=True)
    await restate_agent.run('go')

    assert fake_ctx.calls.count('Model call') >= 2
    assert 'Calling plus_one' not in fake_ctx.calls


@pytest.mark.anyio
async def test_restate_agent_event_stream_handler_and_iter():
    fake_ctx = FakeRestateContext()
    seen_event_kinds: list[str] = []

    async def event_stream_handler(ctx: RunContext[Any], stream: AsyncIterator[Any]) -> None:
        async for event in stream:
            seen_event_kinds.append(getattr(event, 'event_kind', 'unknown'))

    agent = Agent(TestModel())

    @agent.tool
    async def plus_one(ctx: RunContext, x: int) -> int:
        return x + 1

    restate_agent = RestateAgent(agent, fake_ctx, event_stream_handler=event_stream_handler)
    await restate_agent.run('go')

    # Model calls should use the durable stream path when an event handler is present.
    assert fake_ctx.calls.count('Model stream call') >= 2
    # Tool execution is still durable by default.
    assert 'Calling plus_one' in fake_ctx.calls
    # Tool/model events are executed via ctx.run_typed() per event.
    assert 'run event' in fake_ctx.calls
    assert seen_event_kinds

    fake_ctx.calls.clear()
    async with restate_agent.iter('go') as agent_run:
        async for _ in agent_run:
            pass
    assert fake_ctx.calls


@pytest.mark.anyio
async def test_restate_agent_restrictions():
    fake_ctx = FakeRestateContext()

    agent_without_model = Agent()
    with pytest.raises(TerminalError, match='needs to have a `model`'):
        RestateAgent(agent_without_model, fake_ctx)

    agent = Agent(TestModel())
    restate_agent = RestateAgent(agent, fake_ctx)

    with pytest.raises(TerminalError, match='cannot be set at agent run time'):
        await restate_agent.run('x', model=TestModel())

    async def handler(_: RunContext[Any], __: AsyncIterator[Any]) -> None:
        return None

    with pytest.raises(TerminalError, match='Event stream handler cannot be set'):
        await restate_agent.run('x', event_stream_handler=handler)

    with pytest.raises(TerminalError, match=r'agent\.run_stream\(\)'):
        async with restate_agent.run_stream('x'):
            pass

    with pytest.raises(TerminalError, match=r'agent\.run_stream_events\(\)'):
        restate_agent.run_stream_events('x')


@pytest.mark.anyio
async def test_restate_mcp_server_wrapping_and_agent_mcp_wrapping():
    mcp = pytest.importorskip('pydantic_ai.mcp')
    from pydantic_ai.durable_exec.restate._mcp_server import RestateMCPServer

    MCPServer = mcp.MCPServer

    # Create a minimal concrete MCPServer that doesn't actually connect anywhere.
    class FakeMCPServer(MCPServer):
        @asynccontextmanager
        async def client_streams(self):  # type: ignore[override]
            raise RuntimeError('not used')
            yield  # pragma: no cover

        async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
            tool_def = mcp.ToolDefinition(  # type: ignore[attr-defined]
                name='echo',
                description='echo',
                parameters_json_schema={'type': 'object', 'properties': {'value': {}}, 'required': ['value']},
            )
            return {
                'echo': ToolsetTool[Any](
                    toolset=self,
                    tool_def=tool_def,
                    max_retries=1,
                    args_validator=TypeAdapter(dict[str, Any]).validator,
                )
            }

        async def call_tool(
            self,
            name: str,
            tool_args: dict[str, Any],
            ctx: RunContext[Any],
            tool: ToolsetTool[Any],
        ) -> Any:
            return {'name': name, 'args': tool_args}

    fake_ctx = FakeRestateContext()
    fake_server = FakeMCPServer(tool_prefix='fake')

    restate_server = RestateMCPServer(fake_server, fake_ctx)  # type: ignore[arg-type]
    restate_server = restate_server.visit_and_replace(lambda t: t)  # cover visit_and_replace

    ctx = _run_ctx()
    tools = await restate_server.get_tools(ctx)
    assert fake_ctx.calls == ['get mcp tools']
    assert 'echo' in tools

    fake_ctx.calls.clear()
    result = await restate_server.call_tool('echo', {'value': 123}, ctx, tools['echo'])
    assert fake_ctx.calls == ['Calling mcp tool echo']
    assert result == {'name': 'echo', 'args': {'value': 123}}

    # Ensure RestateAgent hits the MCP wrapping branch when an MCPServer toolset is present.
    fake_ctx.calls.clear()
    agent = Agent(TestModel(call_tools=[]), toolsets=[fake_server])
    RestateAgent(agent, fake_ctx)
    assert fake_ctx.calls == []

