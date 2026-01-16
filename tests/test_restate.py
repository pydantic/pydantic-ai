from __future__ import annotations

import inspect
from collections.abc import AsyncIterable
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
from pydantic_ai.models import Model
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.toolsets.external import TOOL_SCHEMA_VALIDATOR
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
        raise CallDeferred(metadata={'foo': 'bar'})

    async def approval(ctx: RunContext[None]) -> None:
        raise ApprovalRequired(metadata={'hello': 'world'})

    async def user_error(ctx: RunContext[None]) -> None:
        raise UserError('bad')

    toolset.add_function(ok, takes_ctx=True)
    toolset.add_function(retry, takes_ctx=True)
    toolset.add_function(deferred, takes_ctx=True)
    toolset.add_function(approval, takes_ctx=True)
    toolset.add_function(user_error, takes_ctx=True)

    ctx = _run_ctx()
    tools = await toolset.get_tools(ctx)

    wrapped = RestateContextRunToolSet(toolset, fake_ctx)  # type: ignore[arg-type]

    assert await wrapped.call_tool('ok', {}, ctx, tools['ok']) == 'ok'
    assert 'Calling ok' in fake_ctx.calls

    with pytest.raises(ModelRetry, match='nope'):
        await wrapped.call_tool('retry', {}, ctx, tools['retry'])

    with pytest.raises(CallDeferred) as exc_info:
        await wrapped.call_tool('deferred', {}, ctx, tools['deferred'])
    assert exc_info.value.metadata == {'foo': 'bar'}

    with pytest.raises(ApprovalRequired) as exc_info:
        await wrapped.call_tool('approval', {}, ctx, tools['approval'])
    assert exc_info.value.metadata == {'hello': 'world'}

    with pytest.raises(TerminalError, match='bad'):
        await wrapped.call_tool('user_error', {}, ctx, tools['user_error'])


@pytest.mark.anyio
async def test_restate_model_wrapper_request_and_request_stream_errors():
    fake_ctx = FakeRestateContext()
    model = TestModel(call_tools=[], custom_output_text='hi')
    wrapped = RestateModelWrapper(model, fake_ctx)  # type: ignore[arg-type]

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
async def test_restate_model_wrapper_maps_user_error_to_terminal_error():
    class ErrorModel(Model):
        async def request(self, *args: Any, **kwargs: Any):
            raise UserError('bad')

        @property
        def model_name(self) -> str:
            return 'error-model'

        @property
        def system(self) -> str:
            return 'test'

    fake_ctx = FakeRestateContext()
    wrapped = RestateModelWrapper(ErrorModel(), fake_ctx)  # type: ignore[arg-type]
    with pytest.raises(TerminalError, match='bad'):
        await wrapped.request([], None, ModelRequestParameters())


@pytest.mark.anyio
async def test_restate_agent_wraps_model_and_tool_calls():
    fake_ctx = FakeRestateContext()
    agent = Agent(TestModel())

    @agent.tool
    async def plus_one(ctx: RunContext, x: int) -> int:
        return x + 1

    restate_agent = RestateAgent(agent, fake_ctx)  # type: ignore[arg-type]
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

    restate_agent = RestateAgent(agent, fake_ctx, disable_auto_wrapping_tools=True)  # type: ignore[arg-type]
    await restate_agent.run('go')

    assert fake_ctx.calls.count('Model call') >= 2
    assert 'Calling plus_one' not in fake_ctx.calls


@pytest.mark.anyio
async def test_restate_agent_event_stream_handler_and_iter():
    fake_ctx = FakeRestateContext()
    seen_event_kinds: list[str] = []

    async def event_stream_handler(ctx: RunContext[Any], stream: AsyncIterable[Any]) -> None:
        async for event in stream:
            seen_event_kinds.append(getattr(event, 'event_kind', 'unknown'))

    agent = Agent(TestModel())

    @agent.tool
    async def plus_one(ctx: RunContext, x: int) -> int:
        return x + 1

    restate_agent = RestateAgent(agent, fake_ctx, event_stream_handler=event_stream_handler)  # type: ignore[arg-type]
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
        RestateAgent(agent_without_model, fake_ctx)  # type: ignore[arg-type]

    agent = Agent(TestModel())
    restate_agent = RestateAgent(agent, fake_ctx)  # type: ignore[arg-type]

    with pytest.raises(TerminalError, match='cannot be set at agent run time'):
        await restate_agent.run('x', model=TestModel())

    async def handler(_: RunContext[Any], __: AsyncIterable[Any]) -> None:
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
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.enter_calls = 0
            self.exit_calls = 0

        @asynccontextmanager
        async def client_streams(self):
            raise RuntimeError('not used')
            yield  # pragma: no cover

        async def __aenter__(self):
            self.enter_calls += 1
            return self

        async def __aexit__(self, *args: Any) -> bool | None:
            self.exit_calls += 1
            return None

        async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
            async with self:
                tool_defs = {
                    name: mcp.ToolDefinition(
                        name=name,
                        description=name,
                        parameters_json_schema={'type': 'object', 'properties': {'value': {}}, 'required': ['value']},
                    )
                    for name in ('echo', 'retry', 'deferred', 'approval')
                }
                return {
                    name: ToolsetTool[Any](
                        toolset=self,
                        tool_def=tool_def,
                        max_retries=1,
                        args_validator=TypeAdapter(dict[str, Any]).validator,  # type: ignore[arg-type]
                    )
                    for name, tool_def in tool_defs.items()
                }

        async def call_tool(
            self,
            name: str,
            tool_args: dict[str, Any],
            ctx: RunContext[Any],
            tool: ToolsetTool[Any],
        ) -> Any:
            async with self:
                if name == 'retry':
                    raise ModelRetry('nope')
                elif name == 'deferred':
                    raise CallDeferred(metadata={'foo': 'bar'})
                elif name == 'approval':
                    raise ApprovalRequired(metadata={'hello': 'world'})
                return {'name': name, 'args': tool_args}

    fake_ctx = FakeRestateContext()
    fake_server = FakeMCPServer(tool_prefix='fake')

    restate_server = RestateMCPServer(fake_server, fake_ctx)  # type: ignore[arg-type]
    restate_server = restate_server.visit_and_replace(lambda t: t)  # cover visit_and_replace

    # Entering the Restate wrapper should be a no-op (no I/O outside ctx.run_typed).
    async with restate_server:
        pass
    assert fake_server.enter_calls == 0

    ctx = _run_ctx()
    tools = await restate_server.get_tools(ctx)
    assert fake_ctx.calls == ['get mcp tools']
    assert 'echo' in tools
    assert fake_server.enter_calls == 1
    assert fake_server.exit_calls == 1

    fake_ctx.calls.clear()
    result = await restate_server.call_tool('echo', {'value': 123}, ctx, tools['echo'])
    assert fake_ctx.calls == ['Calling mcp tool echo']
    assert result == {'name': 'echo', 'args': {'value': 123}}
    assert fake_server.enter_calls == 2
    assert fake_server.exit_calls == 2

    with pytest.raises(ModelRetry, match='nope'):
        await restate_server.call_tool('retry', {'value': 1}, ctx, tools['retry'])

    with pytest.raises(CallDeferred) as exc_info:
        await restate_server.call_tool('deferred', {'value': 1}, ctx, tools['deferred'])
    assert exc_info.value.metadata == {'foo': 'bar'}

    with pytest.raises(ApprovalRequired) as exc_info:
        await restate_server.call_tool('approval', {'value': 1}, ctx, tools['approval'])
    assert exc_info.value.metadata == {'hello': 'world'}

    # Ensure RestateAgent hits the MCP wrapping branch when an MCPServer toolset is present.
    fake_ctx.calls.clear()
    agent = Agent(TestModel(call_tools=[]), toolsets=[fake_server])
    RestateAgent(agent, fake_ctx)  # type: ignore[arg-type]
    assert fake_ctx.calls == []


@pytest.mark.anyio
async def test_restate_dynamic_toolset_is_durable_and_revalidates_args():
    from pydantic_ai.durable_exec.restate._dynamic_toolset import RestateDynamicToolset
    from pydantic_ai.toolsets._dynamic import DynamicToolset

    fake_ctx = FakeRestateContext()

    def toolset_func(ctx: RunContext[None]) -> FunctionToolset[None]:
        toolset = FunctionToolset[None](id='dynamic')

        @toolset.tool
        async def add_one(x: int) -> int:
            return x + 1

        return toolset

    dynamic = DynamicToolset[None](toolset_func=toolset_func, id='dyn')
    durable_dynamic = RestateDynamicToolset(dynamic, fake_ctx, disable_auto_wrapping_tools=False)  # type: ignore[arg-type]

    ctx = _run_ctx()
    tools = await durable_dynamic.get_tools(ctx)
    assert 'get dynamic tools' in fake_ctx.calls
    assert 'add_one' in tools
    assert tools['add_one'].toolset is durable_dynamic
    assert tools['add_one'].args_validator is TOOL_SCHEMA_VALIDATOR

    # Re-validation happens inside ctx.run_typed, so string input is coerced to int.
    fake_ctx.calls.clear()
    assert await durable_dynamic.call_tool('add_one', {'x': '1'}, ctx, tools['add_one']) == 2
    assert fake_ctx.calls == ['Calling dynamic tool add_one']

    with pytest.raises(ModelRetry):
        await durable_dynamic.call_tool('add_one', {'x': 'not-an-int'}, ctx, tools['add_one'])


@pytest.mark.anyio
async def test_restate_fastmcp_toolset_wrapping_smoke():
    pytest.importorskip('pydantic_ai.toolsets.fastmcp')
    fastmcp = pytest.importorskip('fastmcp')

    from fastmcp import FastMCP
    from pydantic_ai.durable_exec.restate._fastmcp_toolset import RestateFastMCPToolset
    from pydantic_ai.toolsets.fastmcp import FastMCPToolset

    mcp = FastMCP('test')

    @mcp.tool
    def echo(value: int) -> dict[str, Any]:
        return {'value': value}

    class InstrumentedFastMCPToolset(FastMCPToolset[Any]):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.enter_calls = 0
            self.exit_calls = 0

        async def __aenter__(self):
            self.enter_calls += 1
            return await super().__aenter__()

        async def __aexit__(self, *args: Any) -> bool | None:
            self.exit_calls += 1
            return await super().__aexit__(*args)

    toolset = InstrumentedFastMCPToolset(mcp, id='fastmcp')
    fake_ctx = FakeRestateContext()
    durable_toolset = RestateFastMCPToolset(toolset, fake_ctx)  # type: ignore[arg-type]

    # Entering the Restate wrapper should be a no-op (no I/O outside ctx.run_typed).
    async with durable_toolset:
        pass
    assert toolset.enter_calls == 0

    ctx = _run_ctx()
    tools = await durable_toolset.get_tools(ctx)
    assert 'get fastmcp tools' in fake_ctx.calls
    assert 'echo' in tools
    assert toolset.enter_calls == 1
    assert toolset.exit_calls == 1

    fake_ctx.calls.clear()
    result = await durable_toolset.call_tool('echo', {'value': 123}, ctx, tools['echo'])
    assert fake_ctx.calls == ['Calling fastmcp tool echo']
    assert isinstance(result, dict)
    assert result.get('value') == 123
    assert toolset.enter_calls == 2
    assert toolset.exit_calls == 2


@pytest.mark.anyio
async def test_restate_agent_wraps_dynamic_toolset():
    from pydantic_ai.toolsets._dynamic import DynamicToolset

    fake_ctx = FakeRestateContext()

    def toolset_func(ctx: RunContext[None]) -> FunctionToolset[None]:
        toolset = FunctionToolset[None](id='dynamic')

        @toolset.tool
        async def plus_one(x: int) -> int:
            return x + 1

        return toolset

    agent = Agent(TestModel(), toolsets=[DynamicToolset(toolset_func=toolset_func, id='dyn')])
    restate_agent = RestateAgent(agent, fake_ctx)  # type: ignore[arg-type]
    await restate_agent.run('go')

    assert 'get dynamic tools' in fake_ctx.calls
    assert any(call.startswith('Calling dynamic tool plus_one') for call in fake_ctx.calls)


@pytest.mark.anyio
async def test_restate_agent_disable_auto_wrapping_tools_does_not_wrap_dynamic_function_tools():
    from pydantic_ai.toolsets._dynamic import DynamicToolset

    fake_ctx = FakeRestateContext()

    def toolset_func(ctx: RunContext[None]) -> FunctionToolset[None]:
        toolset = FunctionToolset[None](id='dynamic')

        @toolset.tool
        async def plus_one(x: int) -> int:
            return x + 1

        return toolset

    agent = Agent(TestModel(), toolsets=[DynamicToolset(toolset_func=toolset_func, id='dyn')])
    restate_agent = RestateAgent(agent, fake_ctx, disable_auto_wrapping_tools=True)  # type: ignore[arg-type]
    await restate_agent.run('go')

    # Discovery is still durable.
    assert 'get dynamic tools' in fake_ctx.calls
    # But execution of function tools is not automatically wrapped.
    assert not any(call.startswith('Calling dynamic tool plus_one') for call in fake_ctx.calls)

