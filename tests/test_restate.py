from __future__ import annotations

import inspect
from collections.abc import AsyncIterable
from contextlib import asynccontextmanager
from typing import Any, cast

import pytest

try:
    from restate import TerminalError
except ImportError:  # pragma: lax no cover
    pytest.skip('restate not installed', allow_module_level=True)

from pydantic_ai import Agent, RunContext, ToolDefinition
from pydantic_ai.durable_exec.restate import RestateAgent
from pydantic_ai.durable_exec.restate._model import RestateModelWrapper
from pydantic_ai.durable_exec.restate._serde import PydanticTypeAdapter
from pydantic_ai.durable_exec.restate._toolset import RestateContextRunToolset
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets._dynamic import DynamicToolset
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
async def test_fake_restate_context_run_typed_sync_fn():
    fake_ctx = FakeRestateContext()
    result = await fake_ctx.run_typed('sync', lambda: 'ok')
    assert result == 'ok'
    assert fake_ctx.calls == ['sync']


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

    wrapped = RestateContextRunToolset(toolset, fake_ctx)

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

    model = ErrorModel()
    assert model.model_name == 'error-model'
    assert model.system == 'test'

    fake_ctx = FakeRestateContext()
    wrapped = RestateModelWrapper(model, fake_ctx)
    with pytest.raises(TerminalError, match='bad'):
        await wrapped.request([], None, ModelRequestParameters())


@pytest.mark.anyio
async def test_restate_model_wrapper_request_stream_maps_user_error_to_terminal_error():
    fake_ctx = FakeRestateContext()

    async def boom(_: RunContext[Any], __: AsyncIterable[Any]) -> None:
        raise UserError('boom')

    wrapped = RestateModelWrapper(TestModel(call_tools=[]), fake_ctx, event_stream_handler=boom)
    mrp = ModelRequestParameters()
    ctx = _run_ctx()
    with pytest.raises(TerminalError, match='boom'):
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

    async def event_stream_handler(ctx: RunContext[Any], stream: AsyncIterable[Any]) -> None:
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
    assert any(call.startswith('handle event:') for call in fake_ctx.calls)
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

    with pytest.raises(TerminalError, match=r'agent\.run_sync\(\)'):
        restate_agent.run_sync('x')

    with pytest.raises(TerminalError, match='cannot be set at agent run time'):
        await restate_agent.run('x', model=TestModel())

    async def handler(_: RunContext[Any], __: AsyncIterable[Any]) -> None:
        return None

    await handler(_run_ctx(), cast(AsyncIterable[Any], []))

    with pytest.raises(TerminalError, match='Event stream handler cannot be set'):
        await restate_agent.run('x', event_stream_handler=handler)

    with pytest.raises(TerminalError, match=r'agent\.run_stream\(\)'):
        async with restate_agent.run_stream('x'):
            pass

    with pytest.raises(TerminalError, match=r'agent\.run_stream_events\(\)'):
        await anext(restate_agent.run_stream_events('x'))


@pytest.mark.anyio
async def test_restate_mcp_server_wrapping_and_agent_mcp_wrapping():
    from pydantic_ai.durable_exec.restate._mcp_server import RestateMCPServer

    try:
        from pydantic_ai.mcp import MCPServer
    except ImportError:  # pragma: no cover
        pytest.skip('mcp not installed')

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
                    name: ToolDefinition(
                        name=name,
                        description=name,
                        parameters_json_schema={'type': 'object', 'properties': {'value': {}}, 'required': ['value']},
                    )
                    for name in ('echo', 'none', 'retry', 'deferred', 'approval', 'user_error')
                }
                return {
                    name: ToolsetTool[Any](
                        toolset=self,
                        tool_def=tool_def,
                        max_retries=1,
                        args_validator=TOOL_SCHEMA_VALIDATOR,
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
                elif name == 'user_error':
                    raise UserError('bad')
                elif name == 'none':
                    return None
                return {'name': name, 'args': tool_args}

    fake_ctx = FakeRestateContext()
    fake_server = FakeMCPServer(tool_prefix='fake')

    with pytest.raises(RuntimeError, match='not used'):
        async with fake_server.client_streams():
            pass

    restate_server = RestateMCPServer(fake_server, fake_ctx)
    restate_server = restate_server.visit_and_replace(lambda t: t)  # cover visit_and_replace

    # Entering the Restate wrapper should be a no-op (no I/O outside ctx.run_typed).
    async with restate_server:
        pass
    assert fake_server.enter_calls == 0

    ctx = _run_ctx()
    tools = await restate_server.get_tools(ctx)
    assert fake_ctx.calls == ['get mcp tools']
    assert 'echo' in tools
    assert tools['echo'].toolset is restate_server
    assert fake_server.enter_calls == 1
    assert fake_server.exit_calls == 1

    fake_ctx.calls.clear()
    result = await restate_server.call_tool('echo', {'value': 123}, ctx, tools['echo'])
    assert fake_ctx.calls == ['Calling mcp tool echo']
    assert result == {'name': 'echo', 'args': {'value': 123}}
    assert fake_server.enter_calls == 2
    assert fake_server.exit_calls == 2

    fake_ctx.calls.clear()
    none_result = await restate_server.call_tool('none', {'value': 0}, ctx, tools['none'])
    assert fake_ctx.calls == ['Calling mcp tool none']
    assert none_result is None
    assert fake_server.enter_calls == 3
    assert fake_server.exit_calls == 3

    with pytest.raises(ModelRetry, match='nope'):
        await restate_server.call_tool('retry', {'value': 1}, ctx, tools['retry'])

    with pytest.raises(CallDeferred) as exc_info:
        await restate_server.call_tool('deferred', {'value': 1}, ctx, tools['deferred'])
    assert exc_info.value.metadata == {'foo': 'bar'}

    with pytest.raises(ApprovalRequired) as exc_info:
        await restate_server.call_tool('approval', {'value': 1}, ctx, tools['approval'])
    assert exc_info.value.metadata == {'hello': 'world'}

    with pytest.raises(TerminalError, match='bad'):
        await restate_server.call_tool('user_error', {'value': 1}, ctx, tools['user_error'])

    # Ensure RestateAgent hits the MCP wrapping branch when an MCPServer toolset is present.
    fake_ctx.calls.clear()
    agent = Agent(TestModel(call_tools=[]), toolsets=[fake_server])
    RestateAgent(agent, fake_ctx)
    assert fake_ctx.calls == []


@pytest.mark.anyio
async def test_restate_dynamic_toolset_is_durable_and_revalidates_args():
    from pydantic_ai.durable_exec.restate._dynamic_toolset import RestateDynamicToolset

    fake_ctx = FakeRestateContext()

    def toolset_func(ctx: RunContext[None]) -> FunctionToolset[None]:
        toolset = FunctionToolset[None](id='dynamic')

        @toolset.tool
        async def add_one(x: int) -> int:
            return x + 1

        return toolset

    dynamic = DynamicToolset[None](toolset_func=toolset_func, id='dyn')
    durable_dynamic = RestateDynamicToolset(dynamic, fake_ctx, disable_auto_wrapping_tools=False)

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
async def test_restate_dynamic_toolset_maps_control_flow_exceptions_and_user_error():
    from pydantic_ai.durable_exec.restate._dynamic_toolset import RestateDynamicToolset

    fake_ctx = FakeRestateContext()

    def toolset_func(ctx: RunContext[None]) -> FunctionToolset[None]:
        toolset = FunctionToolset[None](id='dynamic')

        @toolset.tool
        async def retry() -> None:
            raise ModelRetry('nope')

        @toolset.tool
        async def deferred() -> None:
            raise CallDeferred(metadata={'foo': 'bar'})

        @toolset.tool
        async def approval() -> None:
            raise ApprovalRequired(metadata={'hello': 'world'})

        @toolset.tool
        async def user_error() -> None:
            raise UserError('bad')

        return toolset

    dynamic = DynamicToolset[None](toolset_func=toolset_func, id='dyn')
    durable_dynamic = RestateDynamicToolset(dynamic, fake_ctx, disable_auto_wrapping_tools=False)

    ctx = _run_ctx()
    tools = await durable_dynamic.get_tools(ctx)

    with pytest.raises(ModelRetry, match='nope'):
        await durable_dynamic.call_tool('retry', {}, ctx, tools['retry'])

    with pytest.raises(CallDeferred) as exc_info:
        await durable_dynamic.call_tool('deferred', {}, ctx, tools['deferred'])
    assert exc_info.value.metadata == {'foo': 'bar'}

    with pytest.raises(ApprovalRequired) as exc_info:
        await durable_dynamic.call_tool('approval', {}, ctx, tools['approval'])
    assert exc_info.value.metadata == {'hello': 'world'}

    with pytest.raises(TerminalError, match='bad'):
        await durable_dynamic.call_tool('user_error', {}, ctx, tools['user_error'])

    fake_ctx.calls.clear()
    with pytest.raises(TerminalError, match="Tool 'missing' not found"):
        await durable_dynamic.call_tool('missing', {}, ctx, tools['retry'])
    assert fake_ctx.calls == ['Calling dynamic tool missing']


@pytest.mark.anyio
async def test_restate_dynamic_toolset_disable_auto_wrapping_tools_validation_error_raises_model_retry():
    from pydantic_ai.durable_exec.restate._dynamic_toolset import RestateDynamicToolset

    fake_ctx = FakeRestateContext()

    def toolset_func(ctx: RunContext[None]) -> FunctionToolset[None]:
        toolset = FunctionToolset[None](id='dynamic')

        @toolset.tool
        async def add_one(x: int) -> int:
            return x + 1

        return toolset

    dynamic = DynamicToolset[None](toolset_func=toolset_func, id='dyn')
    durable_dynamic = RestateDynamicToolset(dynamic, fake_ctx, disable_auto_wrapping_tools=True)

    ctx = _run_ctx()
    tools = await durable_dynamic.get_tools(ctx)

    fake_ctx.calls.clear()
    with pytest.raises(ModelRetry):
        await durable_dynamic.call_tool('add_one', {'x': 'not-an-int'}, ctx, tools['add_one'])
    assert await durable_dynamic.call_tool('add_one', {'x': 1}, ctx, tools['add_one']) == 2
    assert fake_ctx.calls == []


@pytest.mark.anyio
async def test_restate_dynamic_toolset_function_origin_unwraps_wrapper_toolsets():
    from pydantic_ai.durable_exec.restate._dynamic_toolset import RestateDynamicToolset
    from pydantic_ai.toolsets import PrefixedToolset
    from pydantic_ai.toolsets.abstract import AbstractToolset

    fake_ctx = FakeRestateContext()

    def toolset_func(ctx: RunContext[None]) -> AbstractToolset[None]:
        inner = FunctionToolset[None](id='inner')

        @inner.tool
        async def plus_one(x: int) -> int:
            return x + 1

        return PrefixedToolset(inner, prefix='p')

    dynamic = DynamicToolset[None](toolset_func=toolset_func, id='dyn')
    durable_dynamic = RestateDynamicToolset(dynamic, fake_ctx, disable_auto_wrapping_tools=True)

    ctx = _run_ctx()
    tools = await durable_dynamic.get_tools(ctx)

    assert tools['p_plus_one'].tool_def.metadata is not None
    assert tools['p_plus_one'].tool_def.metadata.get('__pydantic_ai_restate_dynamic_origin') == 'function'
    assert await durable_dynamic.call_tool('p_plus_one', {'x': 1}, ctx, tools['p_plus_one']) == 2


@pytest.mark.anyio
async def test_restate_fastmcp_toolset_wrapping_smoke():
    pytest.importorskip('pydantic_ai.toolsets.fastmcp')
    pytest.importorskip('fastmcp')

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
    durable_toolset = RestateFastMCPToolset(toolset, fake_ctx)

    # Entering the Restate wrapper should be a no-op (no I/O outside ctx.run_typed).
    async with durable_toolset:
        pass
    assert toolset.enter_calls == 0

    ctx = _run_ctx()
    tools = await durable_toolset.get_tools(ctx)
    assert 'get fastmcp tools' in fake_ctx.calls
    assert 'echo' in tools
    assert tools['echo'].toolset is durable_toolset
    assert toolset.enter_calls == 1
    assert toolset.exit_calls == 1

    fake_ctx.calls.clear()
    result = await durable_toolset.call_tool('echo', {'value': 123}, ctx, tools['echo'])
    assert fake_ctx.calls == ['Calling fastmcp tool echo']
    assert isinstance(result, dict)
    assert cast(dict[str, Any], result).get('value') == 123
    assert toolset.enter_calls == 2
    assert toolset.exit_calls == 2


@pytest.mark.anyio
async def test_restate_fastmcp_toolset_maps_control_flow_exceptions_and_user_error():
    pytest.importorskip('pydantic_ai.toolsets.fastmcp')
    pytest.importorskip('fastmcp')

    from fastmcp import FastMCP

    from pydantic_ai.durable_exec.restate._fastmcp_toolset import RestateFastMCPToolset
    from pydantic_ai.toolsets.fastmcp import FastMCPToolset

    mcp = FastMCP('test')

    def echo(value: int) -> dict[str, Any]:
        return {'value': value}

    # Placeholder tool implementations; behavior is overridden by `RaisingFastMCPToolset`.
    def retry() -> None:  # pragma: no cover
        raise RuntimeError('not used')

    def deferred() -> None:  # pragma: no cover
        raise RuntimeError('not used')

    def approval() -> None:  # pragma: no cover
        raise RuntimeError('not used')

    def user_error() -> None:  # pragma: no cover
        raise RuntimeError('not used')

    mcp.tool(echo)
    mcp.tool(retry)
    mcp.tool(deferred)
    mcp.tool(approval)
    mcp.tool(user_error)

    class RaisingFastMCPToolset(FastMCPToolset[Any]):
        async def call_tool(
            self,
            name: str,
            tool_args: dict[str, Any],
            ctx: RunContext[Any],
            tool: ToolsetTool[Any],
        ) -> Any:
            if name == 'retry':
                raise ModelRetry('nope')
            elif name == 'deferred':
                raise CallDeferred(metadata={'foo': 'bar'})
            elif name == 'approval':
                raise ApprovalRequired(metadata={'hello': 'world'})
            elif name == 'user_error':
                raise UserError('bad')
            return await super().call_tool(name, tool_args, ctx, tool)

    toolset = RaisingFastMCPToolset(mcp, id='fastmcp')
    fake_ctx = FakeRestateContext()
    durable_toolset = RestateFastMCPToolset(toolset, fake_ctx)
    assert durable_toolset.visit_and_replace(lambda t: t) is durable_toolset

    ctx = _run_ctx()
    tools = await durable_toolset.get_tools(ctx)

    result = await durable_toolset.call_tool('echo', {'value': 123}, ctx, tools['echo'])
    assert isinstance(result, dict)
    assert cast(dict[str, Any], result).get('value') == 123

    with pytest.raises(ModelRetry, match='nope'):
        await durable_toolset.call_tool('retry', {}, ctx, tools['retry'])

    with pytest.raises(CallDeferred) as exc_info:
        await durable_toolset.call_tool('deferred', {}, ctx, tools['deferred'])
    assert exc_info.value.metadata == {'foo': 'bar'}

    with pytest.raises(ApprovalRequired) as exc_info:
        await durable_toolset.call_tool('approval', {}, ctx, tools['approval'])
    assert exc_info.value.metadata == {'hello': 'world'}

    with pytest.raises(TerminalError, match='bad'):
        await durable_toolset.call_tool('user_error', {}, ctx, tools['user_error'])


@pytest.mark.anyio
async def test_restate_agent_wraps_fastmcp_toolset_and_disables_event_handler_wrapping():
    pytest.importorskip('pydantic_ai.toolsets.fastmcp')
    pytest.importorskip('fastmcp')

    from fastmcp import FastMCP

    from pydantic_ai.durable_exec.restate._fastmcp_toolset import RestateFastMCPToolset
    from pydantic_ai.messages import AgentStreamEvent
    from pydantic_ai.toolsets.fastmcp import FastMCPToolset

    mcp = FastMCP('test')

    def echo(value: int) -> dict[str, Any]:  # pragma: no cover
        return {'value': value}

    mcp.tool(echo)

    async def handler(_: RunContext[Any], __: AsyncIterable[AgentStreamEvent]) -> None:  # pragma: no cover
        return None

    toolset = FastMCPToolset(mcp, id='fastmcp')
    agent = Agent(TestModel(call_tools=[]), toolsets=[toolset])
    fake_ctx = FakeRestateContext()

    restate_agent = RestateAgent(
        agent,
        fake_ctx,
        event_stream_handler=handler,
        disable_auto_wrapping_tools=True,
    )

    assert isinstance(restate_agent.model, RestateModelWrapper)
    assert restate_agent.event_stream_handler is handler
    assert any(isinstance(ts, RestateFastMCPToolset) for ts in restate_agent.toolsets)


@pytest.mark.anyio
async def test_restate_agent_wraps_dynamic_toolset():
    fake_ctx = FakeRestateContext()

    def toolset_func(ctx: RunContext[None]) -> FunctionToolset[None]:
        toolset = FunctionToolset[None](id='dynamic')

        @toolset.tool
        async def plus_one(x: int) -> int:
            return x + 1

        return toolset

    agent = Agent(TestModel(), toolsets=[DynamicToolset(toolset_func=toolset_func, id='dyn')])
    restate_agent = RestateAgent(agent, fake_ctx)
    await restate_agent.run('go')

    assert 'get dynamic tools' in fake_ctx.calls
    assert any(call.startswith('Calling dynamic tool plus_one') for call in fake_ctx.calls)


@pytest.mark.anyio
async def test_restate_agent_disable_auto_wrapping_tools_does_not_wrap_dynamic_function_tools():
    fake_ctx = FakeRestateContext()

    def toolset_func(ctx: RunContext[None]) -> FunctionToolset[None]:
        toolset = FunctionToolset[None](id='dynamic')

        @toolset.tool
        async def plus_one(x: int) -> int:
            return x + 1

        return toolset

    agent = Agent(TestModel(), toolsets=[DynamicToolset(toolset_func=toolset_func, id='dyn')])
    restate_agent = RestateAgent(agent, fake_ctx, disable_auto_wrapping_tools=True)
    await restate_agent.run('go')

    # Discovery is still durable.
    assert 'get dynamic tools' in fake_ctx.calls
    # But execution of function tools is not automatically wrapped.
    assert not any(call.startswith('Calling dynamic tool plus_one') for call in fake_ctx.calls)


@pytest.mark.anyio
async def test_restate_agent_misc_properties_and_wrapped_event_handler_noop_and_iter_model_guard():
    from pydantic_ai.messages import AgentStreamEvent

    class EmptyAgentStream:
        def __aiter__(self) -> EmptyAgentStream:
            return self

        async def __anext__(self) -> AgentStreamEvent:
            raise StopAsyncIteration

    fake_ctx = FakeRestateContext()
    agent = Agent(TestModel(call_tools=[]))
    restate_agent = RestateAgent(agent, fake_ctx)

    assert isinstance(restate_agent.model, RestateModelWrapper)
    _ = restate_agent.toolsets

    stream = EmptyAgentStream()
    assert stream.__aiter__() is stream
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()

    assert restate_agent.event_stream_handler is None
    await cast(Any, restate_agent)._wrapped_event_stream_handler(_run_ctx(), stream)

    with pytest.raises(TerminalError, match='cannot be set at agent run time'):
        async with restate_agent.iter('go', model=TestModel(call_tools=[])):
            pass
