"""MCP durable-wrapping tests for `AbsurdDurability`.

The MCP extra (`mcp`/`fastmcp`) may be absent, so the module `importorskip`s it. The shared
`FakeMCPToolset` stands in for a real server.
"""

from __future__ import annotations

import pytest

pytest.importorskip('absurd_sdk')
pytest.importorskip('pydantic_ai.mcp')


from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.absurd import AbsurdDurability

from .._fake_mcp import FakeMCPToolset
from ._helpers import FakeAsyncTaskContext, absurd_task_context


def _add_then_done_model() -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        answered = any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)
        if not answered:
            return ModelResponse(parts=[ToolCallPart(tool_name='add', args={'a': 2, 'b': 3})])
        return ModelResponse(parts=[TextPart(content='summed')])

    return FunctionModel(fn, model_name='fn')


class TestMcpCheckpointing:
    async def test_get_tools_get_instructions_and_call_tool_checkpointed(self) -> None:
        server = FakeMCPToolset(id='calc', instructions='Use the calculator.', include_instructions=True)
        agent = Agent(_add_then_done_model(), name='calc', toolsets=[server], capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            result = await agent.run('add 2 and 3')

        assert result.output == 'summed'
        assert server.tool_calls == [('add', {'a': 2, 'b': 3})]
        assert 'calc__mcp_server__calc.get_tools' in ctx.stored
        assert 'calc__mcp_server__calc.get_instructions' in ctx.stored
        assert 'calc__mcp_server__calc.call_tool' in ctx.stored

    async def test_replay_does_not_rehit_server(self) -> None:
        server = FakeMCPToolset(id='calc', instructions='Use the calculator.', include_instructions=True)
        agent = Agent(_add_then_done_model(), name='calc', toolsets=[server], capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            first = await agent.run('add 2 and 3')

        replay = ctx.replay()
        with absurd_task_context(replay):
            second = await agent.run('add 2 and 3')

        assert first.output == second.output == 'summed'
        assert server.tool_calls == [('add', {'a': 2, 'b': 3})]
        assert replay.invoked == []


class TestIdLessMcpToolset:
    async def test_id_less_server_uses_pydantic_ai_absurd_step_names(self) -> None:
        # An MCP toolset constructed without an `id` (for example an in-process server) keeps the
        # `pydantic-ai-absurd` step names, which have no `__<id>` segment.
        server = FakeMCPToolset(id=None, instructions='Use the calculator.', include_instructions=True)
        agent = Agent(_add_then_done_model(), name='calc', toolsets=[server], capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            await agent.run('add 2 and 3')

        assert [name for name in ctx.stored if '__mcp_server' in name] == snapshot(
            [
                'calc__mcp_server.get_tools',
                'calc__mcp_server.get_instructions',
                'calc__mcp_server.call_tool',
                'calc__mcp_server.get_instructions#2',
            ]
        )
        assert ctx.stored['calc__mcp_server.call_tool'] == 5

        replay = ctx.replay()
        with absurd_task_context(replay):
            await agent.run('add 2 and 3')
        assert server.tool_calls == [('add', {'a': 2, 'b': 3})]
        assert replay.invoked == []


class TestMcpSessionLifecycle:
    async def test_replay_opens_no_mcp_session(self) -> None:
        # As in `pydantic-ai-absurd`, the wrapper does not enter the server itself, so a replay that
        # is served entirely from checkpoints never connects to it.
        server = FakeMCPToolset(id='calc', instructions='Use the calculator.', include_instructions=True)
        agent = Agent(_add_then_done_model(), name='calc', toolsets=[server], capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            await agent.run('add 2 and 3')
        sessions_after_first_run = server.enter_count

        with absurd_task_context(ctx.replay()):
            result = await agent.run('add 2 and 3')

        assert result.output == 'summed'
        assert server.enter_count == sessions_after_first_run
