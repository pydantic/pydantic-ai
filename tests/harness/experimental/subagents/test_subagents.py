"""Tests for the SubAgents capability."""

from __future__ import annotations

from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import AgentDepsT, RunContext

from pydantic_ai_harness.experimental.subagents import SubAgents, SubAgentToolset


@dataclass
class _RecordingCapability(AbstractCapability[AgentDepsT]):
    """Test capability whose dynamic instruction records each time it runs."""

    log: list[str] = field(default_factory=list[str])

    def get_instructions(self) -> Any:
        log = self.log

        def _instructions(ctx: RunContext[AgentDepsT]) -> str:
            log.append('applied')
            return ''

        return _instructions


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (matching upstream pydantic-ai)."""
    return 'asyncio'


def _delegate_then_finish(agent_name: str, *, retries_before: int = 0) -> FunctionModel:
    """A parent model that delegates to `agent_name` once, then replies with text.

    `retries_before` extra delegations to a bogus agent happen first (to exercise
    the unknown-agent retry path).
    """
    calls = {'n': 0}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls['n'] += 1
        if calls['n'] <= retries_before:
            return ModelResponse(
                parts=[
                    ToolCallPart('delegate_task', {'agent_name': 'ghost', 'task': 't'}, tool_call_id=f'b{calls["n"]}')
                ]
            )
        if calls['n'] == retries_before + 1:
            return ModelResponse(
                parts=[ToolCallPart('delegate_task', {'agent_name': agent_name, 'task': 'do it'}, tool_call_id='c1')]
            )
        return ModelResponse(parts=[TextPart('all done')])

    return FunctionModel(model_fn)


class TestConstruction:
    def test_serialization_name_is_none(self) -> None:
        assert SubAgents.get_serialization_name() is None

    def test_empty_agents_no_instructions(self) -> None:
        assert SubAgents[None]().get_instructions() is None

    def test_empty_agents_no_toolset(self) -> None:
        assert SubAgents[None]().get_toolset() is None


class TestInstructions:
    def test_lists_agent_with_description(self) -> None:
        agent = Agent(TestModel(), name='researcher', description='Researches topics')
        instructions = SubAgents(agents={'researcher': agent}).get_instructions()
        assert isinstance(instructions, str)
        assert '- researcher: Researches topics' in instructions
        assert 'delegate_task' in instructions

    def test_description_override_wins(self) -> None:
        agent = Agent(TestModel(), name='researcher', description='original')
        instructions = SubAgents(
            agents={'researcher': agent}, descriptions={'researcher': 'overridden'}
        ).get_instructions()
        assert isinstance(instructions, str)
        assert '- researcher: overridden' in instructions
        assert 'original' not in instructions

    def test_name_only_when_no_description(self) -> None:
        agent = Agent(TestModel(), name='plain')
        instructions = SubAgents(agents={'plain': agent}).get_instructions()
        assert isinstance(instructions, str)
        assert '- plain' in instructions
        assert '- plain:' not in instructions

    def test_custom_tool_name_in_instructions(self) -> None:
        agent = Agent(TestModel(), name='x')
        instructions = SubAgents(agents={'x': agent}, tool_name='run_agent').get_instructions()
        assert isinstance(instructions, str)
        assert 'run_agent' in instructions


class TestToolset:
    def test_get_toolset_exposes_delegate_tool(self) -> None:
        agent = Agent(TestModel(), name='x')
        toolset = SubAgents(agents={'x': agent}).get_toolset()
        assert isinstance(toolset, SubAgentToolset)
        assert 'delegate_task' in toolset.tools

    def test_custom_tool_name(self) -> None:
        agent = Agent(TestModel(), name='x')
        toolset = SubAgents(agents={'x': agent}, tool_name='run_agent').get_toolset()
        assert isinstance(toolset, SubAgentToolset)
        assert 'run_agent' in toolset.tools


class TestDelegation:
    async def test_delegates_and_returns_output(self) -> None:
        worker = Agent(TestModel(custom_output_text='WORKER RESULT'), name='worker')
        parent: Agent[None, str] = Agent(
            _delegate_then_finish('worker'), capabilities=[SubAgents(agents={'worker': worker})]
        )
        result = await parent.run('go')
        assert result.output == 'all done'
        returns = [
            part.content
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'delegate_task'
        ]
        assert returns == ['WORKER RESULT']

    async def test_unknown_agent_triggers_retry_then_succeeds(self) -> None:
        worker = Agent(TestModel(custom_output_text='OK'), name='worker')
        helper = Agent(TestModel(custom_output_text='OK'), name='helper')
        parent: Agent[None, str] = Agent(
            _delegate_then_finish('worker', retries_before=1),
            capabilities=[SubAgents(agents={'worker': worker, 'helper': helper})],
        )
        result = await parent.run('go')
        assert result.output == 'all done'
        # The bogus delegation produced a retry prompt naming the unknown agent and
        # listing the valid ones, sorted and comma-separated.
        retries = [
            part.content
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, RetryPromptPart) and part.tool_name == 'delegate_task'
        ]
        assert any(
            "Unknown sub-agent 'ghost'" in str(r) and 'Available sub-agents: helper, worker' in str(r) for r in retries
        )

    async def test_forwards_deps_and_shares_usage_by_default(self) -> None:
        captured: dict[str, Any] = {}
        parent_usage: dict[str, Any] = {}

        worker = Agent(TestModel(custom_output_text='W'), name='worker', deps_type=str)

        @worker.instructions
        def _capture(ctx: RunContext[str]) -> str:  # pyright: ignore[reportUnusedFunction]
            captured['deps'] = ctx.deps
            captured['usage_is_parent'] = ctx.usage is parent_usage.get('usage')
            return ''

        parent: Agent[str, str] = Agent(
            _delegate_then_finish('worker'),
            deps_type=str,
            capabilities=[SubAgents(agents={'worker': worker})],
        )

        @parent.instructions
        def _remember_usage(ctx: RunContext[str]) -> str:  # pyright: ignore[reportUnusedFunction]
            parent_usage['usage'] = ctx.usage
            return ''

        result = await parent.run('go', deps='PARENT')
        assert result.output == 'all done'
        assert captured['deps'] == 'PARENT'  # deps always forwarded
        assert captured['usage_is_parent'] is True  # usage shared by default

    async def test_inherit_tools_exposes_parent_tools_but_not_delegate(self) -> None:
        offered: list[str] = []

        def worker_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not offered:
                offered.extend(tool.name for tool in info.function_tools)
                # Call the inherited tool to prove it's actually usable, not just listed.
                return ModelResponse(parts=[ToolCallPart('parent_tool', {}, tool_call_id='p1')])
            return ModelResponse(parts=[TextPart('sub done')])

        worker = Agent(FunctionModel(worker_fn), name='worker')
        parent: Agent[None, str] = Agent(
            _delegate_then_finish('worker'),
            capabilities=[SubAgents(agents={'worker': worker}, inherit_tools=True)],
        )

        @parent.tool_plain
        def parent_tool() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'PT'

        result = await parent.run('go')
        assert result.output == 'all done'
        assert 'parent_tool' in offered  # the parent's tool is inherited by the sub-agent
        assert 'delegate_task' not in offered  # the delegate tool is filtered out, so no recursion

    async def test_shared_capabilities_applied_to_subagent(self) -> None:
        cap: _RecordingCapability[None] = _RecordingCapability()
        worker = Agent(TestModel(custom_output_text='W'), name='worker')
        parent: Agent[None, str] = Agent(
            _delegate_then_finish('worker'),
            capabilities=[SubAgents(agents={'worker': worker}, shared_capabilities=[cap])],
        )
        result = await parent.run('go')
        assert result.output == 'all done'
        assert cap.log == ['applied']  # the shared capability ran during the sub-agent run

    async def test_event_stream_handler_forwarded_to_subagent(self) -> None:
        events: list[str] = []

        async def handler(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                events.append(type(event).__name__)

        worker = Agent(TestModel(custom_output_text='W'), name='worker')
        parent: Agent[None, str] = Agent(
            _delegate_then_finish('worker'),
            capabilities=[SubAgents(agents={'worker': worker}, event_stream_handler=handler)],
        )
        result = await parent.run('go')
        assert result.output == 'all done'
        assert events  # the sub-agent's run streamed events to the handler

    async def test_hard_limit_propagates(self) -> None:
        def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise UsageLimitExceeded('limit hit')

        limited = Agent(FunctionModel(boom), name='limited')
        parent: Agent[None, str] = Agent(
            _delegate_then_finish('limited'),
            capabilities=[SubAgents(agents={'limited': limited})],
        )
        # Hard limits are not converted to a retry -- they propagate to stop the run.
        with pytest.raises(UsageLimitExceeded):
            await parent.run('go')

    async def test_soft_subagent_failure_becomes_model_retry(self) -> None:
        def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise UnexpectedModelBehavior('kaboom')

        boomer = Agent(FunctionModel(boom), name='boomer')
        worker = Agent(TestModel(custom_output_text='OK'), name='worker')

        calls = {'n': 0}

        def parent_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls['n'] += 1
            if calls['n'] == 1:
                return ModelResponse(
                    parts=[ToolCallPart('delegate_task', {'agent_name': 'boomer', 'task': 't'}, tool_call_id='c1')]
                )
            if calls['n'] == 2:
                return ModelResponse(
                    parts=[ToolCallPart('delegate_task', {'agent_name': 'worker', 'task': 't'}, tool_call_id='c2')]
                )
            return ModelResponse(parts=[TextPart('all done')])

        parent: Agent[None, str] = Agent(
            FunctionModel(parent_fn),
            capabilities=[SubAgents(agents={'boomer': boomer, 'worker': worker})],
        )
        result = await parent.run('go')
        assert result.output == 'all done'
        retries = [
            part.content
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, RetryPromptPart) and part.tool_name == 'delegate_task'
        ]
        assert any("Sub-agent 'boomer' failed" in str(r) for r in retries)

    async def test_usage_not_shared_when_disabled(self) -> None:
        captured: dict[str, Any] = {}
        parent_usage: dict[str, Any] = {}

        worker = Agent(TestModel(custom_output_text='W'), name='worker', deps_type=str)

        @worker.instructions
        def _capture(ctx: RunContext[str]) -> str:  # pyright: ignore[reportUnusedFunction]
            captured['deps'] = ctx.deps
            captured['usage_is_parent'] = ctx.usage is parent_usage.get('usage')
            return ''

        parent: Agent[str, str] = Agent(
            _delegate_then_finish('worker'),
            deps_type=str,
            capabilities=[SubAgents(agents={'worker': worker}, forward_usage=False)],
        )

        @parent.instructions
        def _remember_usage(ctx: RunContext[str]) -> str:  # pyright: ignore[reportUnusedFunction]
            parent_usage['usage'] = ctx.usage
            return ''

        result = await parent.run('go', deps='PARENT')
        assert result.output == 'all done'
        assert captured['deps'] == 'PARENT'  # deps still forwarded
        assert captured['usage_is_parent'] is False  # usage isolated
