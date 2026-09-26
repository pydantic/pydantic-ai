"""Tests for the SubAgents capability."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior, UsageLimitExceeded, UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness.subagents import SubAgent, SubAgents, SubAgentToolset


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


def _delegate_n_then_finish(agent_name: str, n: int) -> FunctionModel:
    """A parent model that delegates to `agent_name` `n` times, then replies with text."""
    calls = {'n': 0}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls['n'] += 1
        if calls['n'] <= n:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        'delegate_task', {'agent_name': agent_name, 'task': 't'}, tool_call_id=f'c{calls["n"]}'
                    )
                ]
            )
        return ModelResponse(parts=[TextPart('all done')])

    return FunctionModel(model_fn)


def _delegate_two_then_finish(first: str, second: str) -> FunctionModel:
    """A parent model that delegates to `first`, then `second`, then replies with text."""
    calls = {'n': 0}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls['n'] += 1
        if calls['n'] == 1:
            return ModelResponse(
                parts=[ToolCallPart('delegate_task', {'agent_name': first, 'task': 't'}, tool_call_id='c1')]
            )
        if calls['n'] == 2:
            return ModelResponse(
                parts=[ToolCallPart('delegate_task', {'agent_name': second, 'task': 't'}, tool_call_id='c2')]
            )
        return ModelResponse(parts=[TextPart('all done')])

    return FunctionModel(model_fn)


def _delegate_returns(result: Any) -> list[str]:
    """The `delegate_task` tool-return contents from a run result, in order."""
    return [
        str(part.content)
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'delegate_task'
    ]


class TestConstruction:
    def test_serialization_name_is_none(self) -> None:
        assert SubAgents.get_serialization_name() is None

    def test_empty_agents_no_instructions(self) -> None:
        assert SubAgents[object]().get_instructions() is None

    def test_empty_agents_no_toolset(self) -> None:
        assert SubAgents[object]().get_toolset() is None


class TestInstructions:
    def test_lists_agent_with_description(self) -> None:
        agent = Agent(TestModel(), name='researcher', description='Researches topics')
        instructions = SubAgents(agents=[SubAgent(agent)]).get_instructions()
        assert isinstance(instructions, str)
        assert '- researcher: Researches topics' in instructions
        assert 'delegate_task' in instructions

    def test_description_override_wins(self) -> None:
        agent = Agent(TestModel(), name='researcher', description='original')
        instructions = SubAgents(agents=[SubAgent(agent, description='overridden')]).get_instructions()
        assert isinstance(instructions, str)
        assert '- researcher: overridden' in instructions
        assert 'original' not in instructions

    def test_name_only_when_no_description(self) -> None:
        agent = Agent(TestModel(), name='plain')
        instructions = SubAgents(agents=[SubAgent(agent)]).get_instructions()
        assert isinstance(instructions, str)
        assert '- plain' in instructions
        assert '- plain:' not in instructions

    def test_name_override_wins(self) -> None:
        agent = Agent(TestModel(), name='internal')
        instructions = SubAgents(agents=[SubAgent(agent, name='public')]).get_instructions()
        assert isinstance(instructions, str)
        assert '- public' in instructions
        assert 'internal' not in instructions

    def test_custom_tool_name_in_instructions(self) -> None:
        agent = Agent(TestModel(), name='x')
        instructions = SubAgents(agents=[SubAgent(agent)], tool_name='run_agent').get_instructions()
        assert isinstance(instructions, str)
        assert 'run_agent' in instructions


class TestToolset:
    def test_get_toolset_exposes_delegate_tool(self) -> None:
        agent = Agent(TestModel(), name='x')
        toolset = SubAgents(agents=[SubAgent(agent)]).get_toolset()
        assert isinstance(toolset, SubAgentToolset)
        assert 'delegate_task' in toolset.tools

    def test_custom_tool_name(self) -> None:
        agent = Agent(TestModel(), name='x')
        toolset = SubAgents(agents=[SubAgent(agent)], tool_name='run_agent').get_toolset()
        assert isinstance(toolset, SubAgentToolset)
        assert 'run_agent' in toolset.tools

    def test_tool_retries_default_is_resilient(self) -> None:
        agent = Agent(TestModel(), name='x')
        toolset = SubAgents(agents=[SubAgent(agent)]).get_toolset()
        assert isinstance(toolset, SubAgentToolset)
        assert toolset.tools['delegate_task'].max_retries == 2

    def test_tool_retries_none_inherits_agent_default(self) -> None:
        agent = Agent(TestModel(), name='x')
        toolset = SubAgents(agents=[SubAgent(agent)], tool_retries=None).get_toolset()
        assert isinstance(toolset, SubAgentToolset)
        assert toolset.tools['delegate_task'].max_retries is None

    def test_tool_retries_configures_delegate_tool(self) -> None:
        agent = Agent(TestModel(), name='x')
        toolset = SubAgents(agents=[SubAgent(agent)], tool_retries=3).get_toolset()
        assert isinstance(toolset, SubAgentToolset)
        assert toolset.tools['delegate_task'].max_retries == 3


class TestDelegation:
    async def test_delegates_and_returns_output(self) -> None:
        worker = Agent(TestModel(custom_output_text='WORKER RESULT'), name='worker')
        parent: Agent[object, str] = Agent(
            _delegate_then_finish('worker'), capabilities=[SubAgents(agents=[SubAgent(worker)])]
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

    async def test_delegates_via_name_override(self) -> None:
        worker = Agent(TestModel(custom_output_text='WORKER RESULT'), name='internal')
        parent: Agent[object, str] = Agent(
            _delegate_then_finish('public'),
            capabilities=[SubAgents(agents=[SubAgent(worker, name='public')])],
        )
        result = await parent.run('go')
        assert result.output == 'all done'
        assert _delegate_returns(result) == ['WORKER RESULT']

    async def test_unknown_agent_triggers_retry_then_succeeds(self) -> None:
        worker = Agent(TestModel(custom_output_text='OK'), name='worker')
        helper = Agent(TestModel(custom_output_text='OK'), name='helper')
        parent: Agent[object, str] = Agent(
            _delegate_then_finish('worker', retries_before=1),
            capabilities=[SubAgents(agents=[SubAgent(worker), SubAgent(helper)])],
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
            capabilities=[SubAgents(agents=[SubAgent(worker)])],
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
        parent: Agent[object, str] = Agent(
            _delegate_then_finish('worker'),
            capabilities=[SubAgents(agents=[SubAgent(worker)], inherit_tools=True)],
        )

        @parent.tool_plain
        def parent_tool() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'PT'

        result = await parent.run('go')
        assert result.output == 'all done'
        assert 'parent_tool' in offered  # the parent's tool is inherited by the sub-agent
        assert 'delegate_task' not in offered  # the delegate tool is filtered out, so no recursion

    async def test_directly_registered_toolset_still_filters_delegate_tool(self) -> None:
        """`SubAgentToolset` used without the `SubAgents` capability must not recurse.

        Registered directly in `Agent(toolsets=[...])` it is not wrapped in
        `CapabilityOwnedToolset`, so only the name filter keeps `delegate_task`
        out of inherited toolsets.
        """
        offered: list[str] = []

        def worker_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            offered.extend(tool.name for tool in info.function_tools)
            return ModelResponse(parts=[TextPart('sub done')])

        worker = Agent(FunctionModel(worker_fn), name='worker')
        toolset: SubAgentToolset[object] = SubAgentToolset(
            agents={'worker': SubAgent(worker)},
            forward_usage=True,
            inherit_tools=True,
            shared_capabilities=[],
            event_stream_handler=None,
            tool_name='delegate_task',
            tool_retries=1,
            contain_errors=False,
            call_counts={},
        )
        parent: Agent[object, str] = Agent(_delegate_then_finish('worker'), toolsets=[toolset])

        @parent.tool_plain
        def parent_tool() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'PT'  # pragma: no cover - listed but not called in this test

        result = await parent.run('go')
        assert result.output == 'all done'
        assert 'parent_tool' in offered  # the parent's own tool is still inherited
        assert 'delegate_task' not in offered  # the delegate tool is filtered by name

    async def test_inherit_tools_excludes_capability_contributed_tools(self) -> None:
        """Tools contributed by the parent's capabilities stay out of sub-agent runs.

        They are bound to capability instances registered in the parent run; sharing
        them is `shared_capabilities`' job (see the `_inherited_toolsets` docstring).
        """

        @dataclass
        class _ToolCapability(AbstractCapability[object]):
            def get_toolset(self) -> Any:
                def cap_tool() -> str:
                    return 'CT'  # pragma: no cover - never offered to the sub-agent

                return FunctionToolset[object](tools=[cap_tool])

        offered: list[str] = []

        def worker_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            offered.extend(tool.name for tool in info.function_tools)
            return ModelResponse(parts=[TextPart('sub done')])

        worker = Agent(FunctionModel(worker_fn), name='worker')
        parent: Agent[object, str] = Agent(
            _delegate_then_finish('worker'),
            capabilities=[SubAgents(agents=[SubAgent(worker)], inherit_tools=True), _ToolCapability()],
        )

        @parent.tool_plain
        def parent_tool() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'PT'  # pragma: no cover - listed but not called in this test

        result = await parent.run('go')
        assert result.output == 'all done'
        assert 'parent_tool' in offered
        assert 'cap_tool' not in offered

    async def test_shared_capabilities_applied_to_subagent(self) -> None:
        cap: _RecordingCapability[object] = _RecordingCapability()
        worker = Agent(TestModel(custom_output_text='W'), name='worker')
        parent: Agent[object, str] = Agent(
            _delegate_then_finish('worker'),
            capabilities=[SubAgents(agents=[SubAgent(worker)], shared_capabilities=[cap])],
        )
        result = await parent.run('go')
        assert result.output == 'all done'
        assert cap.log == ['applied']  # the shared capability ran during the sub-agent run

    async def test_event_stream_handler_forwarded_to_subagent(self) -> None:
        events: list[str] = []

        async def handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                events.append(type(event).__name__)

        worker = Agent(TestModel(custom_output_text='W'), name='worker')
        parent: Agent[object, str] = Agent(
            _delegate_then_finish('worker'),
            capabilities=[SubAgents(agents=[SubAgent(worker)], event_stream_handler=handler)],
        )
        result = await parent.run('go')
        assert result.output == 'all done'
        assert events  # the sub-agent's run streamed events to the handler

    async def test_hard_limit_propagates(self) -> None:
        def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise UsageLimitExceeded('limit hit')

        limited = Agent(FunctionModel(boom), name='limited')
        parent: Agent[object, str] = Agent(
            _delegate_then_finish('limited'),
            capabilities=[SubAgents(agents=[SubAgent(limited)])],
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

        parent: Agent[object, str] = Agent(
            FunctionModel(parent_fn),
            capabilities=[SubAgents(agents=[SubAgent(boomer), SubAgent(worker)])],
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
            capabilities=[SubAgents(agents=[SubAgent(worker)], forward_usage=False)],
        )

        @parent.instructions
        def _remember_usage(ctx: RunContext[str]) -> str:  # pyright: ignore[reportUnusedFunction]
            parent_usage['usage'] = ctx.usage
            return ''

        result = await parent.run('go', deps='PARENT')
        assert result.output == 'all done'
        assert captured['deps'] == 'PARENT'  # deps still forwarded
        assert captured['usage_is_parent'] is False  # usage isolated


class TestRunControls:
    async def test_usage_limits_isolate_child_accounting(self) -> None:
        captured: dict[str, Any] = {}
        parent_usage: dict[str, Any] = {}

        worker = Agent(TestModel(custom_output_text='W'), name='worker')

        @worker.instructions
        def _capture(ctx: RunContext[object]) -> str:  # pyright: ignore[reportUnusedFunction]
            captured['usage_is_parent'] = ctx.usage is parent_usage.get('usage')
            return ''

        parent: Agent[object, str] = Agent(
            _delegate_then_finish('worker'),
            capabilities=[SubAgents(agents=[SubAgent(worker, usage_limits=UsageLimits(request_limit=5))])],
        )

        @parent.instructions
        def _remember_usage(ctx: RunContext[object]) -> str:  # pyright: ignore[reportUnusedFunction]
            parent_usage['usage'] = ctx.usage
            return ''

        result = await parent.run('go')
        assert result.output == 'all done'
        # A per-child usage_limits forces isolated accounting even though forward_usage defaults to True.
        assert captured['usage_is_parent'] is False

    async def test_usage_budget_reached_is_soft(self) -> None:
        counter = {'n': 0}

        def worker_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            counter['n'] += 1
            if counter['n'] == 1:
                return ModelResponse(parts=[ToolCallPart('noop', {}, tool_call_id='n1')])
            return ModelResponse(parts=[TextPart('done')])  # pragma: no cover - blocked by the request budget

        worker = Agent(FunctionModel(worker_fn), name='worker')

        @worker.tool_plain
        def noop() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'x'

        parent: Agent[object, str] = Agent(
            _delegate_then_finish('worker'),
            capabilities=[SubAgents(agents=[SubAgent(worker, usage_limits=UsageLimits(request_limit=1))])],
        )
        result = await parent.run('go')
        assert result.output == 'all done'
        returns = _delegate_returns(result)
        # The child's own budget being hit is recoverable, not a run-stopping UsageLimitExceeded.
        assert len(returns) == 1
        assert "Sub-agent 'worker' reached its usage budget" in returns[0]

    async def test_shared_usage_limit_still_propagates(self) -> None:
        # No per-child limit -> the child shares accounting and a parent-level usage
        # limit remains a hard stop for the whole tree.
        worker = Agent(TestModel(custom_output_text='W'), name='worker')
        parent: Agent[object, str] = Agent(
            _delegate_then_finish('worker'),
            capabilities=[SubAgents(agents=[SubAgent(worker)])],
        )
        with pytest.raises(UsageLimitExceeded):
            await parent.run('go', usage_limits=UsageLimits(request_limit=1))

    async def test_timeout_returns_soft_message(self) -> None:
        async def slow_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            await asyncio.sleep(1)
            return ModelResponse(parts=[TextPart('late')])  # pragma: no cover - cancelled by the timeout

        worker = Agent(FunctionModel(slow_fn), name='worker')
        parent: Agent[object, str] = Agent(
            _delegate_then_finish('worker'),
            capabilities=[SubAgents(agents=[SubAgent(worker, timeout_seconds=0.01)])],
        )
        result = await parent.run('go')
        assert result.output == 'all done'
        returns = _delegate_returns(result)
        assert len(returns) == 1
        assert "Sub-agent 'worker' exceeded its 0.01s time budget" in returns[0]

    async def test_max_calls_exhausted_returns_soft_and_skips_child(self) -> None:
        runs = {'n': 0}

        def worker_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            runs['n'] += 1
            return ModelResponse(parts=[TextPart('W')])

        worker = Agent(FunctionModel(worker_fn), name='worker')
        parent: Agent[object, str] = Agent(
            _delegate_n_then_finish('worker', 2),
            capabilities=[SubAgents(agents=[SubAgent(worker, max_calls=1)])],
        )
        result = await parent.run('go')
        assert result.output == 'all done'
        returns = _delegate_returns(result)
        assert len(returns) == 2
        assert returns[0] == 'W'
        assert "Delegate budget for 'worker' is exhausted" in returns[1]
        assert runs['n'] == 1  # the over-budget delegation never ran the child

    async def test_call_counts_reset_between_runs(self) -> None:
        worker = Agent(TestModel(custom_output_text='W'), name='worker')
        capability = SubAgents(agents=[SubAgent(worker, max_calls=1)])
        # Two parents share the one capability (and its run-scoped count store); each
        # gets a fresh delegate-once model so the second run actually delegates again.
        first = await Agent(_delegate_then_finish('worker'), capabilities=[capability]).run('go')
        second = await Agent(_delegate_then_finish('worker'), capabilities=[capability]).run('go')
        # A fresh run starts the budget over: each delegation succeeds, none is exhausted.
        assert _delegate_returns(first) == ['W']
        assert _delegate_returns(second) == ['W']
        # wrap_run clears each run's counts, so the store does not accumulate.
        assert capability._call_counts == {}

    async def test_on_failure_makes_child_failure_soft(self) -> None:
        def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise UnexpectedModelBehavior('kaboom')

        boomer = Agent(FunctionModel(boom), name='boomer')
        parent: Agent[object, str] = Agent(
            _delegate_then_finish('boomer'),
            capabilities=[SubAgents(agents=[SubAgent(boomer, on_failure='steer: use existing evidence')])],
        )
        result = await parent.run('go')
        assert result.output == 'all done'
        # on_failure returns a normal tool result, so there is no RetryPrompt.
        retries = [
            part
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, RetryPromptPart) and part.tool_name == 'delegate_task'
        ]
        assert retries == []
        assert _delegate_returns(result) == ['steer: use existing evidence']

    async def test_on_failure_overrides_default_steering(self) -> None:
        worker = Agent(TestModel(custom_output_text='W'), name='worker')
        parent: Agent[object, str] = Agent(
            _delegate_n_then_finish('worker', 2),
            capabilities=[SubAgents(agents=[SubAgent(worker, max_calls=1, on_failure='custom budget note')])],
        )
        result = await parent.run('go')
        assert result.output == 'all done'
        returns = _delegate_returns(result)
        assert returns[1] == 'custom budget note'

    async def test_limits_without_budget_run_normally(self) -> None:
        # A SubAgent with only an unrelated control set must not alter the happy path.
        worker = Agent(TestModel(custom_output_text='W'), name='worker')
        parent: Agent[object, str] = Agent(
            _delegate_then_finish('worker'),
            capabilities=[SubAgents(agents=[SubAgent(worker, timeout_seconds=30)])],
        )
        result = await parent.run('go')
        assert _delegate_returns(result) == ['W']

    async def test_max_calls_counts_parallel_delegations_in_one_step(self) -> None:
        runs = {'n': 0}

        def worker_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            runs['n'] += 1
            return ModelResponse(parts=[TextPart('W')])

        worker = Agent(FunctionModel(worker_fn), name='worker')

        # Two delegations issued in a single parent model step run concurrently; the
        # synchronous increment in _budget_exhausted must still cap them at max_calls=1.
        step = {'n': 0}

        def parent_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            step['n'] += 1
            if step['n'] == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart('delegate_task', {'agent_name': 'worker', 'task': 't'}, tool_call_id='a'),
                        ToolCallPart('delegate_task', {'agent_name': 'worker', 'task': 't'}, tool_call_id='b'),
                    ]
                )
            return ModelResponse(parts=[TextPart('all done')])

        parent: Agent[object, str] = Agent(
            FunctionModel(parent_fn),
            capabilities=[SubAgents(agents=[SubAgent(worker, max_calls=1)])],
        )
        result = await parent.run('go')
        assert result.output == 'all done'
        returns = _delegate_returns(result)
        # Exactly one delegation ran the child; the other was over budget.
        assert runs['n'] == 1
        assert len(returns) == 2
        assert 'W' in returns
        assert any("Delegate budget for 'worker' is exhausted" in r for r in returns)


def _crash(message: str = 'provider down') -> FunctionModel:
    """A sub-agent model that raises a `ModelAPIError` (an unexpected crash, not a soft failure)."""

    def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelAPIError('m', message)

    return FunctionModel(boom)


def _delegate_retries(result: Any) -> list[str]:
    """The `delegate_task` retry-prompt contents from a run result, in order."""
    return [
        str(part.content)
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, RetryPromptPart) and part.tool_name == 'delegate_task'
    ]


class TestContainErrors:
    """`contain_errors`: an unexpected sub-agent crash becomes a bounded retry instead of aborting the parent."""

    async def test_crash_propagates_by_default(self) -> None:
        boomer = Agent(_crash(), name='boomer')
        parent: Agent[object, str] = Agent(
            _delegate_then_finish('boomer'),
            capabilities=[SubAgents(agents=[SubAgent(boomer)])],
        )
        with pytest.raises(ModelAPIError):
            await parent.run('go')

    async def test_contained_crash_becomes_model_retry(self) -> None:
        boomer = Agent(_crash(), name='boomer')
        worker = Agent(TestModel(custom_output_text='OK'), name='worker')
        parent: Agent[object, str] = Agent(
            _delegate_two_then_finish('boomer', 'worker'),
            capabilities=[SubAgents(agents=[SubAgent(boomer, contain_errors=True), SubAgent(worker)])],
        )
        result = await parent.run('go')
        assert result.output == 'all done'
        retries = _delegate_retries(result)
        assert any('crashed: ModelAPIError' in r and 'provider down' in r for r in retries)

    async def test_capability_default_contains_unset_delegate(self) -> None:
        boomer = Agent(_crash(), name='boomer')
        worker = Agent(TestModel(custom_output_text='OK'), name='worker')
        parent: Agent[object, str] = Agent(
            _delegate_two_then_finish('boomer', 'worker'),
            capabilities=[SubAgents(agents=[SubAgent(boomer), SubAgent(worker)], contain_errors=True)],
        )
        result = await parent.run('go')
        assert result.output == 'all done'
        assert any('crashed' in r for r in _delegate_retries(result))

    async def test_delegate_override_beats_capability_default(self) -> None:
        boomer = Agent(_crash(), name='boomer')
        parent: Agent[object, str] = Agent(
            _delegate_then_finish('boomer'),
            capabilities=[SubAgents(agents=[SubAgent(boomer, contain_errors=False)], contain_errors=True)],
        )
        with pytest.raises(ModelAPIError):
            await parent.run('go')

    async def test_user_error_always_propagates(self) -> None:
        def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise UserError('bad setup')

        boomer = Agent(FunctionModel(boom), name='boomer')
        parent: Agent[object, str] = Agent(
            _delegate_then_finish('boomer'),
            capabilities=[SubAgents(agents=[SubAgent(boomer, contain_errors=True)])],
        )
        # A setup bug must reach the developer even under containment, not become a retry.
        with pytest.raises(UserError):
            await parent.run('go')

    async def test_first_party_cancellation_bypasses_containment(self) -> None:
        def call_stop(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart('stop', {}, tool_call_id='s1')])

        canceller = Agent(FunctionModel(call_stop), name='canceller')

        @canceller.tool
        def stop(ctx: RunContext[object]) -> str:
            ctx.cancel()
            return 'ignored'

        parent: Agent[object, str] = Agent(
            _delegate_then_finish('canceller'),
            capabilities=[SubAgents(agents=[SubAgent(canceller, contain_errors=True)])],
        )
        result = await parent.run('go')
        # `RunCancelled` is a plain `Exception`: a deliberate `ctx.cancel()` in the child must
        # escape containment so pydantic-ai isolates it as a failed delegate return, not become
        # a `Sub-agent ... crashed` retry that invites the parent to re-delegate.
        assert result.output == 'all done'
        assert not _delegate_retries(result)
        returns = _delegate_returns(result)
        assert any('The sub-agent run was cancelled' in r for r in returns)

    async def test_on_failure_does_not_soften_contained_crash(self) -> None:
        boomer = Agent(_crash(), name='boomer')
        worker = Agent(TestModel(custom_output_text='OK'), name='worker')
        parent: Agent[object, str] = Agent(
            _delegate_two_then_finish('boomer', 'worker'),
            capabilities=[
                SubAgents(agents=[SubAgent(boomer, contain_errors=True, on_failure='soft note'), SubAgent(worker)])
            ],
        )
        result = await parent.run('go')
        # A crash stays loud: it is a retry, and on_failure's soft message never becomes a delegate return.
        assert any('crashed' in r for r in _delegate_retries(result))
        assert 'soft note' not in _delegate_returns(result)


class TestNameValidation:
    def test_duplicate_name_raises(self) -> None:
        first = Agent(TestModel(), name='dup')
        second = Agent(TestModel(), name='dup')
        with pytest.raises(ValueError, match="Duplicate sub-agent name 'dup'"):
            SubAgents(agents=[SubAgent(first), SubAgent(second)])

    def test_duplicate_via_name_override_raises(self) -> None:
        first = Agent(TestModel(), name='a')
        second = Agent(TestModel(), name='b')
        with pytest.raises(ValueError, match="Duplicate sub-agent name 'a'"):
            SubAgents(agents=[SubAgent(first), SubAgent(second, name='a')])

    def test_missing_name_raises(self) -> None:
        nameless = Agent(TestModel())
        with pytest.raises(ValueError, match='Sub-agent has no name'):
            SubAgents(agents=[SubAgent(nameless)])

    def test_name_override_satisfies_missing_agent_name(self) -> None:
        nameless = Agent(TestModel())
        capability = SubAgents(agents=[SubAgent(nameless, name='worker')])
        assert 'worker' in capability._by_name


def _prompt(messages: list[ModelMessage]) -> str:
    """The prompt the current run started from, which tells a shared model function which run it is serving."""
    part = messages[0].parts[-1]
    assert isinstance(part, UserPromptPart)
    return str(part.content)


def _delegate_to_self(task: str) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart('delegate_task', {'agent_name': 'self', 'task': task})])


class TestIncludeSelf:
    def test_listed_and_exposed_without_a_roster(self) -> None:
        capability = SubAgents[None](include_self=True, agent_folders=None)
        assert '- self: A fresh run of this same agent' in str(capability.get_instructions())
        assert capability.get_toolset() is not None

    async def test_delegates_to_the_running_agent(self) -> None:
        """The delegate is the running agent: its own tools, and the parent run's model."""
        offered: dict[str, list[str]] = {}

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            prompt = _prompt(messages)
            offered.setdefault(prompt, [tool.name for tool in info.function_tools])
            if len(messages) > 1:
                return ModelResponse(parts=[TextPart(f'{prompt} done')])
            if prompt == 'go':
                return _delegate_to_self('subtask')
            return ModelResponse(parts=[ToolCallPart('parent_tool', {})])

        # `inherit_tools=True` would register the parent's tools a second time on a delegate
        # that already has them, which fails on the duplicate name; it does not apply to `self`.
        agent = Agent(capabilities=[SubAgents(include_self=True, agent_folders=None, inherit_tools=True)])

        @agent.tool_plain
        def parent_tool() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'PT'

        result = await agent.run('go', model=FunctionModel(model_fn))
        assert result.output == 'go done'
        assert _delegate_returns(result) == ['subtask done']
        assert offered == {'go': ['parent_tool', 'delegate_task'], 'subtask': ['parent_tool', 'delegate_task']}

    async def test_depth_is_capped(self) -> None:
        """A run at `max_depth` gets neither the delegate tool nor the listing, so it does the work itself."""
        seen: dict[str, tuple[list[str], bool]] = {}

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            prompt = _prompt(messages)
            instructions = str(messages[0].instructions) if isinstance(messages[0], ModelRequest) else ''
            seen.setdefault(
                prompt, ([tool.name for tool in info.function_tools], 'Available sub-agents' in instructions)
            )
            if len(messages) > 1:
                return ModelResponse(parts=[TextPart(f'{prompt} done')])
            if 'delegate_task' in [tool.name for tool in info.function_tools]:
                return _delegate_to_self(f'level {len(seen) + 1}')
            return ModelResponse(parts=[TextPart(f'{prompt} done')])

        agent = Agent(
            FunctionModel(model_fn), capabilities=[SubAgents(include_self=True, max_depth=2, agent_folders=None)]
        )
        result = await agent.run('level 1')
        assert result.output == 'level 1 done'
        assert _delegate_returns(result) == ['level 2 done']
        assert seen == {'level 1': (['delegate_task'], True), 'level 2': ([], False)}

    async def test_directly_registered_toolset_hides_the_tool_at_max_depth(self) -> None:
        offered: list[list[str]] = []

        def worker_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            offered.append([tool.name for tool in info.function_tools])
            return ModelResponse(parts=[TextPart('w')])

        worker: Agent[object, str] = Agent(FunctionModel(worker_fn), name='worker')
        toolset = SubAgentToolset[object](
            agents={'worker': SubAgent(worker)},
            forward_usage=True,
            inherit_tools=False,
            shared_capabilities=(),
            event_stream_handler=None,
            tool_name='delegate_task',
            tool_retries=2,
            contain_errors=False,
            call_counts={},
            max_depth=2,
        )
        worker_with_toolset = Agent(FunctionModel(worker_fn), name='nested', toolsets=[toolset])
        parent = Agent(
            _delegate_then_finish('nested'),
            capabilities=[SubAgents(agents=[SubAgent(worker_with_toolset)], agent_folders=None)],
        )
        await parent.run('go')
        assert offered == [[]]

    async def test_depth_is_restored_after_a_delegation(self) -> None:
        """Consecutive delegations each start one level down, rather than one level below the last."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if _prompt(messages) != 'go':
                return ModelResponse(parts=[TextPart('ran')])
            if len(messages) < 5:
                return _delegate_to_self('subtask')
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(
            FunctionModel(model_fn), capabilities=[SubAgents(include_self=True, max_depth=2, agent_folders=None)]
        )
        result = await agent.run('go')
        assert _delegate_returns(result) == ['ran', 'ran']

    async def test_passing_it_to_run_is_refused(self) -> None:
        agent = Agent(TestModel(call_tools=[]))
        with pytest.raises(UserError, match='only carries what is bound to the `Agent`'):
            await agent.run('go', capabilities=[SubAgents(include_self=True, agent_folders=None)])

    async def test_a_different_bound_one_does_not_stand_in(self) -> None:
        """A run-level `SubAgents` overrides the agent's, so the agent's is not what the delegate would get."""
        agent = Agent(TestModel(call_tools=[]), capabilities=[SubAgents(include_self=True, agent_folders=None)])
        with pytest.raises(UserError, match='only carries what is bound to the `Agent`'):
            await agent.run('go', capabilities=[SubAgents(include_self=True, agent_folders=None, max_depth=2)])

    async def test_an_explicit_delegate_that_is_the_running_agent_keeps_its_own_model(self) -> None:
        """Only the reserved `self` entry runs on the parent run's model; listing the agent by hand does not."""
        runs: list[str] = []

        def own_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            runs.append(_prompt(messages))
            return ModelResponse(parts=[TextPart('own')])

        agent = Agent(FunctionModel(own_model), name='worker')
        await agent.run(
            'go',
            model=_delegate_then_finish('worker'),
            capabilities=[SubAgents(agents=[SubAgent(agent)], agent_folders=None)],
        )
        assert runs == ['do it']

    async def test_unknown_name_lists_self(self) -> None:
        worker = Agent(TestModel(custom_output_text='w'), name='worker')
        agent = Agent(
            _delegate_then_finish('worker', retries_before=1),
            capabilities=[SubAgents(agents=[SubAgent(worker)], include_self=True, agent_folders=None)],
        )
        result = await agent.run('go')
        retries = [
            str(part.content)
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, RetryPromptPart)
        ]
        assert retries == ["Unknown sub-agent 'ghost'. Available sub-agents: self, worker."]

    def test_self_name_is_reserved(self) -> None:
        with pytest.raises(ValueError, match="Sub-agent name 'self' is taken by the running agent"):
            SubAgents(agents=[SubAgent(Agent(TestModel(), name='self'))], include_self=True)

    def test_disk_agent_named_self_is_shadowed(self, tmp_path: Path) -> None:
        (tmp_path / 'self.md').write_text('---\nname: self\n---\n\nBody.\n', encoding='utf-8')
        with pytest.warns(UserWarning, match="Disk sub-agent 'self' is shadowed"):
            capability = SubAgents[None](include_self=True, agent_folders=[tmp_path])
        assert capability._by_name == {}  # pyright: ignore[reportPrivateUsage]

    def test_max_depth_counts_the_top_level_run(self) -> None:
        with pytest.raises(ValueError, match='must be at least 1; got 0'):
            SubAgents[None](max_depth=0)

    def test_combining_keeps_self(self) -> None:
        worker = Agent(TestModel(), name='worker')
        merged = SubAgents.combine(
            [SubAgents(agents=[SubAgent(worker)], agent_folders=None), SubAgents(include_self=True, agent_folders=None)]
        )
        assert isinstance(merged, SubAgents)
        assert merged.include_self
        assert '- self: ' in str(merged.get_instructions())
