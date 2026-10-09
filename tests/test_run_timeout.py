"""Tests for bounding a whole agent run with `timeout=` and `RunContext.deadline`.

Expiry reuses first-party cancellation, so these are unit-style tests like `test_run_cancellation.py`:
the behavior under test is control flow around a timer, which no recorded provider response can trigger.
Models and tools that would block forever stand in for a slow provider or tool.
"""

from __future__ import annotations as _annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, RunCancelled, RunContext, RunTimedOut
from pydantic_ai._cancel import RunCancellation
from pydantic_ai.capabilities import AbstractCapability, WrapperCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage

from .conftest import IsDatetime, IsNow, IsStr

pytestmark = pytest.mark.anyio

# Short enough to keep the suite fast, long enough that a run that should finish does so well before it.
SHORT_TIMEOUT = 0.05
LONG_TIMEOUT = 60


async def _never_returns(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    await asyncio.Event().wait()
    raise AssertionError('unreachable')  # pragma: no cover


async def _never_finishes_streaming(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
    yield 'partial '
    yield 'output'
    await asyncio.Event().wait()
    raise AssertionError('unreachable')  # pragma: no cover


def _call_slow_tool(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
        return ModelResponse(parts=[TextPart('done')])
    return ModelResponse(parts=[ToolCallPart('slow_tool', {}, tool_call_id='call_slow')])


def _slow_tool_agent() -> Agent[object, str]:
    agent = Agent(FunctionModel(_call_slow_tool))

    @agent.tool_plain
    async def slow_tool() -> str:
        await asyncio.Event().wait()
        raise AssertionError('unreachable')  # pragma: no cover

    return agent


async def test_timeout_during_model_request():
    agent = Agent(FunctionModel(_never_returns))

    with pytest.raises(RunTimedOut) as exc_info:
        await agent.run('go', timeout=SHORT_TIMEOUT)

    exc = exc_info.value
    assert isinstance(exc, RunCancelled)
    assert isinstance(exc, TimeoutError)
    assert str(exc) == 'The agent run timed out.'
    assert exc.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='go', timestamp=IsNow(tz=UTC))],
                timestamp=IsNow(tz=UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
            )
        ]
    )


async def test_timeout_during_tool_call_keeps_resumable_state():
    agent = _slow_tool_agent()

    with pytest.raises(RunTimedOut) as exc_info:
        await agent.run('go', timeout=SHORT_TIMEOUT)

    messages = exc_info.value.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='go', timestamp=IsNow(tz=UTC))],
                timestamp=IsNow(tz=UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='slow_tool', args={}, tool_call_id='call_slow')],
                usage=RequestUsage(input_tokens=51, output_tokens=2),
                model_name='function:_call_slow_tool:',
                timestamp=IsNow(tz=UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[],
                timestamp=IsNow(tz=UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
                state='interrupted',
            ),
        ]
    )
    assert exc_info.value.usage.requests == 1

    # The partial run resumes like any cancelled run: the unanswered tool call gets a synthesized return.
    result = await agent.run(message_history=messages)
    assert result.output == 'done'


async def test_timeout_during_run_stream_keeps_partial_response():
    agent = Agent(FunctionModel(stream_function=_never_finishes_streaming))

    chunks: list[str] = []
    with pytest.raises(RunTimedOut) as exc_info:
        async with agent.run_stream('go', timeout=SHORT_TIMEOUT) as result:
            async for chunk in result.stream_text(delta=True, debounce_by=None):
                chunks.append(chunk)

    assert chunks == ['partial ', 'output']
    response = exc_info.value.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert response.state == 'interrupted'
    assert response.parts == [TextPart(content='partial output')]


async def test_timeout_during_run_stream_events():
    agent = Agent(FunctionModel(stream_function=_never_finishes_streaming))

    with pytest.raises(RunTimedOut) as exc_info:
        async with agent.run_stream_events('go', timeout=SHORT_TIMEOUT) as events:
            async for _event in events:
                pass

    response = exc_info.value.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert response.state == 'interrupted'


async def test_timeout_during_iter():
    agent = _slow_tool_agent()

    with pytest.raises(RunTimedOut):
        async with agent.iter('go', timeout=SHORT_TIMEOUT) as agent_run:
            assert agent_run.ctx.deps.deadline == IsDatetime(approx=datetime.now(UTC), delta=timedelta(seconds=5))
            async for _node in agent_run:
                pass


def test_timeout_run_sync():
    agent = _slow_tool_agent()

    with pytest.raises(RunTimedOut):
        agent.run_sync('go', timeout=SHORT_TIMEOUT)


def test_timeout_run_stream_sync():
    agent = Agent(FunctionModel(stream_function=_never_finishes_streaming))

    chunks: list[str] = []
    with pytest.raises(RunTimedOut):
        with agent.run_stream_sync('go', timeout=SHORT_TIMEOUT) as result:
            for chunk in result.stream_text(delta=True, debounce_by=None):
                chunks.append(chunk)
    assert chunks == ['partial ', 'output']


async def test_already_expired_timeout_stops_before_the_first_request():
    requests = 0

    def model_func(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:  # pragma: no cover
        nonlocal requests
        requests += 1
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model_func))

    with pytest.raises(RunTimedOut) as exc_info:
        await agent.run('go', timeout=0)

    assert requests == 0
    assert exc_info.value.usage.requests == 0


async def test_run_finishing_before_its_deadline_is_unaffected():
    agent = Agent(TestModel())

    result = await agent.run('go', timeout=SHORT_TIMEOUT)
    assert result.output == 'success (no tool calls)'

    # The timer is disarmed with the run: it must not cancel later work on the same task.
    await asyncio.sleep(SHORT_TIMEOUT * 2)


async def test_deadline_and_remaining_time_on_run_context():
    seen: list[tuple[datetime | None, float | None]] = []
    agent = Agent(TestModel())

    @agent.tool
    def check_deadline(ctx: RunContext[object]) -> str:
        seen.append((ctx.deadline, ctx.remaining_time()))
        return 'ok'

    before = datetime.now(UTC)
    await agent.run('go', timeout=LONG_TIMEOUT)
    [(deadline, remaining)] = seen
    assert deadline is not None
    assert deadline.tzinfo is UTC
    assert before + timedelta(seconds=LONG_TIMEOUT) <= deadline <= datetime.now(UTC) + timedelta(seconds=LONG_TIMEOUT)
    assert remaining is not None
    assert 0 < remaining <= LONG_TIMEOUT

    seen.clear()
    await agent.run('go')
    assert seen == [(None, None)]


def _delegating_agent(sub_agent: Agent[object, str], *, sub_timeout: float | None) -> Agent[object, str]:
    agent = Agent(TestModel())

    @agent.tool
    async def delegate(ctx: RunContext[object]) -> str:
        result = await sub_agent.run('sub', timeout=sub_timeout)
        return result.output

    return agent


def _deadline_recording_agent(seen: list[datetime | None]) -> Agent[object, str]:
    sub_agent = Agent(TestModel())

    @sub_agent.tool
    def record(ctx: RunContext[object]) -> str:
        seen.append(ctx.deadline)
        return 'ok'

    return sub_agent


async def test_sub_agent_inherits_deadline():
    seen: list[datetime | None] = []
    parent_deadlines: list[datetime | None] = []
    sub_agent = _deadline_recording_agent(seen)
    agent = Agent(TestModel())

    @agent.tool
    async def delegate(ctx: RunContext[object]) -> str:
        parent_deadlines.append(ctx.deadline)
        return (await sub_agent.run('sub')).output

    await agent.run('go', timeout=LONG_TIMEOUT)
    assert parent_deadlines[0] is not None
    assert seen == parent_deadlines

    # Outside the run, nothing is inherited.
    seen.clear()
    await sub_agent.run('sub')
    assert seen == [None]


async def test_sub_agent_can_shorten_but_not_extend_inherited_deadline():
    seen: list[datetime | None] = []
    sub_agent = _deadline_recording_agent(seen)

    await _delegating_agent(sub_agent, sub_timeout=LONG_TIMEOUT * 10).run('go', timeout=LONG_TIMEOUT)
    await _delegating_agent(sub_agent, sub_timeout=LONG_TIMEOUT / 10).run('go', timeout=LONG_TIMEOUT)

    now = datetime.now(UTC)
    capped, shortened = seen
    assert capped is not None and capped <= now + timedelta(seconds=LONG_TIMEOUT)
    assert shortened is not None and shortened <= now + timedelta(seconds=LONG_TIMEOUT / 10)


async def test_sub_agent_times_out_on_inherited_deadline():
    sub_agent = _slow_tool_agent()

    with pytest.raises(RunTimedOut) as exc_info:
        await _delegating_agent(sub_agent, sub_timeout=None).run('go', timeout=SHORT_TIMEOUT)

    # The parent reports its own history, not the sub-agent's.
    [request, response, *_] = exc_info.value.all_messages()
    assert isinstance(request, ModelRequest)
    assert request.parts[0] == UserPromptPart(content='go', timestamp=IsNow(tz=UTC))
    assert isinstance(response, ModelResponse)
    assert isinstance(response.parts[0], ToolCallPart)
    assert response.parts[0].tool_name == 'delegate'


async def test_sub_agent_own_timeout_is_a_failed_tool_call():
    """Like a sub-agent's own `cancel()`, its own shorter timeout fails the tool call, not the calling run."""
    sub_agent = _slow_tool_agent()

    result = await _delegating_agent(sub_agent, sub_timeout=SHORT_TIMEOUT).run('go')

    tool_return = result.all_messages()[2].parts[0]
    assert tool_return == snapshot(
        ToolReturnPart(
            tool_name='delegate',
            content='The sub-agent run was cancelled: The agent run timed out.',
            tool_call_id='pyd_ai_tool_call_id__delegate',
            timestamp=IsNow(tz=UTC),
            outcome='failed',
        )
    )


async def test_nested_timeout_reaching_the_run_edge_keeps_its_type():
    """A nested run's `RunTimedOut` escaping into the outer run (not via a tool) is re-stamped as the outer run's."""
    agent = Agent(TestModel())
    sub_agent = Agent(FunctionModel(_never_returns))

    with pytest.raises(RunTimedOut) as exc_info:
        async with agent.iter('go') as agent_run:
            async for _node in agent_run:
                await sub_agent.run('sub', timeout=SHORT_TIMEOUT)

    assert str(exc_info.value) == 'The agent run timed out in a nested run.'
    nested = exc_info.value.__cause__
    assert isinstance(nested, RunTimedOut)
    # The outer run's state (its first node hadn't run yet), not the nested run's.
    assert exc_info.value.run_id != nested.run_id
    assert exc_info.value.all_messages() == []
    assert len(nested.all_messages()) == 1


async def test_cancel_before_expiry_stays_run_cancelled():
    agent = Agent(TestModel())

    @agent.tool
    async def cancel_run(ctx: RunContext[object]) -> str:
        ctx.cancel()
        await asyncio.sleep(SHORT_TIMEOUT * 4)
        raise AssertionError('unreachable')  # pragma: no cover

    with pytest.raises(RunCancelled) as exc_info:
        await agent.run('go', timeout=SHORT_TIMEOUT)

    assert not isinstance(exc_info.value, RunTimedOut)


async def test_expire_is_a_no_op_after_cancel_or_finish():
    cancelled = RunCancellation()
    cancelled.cancel()
    cancelled.expire()
    assert cancelled.cancel_requested
    assert not cancelled.timed_out

    finished = RunCancellation()
    finished.finish()
    finished.expire()
    assert not finished.cancel_requested
    assert not finished.timed_out


async def test_timeout_does_not_trigger_fallback():
    fallback_calls = 0

    def fallback_model_func(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:  # pragma: no cover
        nonlocal fallback_calls
        fallback_calls += 1
        return ModelResponse(parts=[TextPart('fallback')])

    agent = Agent(FallbackModel(FunctionModel(_never_returns), FunctionModel(fallback_model_func)))

    with pytest.raises(RunTimedOut):
        await agent.run('go', timeout=SHORT_TIMEOUT)

    assert fallback_calls == 0


_FIXED_NOW = datetime(2030, 1, 1, tzinfo=UTC)


@dataclass
class _ReplaySafeClock(AbstractCapability[object]):
    """Stands in for a durability capability's replay-safe clock and recorded start time."""

    def _run_clock(self) -> Callable[[], datetime] | None:
        return lambda: _FIXED_NOW

    async def _run_start_time(self) -> datetime | None:
        return _FIXED_NOW - timedelta(seconds=10)


async def test_capability_supplies_clock_and_start_time():
    """The deadline counts from the capability's start time, and remaining time reads its clock."""
    seen: list[tuple[datetime | None, float | None]] = []
    agent = Agent(TestModel(), capabilities=[WrapperCapability(wrapped=_ReplaySafeClock())])

    @agent.tool
    def check_deadline(ctx: RunContext[object]) -> str:
        seen.append((ctx.deadline, ctx.remaining_time()))
        return 'ok'

    with pytest.raises(RunTimedOut):
        # The fixed clock says the 10-second budget is already spent by the time the run starts.
        await agent.run('go', timeout=10)
    assert seen == []

    await agent.run('go', timeout=60)
    assert seen == [(_FIXED_NOW + timedelta(seconds=50), 50.0)]
