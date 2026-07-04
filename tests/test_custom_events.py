"""Tests for custom events emitted into the run event stream via `emit_event`."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from typing import Any

import pydantic
import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    CustomEvent,
    ModelMessage,
    ToolReturnPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.run import AgentRunResultEvent

from ._inline_snapshot import snapshot

pytestmark = pytest.mark.anyio


def _has_tool_return(messages: list[ModelMessage]) -> bool:
    return any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)


async def _tool_then_text(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
    """Stream a `progress` tool call on the first request, then final text."""
    if not _has_tool_return(messages):
        yield {0: DeltaToolCall(name='progress', json_args='{}', tool_call_id='call_1')}
    else:
        yield 'done'


async def _collect_events(agent: Agent[Any, str], prompt: str = 'go') -> list[AgentStreamEvent]:
    events: list[AgentStreamEvent] = []

    async def event_stream_handler(ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    await agent.run(prompt, event_stream_handler=event_stream_handler)
    return events


async def test_emit_from_tool_auto_stamps_tool_call_id():
    """A `CustomEvent` emitted from a tool reaches the stream with `tool_call_id` auto-stamped."""
    agent = Agent(FunctionModel(stream_function=_tool_then_text))

    @agent.tool
    def progress(ctx: RunContext[Any]) -> str:
        ctx.emit_event(CustomEvent(name='progress', data={'pct': 50}))
        return 'ok'

    events = await _collect_events(agent)
    custom = [event for event in events if isinstance(event, CustomEvent)]
    assert custom == snapshot([CustomEvent(name='progress', data={'pct': 50}, tool_call_id='call_1')])


async def test_explicit_tool_call_id_preserved():
    """An explicit `tool_call_id` on the event is not overwritten by the current tool call."""
    agent = Agent(FunctionModel(stream_function=_tool_then_text))

    @agent.tool
    def progress(ctx: RunContext[Any]) -> str:
        ctx.emit_event(CustomEvent(name='progress', data=None, tool_call_id='explicit'))
        return 'ok'

    events = await _collect_events(agent)
    custom = [event for event in events if isinstance(event, CustomEvent)]
    assert custom == snapshot([CustomEvent(name='progress', tool_call_id='explicit')])


async def test_emit_from_capability_hook():
    """A `CustomEvent` emitted from a capability hook (workflow-side) reaches the stream, un-stamped."""

    async def only_text(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'done'

    @dataclass
    class EmitCapability(AbstractCapability[Any]):
        async def before_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            ctx.emit_event(CustomEvent(name='starting', data='before request'))
            return request_context

    agent = Agent(FunctionModel(stream_function=only_text), capabilities=[EmitCapability()])

    events = await _collect_events(agent)
    custom = [event for event in events if isinstance(event, CustomEvent)]
    assert custom == snapshot([CustomEvent(name='starting', data='before request')])


async def test_agent_run_emit_event():
    """Code driving `agent.iter()` can inject events via `AgentRun.emit_event`."""

    async def only_text(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'done'

    agent = Agent(FunctionModel(stream_function=only_text))

    collected: list[AgentStreamEvent] = []
    async with agent.iter('go') as run:
        run.emit_event(CustomEvent(name='external', data={'source': 'bus'}))
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        collected.append(event)

    custom = [event for event in collected if isinstance(event, CustomEvent)]
    assert custom == snapshot([CustomEvent(name='external', data={'source': 'bus'})])


async def test_agent_run_emit_event_before_call_tools_stream():
    """Events emitted between nodes drain at the start of the next response-handling stream."""
    agent = Agent(FunctionModel(stream_function=_tool_then_text))

    @agent.tool
    def progress(ctx: RunContext[Any]) -> str:
        return 'ok'

    collected: list[AgentStreamEvent] = []
    async with agent.iter('go') as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as request_stream:
                    async for _ in request_stream:
                        pass
            elif Agent.is_call_tools_node(node):
                run.emit_event(CustomEvent(name='before-tools'))
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        collected.append(event)

    # The custom event drains before the node's own events.
    assert [event.event_kind for event in collected[:2]] == snapshot(['custom', 'function_tool_call'])


async def test_emit_from_output_validator():
    """An event emitted after the last framework event (from an output validator) still surfaces."""

    async def only_text(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'done'

    agent = Agent(FunctionModel(stream_function=only_text))

    @agent.output_validator
    def validate(ctx: RunContext[Any], output: str) -> str:
        ctx.emit_event(CustomEvent(name='validated'))
        return output

    events = await _collect_events(agent)
    custom = [event for event in events if isinstance(event, CustomEvent)]
    assert custom == snapshot([CustomEvent(name='validated')])


async def test_custom_events_excluded_from_stream_output():
    """Pending custom events don't disturb `stream_output`, which only reflects model response events."""
    agent = Agent(FunctionModel(stream_function=_tool_then_text))

    @agent.tool
    def progress(ctx: RunContext[Any]) -> str:
        ctx.emit_event(CustomEvent(name='progress'))
        return 'ok'

    outputs: list[str] = []
    async with agent.iter('go') as run:
        run.emit_event(CustomEvent(name='external'))
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for output in stream.stream_output(debounce_by=None):
                        outputs.append(output)

    assert outputs[-1] == 'done'


async def test_surfaced_via_run_stream_events():
    """Custom events surface through `run_stream_events`."""
    agent = Agent(FunctionModel(stream_function=_tool_then_text))

    @agent.tool
    def progress(ctx: RunContext[Any]) -> str:
        ctx.emit_event(CustomEvent(name='progress', data={'pct': 50}))
        return 'ok'

    events: list[AgentStreamEvent | AgentRunResultEvent[str]] = []
    async with agent.run_stream_events('go') as stream:
        async for event in stream:
            events.append(event)

    custom = [event for event in events if isinstance(event, CustomEvent)]
    assert custom == snapshot([CustomEvent(name='progress', data={'pct': 50}, tool_call_id='call_1')])


async def test_surfaced_via_run_stream():
    """Custom events surface through the `run_stream` event stream handler."""
    events: list[AgentStreamEvent] = []

    async def event_stream_handler(ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    agent = Agent(FunctionModel(stream_function=_tool_then_text))

    @agent.tool
    def progress(ctx: RunContext[Any]) -> str:
        ctx.emit_event(CustomEvent(name='progress', data={'pct': 50}))
        return 'ok'

    async with agent.run_stream('go', event_stream_handler=event_stream_handler) as result:
        assert await result.get_output() == 'done'

    custom = [event for event in events if isinstance(event, CustomEvent)]
    assert custom == snapshot([CustomEvent(name='progress', data={'pct': 50}, tool_call_id='call_1')])


def test_emit_without_buffer_raises():
    """A `RunContext` not backed by a running agent has nowhere to emit to."""
    ctx = RunContext[Any](deps=None, model=FunctionModel(stream_function=_tool_then_text), usage=None)  # type: ignore[arg-type]
    with pytest.raises(UserError, match='`emit_event` is only available during an agent run'):
        ctx.emit_event(CustomEvent(name='progress'))


def test_serialization_round_trip():
    """A `CustomEvent` round-trips through the `AgentStreamEvent` discriminated union."""
    adapter = pydantic.TypeAdapter[AgentStreamEvent](AgentStreamEvent)
    event = CustomEvent(name='progress', data={'pct': 50, 'label': 'halfway'}, tool_call_id='call_1')
    dumped = adapter.dump_python(event)
    assert dumped == snapshot(
        {
            'name': 'progress',
            'data': {'pct': 50, 'label': 'halfway'},
            'tool_call_id': 'call_1',
            'event_kind': 'custom',
        }
    )
    assert adapter.validate_python(dumped) == event
