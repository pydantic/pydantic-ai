"""Tests for typed events emitted by capabilities via `emit_event`."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from typing import Any

import pydantic
import pytest

from pydantic_ai import Agent, CapabilityEvent, CustomEvent, RunContext, UnknownCapabilityEvent
from pydantic_ai.capabilities import AbstractCapability, Capability, Hooks
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ToolReturnPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from ._inline_snapshot import snapshot

pytestmark = pytest.mark.anyio

FILE_SYSTEM_EVENTS = 'file_system'


@dataclass(kw_only=True)
class FileReadEvent(CapabilityEvent, namespace=FILE_SYSTEM_EVENTS):
    path: str


@dataclass(kw_only=True)
class FileProgressEvent(FileReadEvent, name='progress'):
    progress: float


@dataclass(kw_only=True)
class BeforeThingEvent(CapabilityEvent, namespace='decision'):
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True


def test_event_kind_definition():
    assert FileReadEvent(path='a.txt').kind == 'file_system.file_read'
    assert FileProgressEvent(path='a.txt', progress=0.5).kind == 'file_system.progress'


def test_missing_namespace_rejected():
    with pytest.raises(TypeError, match='requires a namespace'):

        @dataclass(kw_only=True)
        class MissingEvent(CapabilityEvent):  # pyright: ignore[reportUnusedClass]
            pass


def test_duplicate_kind_rejected():
    with pytest.raises(TypeError, match=r"Duplicate capability event kind 'file_system\.file_read'"):

        @dataclass(kw_only=True)
        class DuplicateEvent(  # pyright: ignore[reportUnusedClass]
            CapabilityEvent, namespace=FILE_SYSTEM_EVENTS, name='file_read'
        ):
            pass


def test_base_instantiation_rejected():
    with pytest.raises(TypeError, match='`CapabilityEvent` is a base class'):
        CapabilityEvent()


def test_round_trip():
    adapter = pydantic.TypeAdapter[AgentStreamEvent](AgentStreamEvent)
    event = FileReadEvent(path='a.txt', capability_id='files')
    dumped = adapter.dump_python(event)
    assert dumped == snapshot(
        {
            'kind': 'file_system.file_read',
            'capability_id': 'files',
            'tool_call_id': None,
            'tool_name': None,
            'event_kind': 'capability',
            'path': 'a.txt',
        }
    )
    assert adapter.validate_python(dumped) == event


def test_unknown_kind_and_late_registration():
    wire = {'event_kind': 'capability', 'kind': 'late.ready', 'value': 42}
    old_adapter = pydantic.TypeAdapter[AgentStreamEvent](AgentStreamEvent)
    with pytest.warns(UserWarning, match="Unknown event kind 'late.ready'"):
        unknown = old_adapter.validate_python(wire)
    assert unknown == snapshot(UnknownCapabilityEvent(kind='late.ready', data={'value': 42}))
    assert old_adapter.dump_python(unknown) == snapshot(
        {
            'value': 42,
            'kind': 'late.ready',
            'capability_id': None,
            'tool_call_id': None,
            'tool_name': None,
            'event_kind': 'capability',
        }
    )

    @dataclass(kw_only=True)
    class ReadyEvent(CapabilityEvent, namespace='late'):
        value: int

    with pytest.warns(UserWarning, match="Unknown event kind 'late.ready'"):
        assert isinstance(old_adapter.validate_python(wire), UnknownCapabilityEvent)
    assert pydantic.TypeAdapter[AgentStreamEvent](AgentStreamEvent).validate_python(wire) == ReadyEvent(value=42)


def test_mutable_decision_field_serializes():
    event = BeforeThingEvent()
    event.cancel()
    assert pydantic.TypeAdapter[AgentStreamEvent](AgentStreamEvent).dump_python(event)['cancelled'] is True


def _has_tool_return(messages: list[ModelMessage]) -> bool:
    return any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)


async def _tool_then_text(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
    if not _has_tool_return(messages):
        yield {0: DeltaToolCall(name='read_file', json_args='{}', tool_call_id='call_1')}
    else:
        yield 'done'


async def _only_text(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield 'done'


async def _collect(agent: Agent[Any, str]) -> list[AgentStreamEvent]:
    events: list[AgentStreamEvent] = []

    async def handler(ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    await agent.run('go', event_stream_handler=handler)
    return events


@dataclass
class EmitCapability(AbstractCapability[Any]):
    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        await ctx.emit_event(FileReadEvent(path='hook.txt'))
        return request_context


@pytest.mark.parametrize(
    ('capability', 'expected_id'), [(EmitCapability(id='my_id'), 'my_id'), (EmitCapability(), 'emit_capability')]
)
async def test_hook_emission_stamps_run_id(capability: EmitCapability, expected_id: str):
    events = await _collect(Agent(FunctionModel(stream_function=_only_text), capabilities=[capability]))
    assert [event.capability_id for event in events if isinstance(event, FileReadEvent)] == [expected_id]


async def test_capability_tool_emission_stamps_attribution():
    capability = Capability[Any](id='files')

    @capability.tool
    async def read_file(ctx: RunContext[Any]) -> str:
        await ctx.emit_event(FileReadEvent(path='tool.txt'))
        return 'ok'

    events = await _collect(Agent(FunctionModel(stream_function=_tool_then_text), capabilities=[capability]))
    assert [event for event in events if isinstance(event, FileReadEvent)] == snapshot(
        [FileReadEvent(path='tool.txt', capability_id='files', tool_call_id='call_1', tool_name='read_file')]
    )


async def test_app_tool_cannot_emit_capability_event():
    agent = Agent(FunctionModel(stream_function=_tool_then_text))

    @agent.tool
    async def read_file(ctx: RunContext[Any]) -> str:
        await ctx.emit_event(FileReadEvent(path='tool.txt'))
        return 'ok'

    with pytest.raises(UserError, match='Capability events belong to capabilities'):
        await _collect(agent)


async def test_capability_cannot_emit_custom_event():
    @dataclass
    class BadCapability(AbstractCapability[Any]):
        async def before_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            await ctx.emit_event(CustomEvent(name='bad'))
            return request_context

    agent = Agent(FunctionModel(stream_function=_only_text), capabilities=[BadCapability()])
    with pytest.raises(UserError, match='Capabilities should define and emit `CapabilityEvent`'):
        await agent.run('go')


async def test_hooks_can_emit_custom_event():
    hooks = Hooks()

    @hooks.on.before_model_request
    async def emit(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
        await ctx.emit_event(CustomEvent(name='bridge'))
        return request_context

    events = await _collect(Agent(FunctionModel(stream_function=_only_text), capabilities=[hooks]))
    assert [event for event in events if isinstance(event, CustomEvent)] == [CustomEvent(name='bridge')]
