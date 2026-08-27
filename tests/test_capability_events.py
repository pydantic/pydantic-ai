"""Tests for typed events emitted by capabilities via `emit_event`."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from typing import Any

import pydantic
import pytest

from pydantic_ai import Agent, CapabilityEvent, CustomEvent, RunContext, UnknownCapabilityEvent
from pydantic_ai.capabilities import AbstractCapability, Capability, Hooks, ProcessEventStream, WrapperCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelResponse, TextPart, ToolReturnPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from ._inline_snapshot import snapshot

pytestmark = pytest.mark.anyio

FILE_SYSTEM_EVENTS = 'test_file_system'


@dataclass(kw_only=True)
class FileReadEvent(CapabilityEvent, namespace=FILE_SYSTEM_EVENTS):
    path: str


@dataclass(kw_only=True)
class FileProgressEvent(FileReadEvent, name='progress'):
    progress: float


@dataclass(kw_only=True)
class BridgeEvent(CustomEvent, name='capability_bridge'):
    pass


@dataclass(kw_only=True)
class ThingStartEvent(CapabilityEvent, namespace='decision'):
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True


def test_event_kind_definition():
    assert FileReadEvent(path='a.txt').kind == 'test_file_system.file_read'
    assert FileProgressEvent(path='a.txt', progress=0.5).kind == 'test_file_system.progress'


def test_missing_namespace_rejected():
    with pytest.raises(TypeError, match='requires a namespace'):

        @dataclass(kw_only=True)
        class MissingEvent(CapabilityEvent):  # pyright: ignore[reportUnusedClass]
            pass


def test_duplicate_kind_rejected():
    with pytest.raises(TypeError, match=r"Duplicate capability event kind 'test_file_system\.file_read'"):

        @dataclass(kw_only=True)
        class DuplicateEvent(  # pyright: ignore[reportUnusedClass]
            CapabilityEvent, namespace=FILE_SYSTEM_EVENTS, name='file_read'
        ):
            pass


def test_redefined_event_class_replaces_registration():
    """Re-executing the same class definition (notebook cell re-run, reload) replaces, not errors."""

    def define() -> CapabilityEvent:
        @dataclass(kw_only=True)
        class RedefinedEvent(CapabilityEvent, namespace='redefinition'):
            value: int

        return RedefinedEvent(value=1)

    first, second = define(), define()
    assert type(first) is not type(second)
    assert second.kind == 'redefinition.redefined'
    adapter = pydantic.TypeAdapter[AgentStreamEvent](AgentStreamEvent)
    wire = {'event_kind': 'capability', 'kind': 'redefinition.redefined', 'value': 1}
    assert type(adapter.validate_python(wire)) is type(second)


def test_base_instantiation_rejected():
    with pytest.raises(TypeError, match='`CapabilityEvent` is a base class'):
        CapabilityEvent()


def test_slotted_event_class():
    """`@dataclass(slots=True)` recreates the class, re-invoking registration without the class
    arguments; the recreated class must keep its registered kind."""

    @dataclass(kw_only=True, slots=True)
    class SlottedEvent(CapabilityEvent, namespace='slotted'):
        value: int

    assert SlottedEvent(value=1).kind == 'slotted.slotted'
    adapter = pydantic.TypeAdapter[AgentStreamEvent](AgentStreamEvent)
    wire = {'event_kind': 'capability', 'kind': 'slotted.slotted', 'value': 1}
    assert type(adapter.validate_python(wire)) is SlottedEvent


def test_instance_kind_override_rejected():
    """A per-instance `kind` override would misroute (de)serialization, so construction rejects it."""
    with pytest.raises(ValueError, match=r"serializes under its registered kind 'test_file_system\.file_read'"):
        FileReadEvent(path='a.txt', kind='other.kind')


def test_non_dataclass_subclass_rejected_at_construction():
    """A registered subclass missing `@dataclass` never receives its injected `kind` default."""

    class PlainEvent(CapabilityEvent, namespace='plain'):
        pass

    with pytest.raises(ValueError, match='must be decorated with `@dataclass`'):
        PlainEvent()


def test_envelope_field_shadowing_rejected():
    """Payload fields can't shadow envelope fields: `data` is the unknown envelope's payload container."""
    with pytest.raises(TypeError, match='reserved for the event envelope: capability_id, data'):

        @dataclass(kw_only=True)
        class ShadowingEvent(CapabilityEvent, namespace='shadowing'):  # pyright: ignore[reportUnusedClass]
            data: dict[str, int]
            capability_id: str | None = None


def test_multi_segment_namespace_inherited():
    """A subclass of an event in a dotted namespace derives the full namespace, not its first segment."""

    @dataclass(kw_only=True)
    class NestedNamespaceEvent(CapabilityEvent, namespace='acme.files'):
        pass

    @dataclass(kw_only=True)
    class DerivedNestedEvent(NestedNamespaceEvent):
        pass

    assert NestedNamespaceEvent().kind == 'acme.files.nested_namespace'
    assert DerivedNestedEvent().kind == 'acme.files.derived_nested'


def test_round_trip():
    adapter = pydantic.TypeAdapter[AgentStreamEvent](AgentStreamEvent)
    event = FileReadEvent(path='a.txt', capability_id='files')
    dumped = adapter.dump_python(event)
    assert dumped == snapshot(
        {
            'kind': 'test_file_system.file_read',
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
    event = ThingStartEvent()
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


@pytest.mark.parametrize(
    ('capability_id', 'expected_id'),
    [('files', 'files'), (None, 'capability')],
    ids=['explicit-id', 'implicit-id'],
)
async def test_capability_tool_emission_stamps_attribution(capability_id: str | None, expected_id: str):
    """Tools contributed by a capability stamp its run id, whether explicitly set or derived."""
    capability = Capability[Any](id=capability_id)

    @capability.tool
    async def read_file(ctx: RunContext[Any]) -> str:
        await ctx.emit_event(FileReadEvent(path='tool.txt'))
        return 'ok'

    events = await _collect(Agent(FunctionModel(stream_function=_tool_then_text), capabilities=[capability]))
    assert [event for event in events if isinstance(event, FileReadEvent)] == [
        FileReadEvent(path='tool.txt', capability_id=expected_id, tool_call_id='call_1', tool_name='read_file')
    ]


async def test_app_tool_cannot_emit_capability_event():
    agent = Agent(FunctionModel(stream_function=_tool_then_text))

    @agent.tool
    async def read_file(ctx: RunContext[Any]) -> str:
        await ctx.emit_event(FileReadEvent(path='tool.txt'))
        return 'ok'  # pragma: no cover - the emit above raises

    with pytest.raises(UserError, match='Capability events belong to capabilities'):
        await _collect(agent)


async def test_agent_run_cannot_emit_capability_event():
    """`AgentRun.emit_event` is a driver-code (application) surface; the guard also holds at runtime."""

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(FunctionModel(model_function))

    async with agent.iter('go') as run:
        with pytest.raises(UserError, match='Capability events belong to capabilities'):
            await run.emit_event(FileReadEvent(path='tool.txt'))  # pyright: ignore[reportArgumentType]
        async for _ in run:
            pass


async def test_capability_cannot_emit_custom_event():
    @dataclass
    class BadCapability(AbstractCapability[Any]):
        async def before_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            await ctx.emit_event(BridgeEvent())
            return request_context  # pragma: no cover - the emit above raises

    agent = Agent(FunctionModel(stream_function=_only_text), capabilities=[BadCapability()])
    with pytest.raises(UserError, match='Capabilities should define and emit `CapabilityEvent`'):
        await agent.run('go')


async def test_hooks_can_emit_custom_event():
    hooks = Hooks()

    @hooks.on.before_model_request
    async def emit(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
        await ctx.emit_event(BridgeEvent())
        return request_context

    events = await _collect(Agent(FunctionModel(stream_function=_only_text), capabilities=[hooks]))
    assert [event for event in events if isinstance(event, CustomEvent)] == [BridgeEvent()]


async def test_wrapped_hooks_can_emit_custom_event():
    """Wrapping an app-facing capability must not revoke its callbacks' `CustomEvent` permission."""
    hooks = Hooks[Any]()

    @hooks.on.before_model_request
    async def emit(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
        await ctx.emit_event(BridgeEvent())
        return request_context

    wrapper = WrapperCapability(wrapped=hooks, id='wrapped_hooks')
    events = await _collect(Agent(FunctionModel(stream_function=_only_text), capabilities=[wrapper]))
    assert [event for event in events if isinstance(event, CustomEvent)] == [BridgeEvent()]


def test_subclass_post_init_override_keeps_guards():
    """A subclass `__post_init__` that doesn't call `super()` cannot bypass the construction guards."""

    @dataclass(kw_only=True)
    class GuardedOverrideEvent(CapabilityEvent, namespace='guarded_override'):
        value: int = 0

        def __post_init__(self) -> None:
            self.value += 1

    with pytest.raises(ValueError, match='serializes under its registered kind'):
        GuardedOverrideEvent(kind='other.kind')
    assert GuardedOverrideEvent().value == 1


async def test_process_event_stream_handler_can_emit_custom_event():
    """`ProcessEventStream` runs app callbacks, so they keep `CustomEvent` permission."""
    emitted = False

    async def handler(ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        nonlocal emitted
        async for _ in stream:
            if not emitted:
                emitted = True
                await ctx.emit_event(BridgeEvent())

    events = await _collect(
        Agent(FunctionModel(stream_function=_only_text), capabilities=[ProcessEventStream(handler)])
    )
    assert [event for event in events if isinstance(event, CustomEvent)] == [BridgeEvent()]


async def test_pre_set_capability_id_is_preserved():
    """A capability re-emitting an event on another instance's behalf keeps the original attribution."""

    @dataclass
    class RelayCapability(AbstractCapability[Any]):
        async def before_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            await ctx.emit_event(FileReadEvent(path='relayed.txt', capability_id='original_instance'))
            return request_context

    events = await _collect(Agent(FunctionModel(stream_function=_only_text), capabilities=[RelayCapability()]))
    assert [event.capability_id for event in events if isinstance(event, FileReadEvent)] == ['original_instance']


async def test_emission_with_unresolvable_tool_name_attributes_nothing():
    """A context whose `tool_name` no longer resolves in the tool manager attributes no capability.

    This can happen when a context copy outlives a dynamic toolset change; the emission still
    succeeds for a `CustomEvent`, un-attributed.
    """
    import dataclasses as dc

    agent: Agent[None, str] = Agent(FunctionModel(stream_function=_tool_then_text))

    @agent.tool
    async def read_file(ctx: RunContext[Any]) -> str:
        stale = dc.replace(ctx, tool_name='vanished')
        await stale.emit_event(BridgeEvent())
        return 'ok'

    events = await _collect(agent)
    bridge = [event for event in events if isinstance(event, BridgeEvent)]
    assert len(bridge) == 1
    assert bridge[0].tool_name == 'vanished'
