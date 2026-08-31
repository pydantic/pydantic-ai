"""Tests for custom events emitted into the run event stream via `emit`."""

from __future__ import annotations

import asyncio
import sys
import textwrap
import warnings
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from typing import Any

import pydantic
import pytest

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai._event_registry import set_replay_isolation_guard
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    CUSTOM_EVENT_TYPES,
    AgentStreamEvent,
    CustomEvent,
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UnknownCustomEvent,
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


@dataclass(kw_only=True)
class ProgressEvent(CustomEvent):
    """Reusable payload-bearing event for tool-emission tests."""

    payload: Any = None


@dataclass(kw_only=True)
class ExternalEvent(CustomEvent):
    """Reusable event for driver-code (`AgentRun.emit`) tests."""

    payload: Any = None


@dataclass(kw_only=True)
class StartingEvent(CustomEvent):
    payload: Any = None


@dataclass(kw_only=True)
class ValidatedEvent(CustomEvent):
    pass


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
    async def progress(ctx: RunContext[Any]) -> str:
        await ctx.emit(ProgressEvent(payload={'pct': 50}))
        return 'ok'

    events = await _collect_events(agent)
    custom = [event for event in events if isinstance(event, CustomEvent)]
    assert custom == snapshot([ProgressEvent(payload={'pct': 50}, tool_call_id='call_1', tool_name='progress')])


async def test_explicit_tool_call_id_preserved():
    """An explicit `tool_call_id` on the event is not overwritten by the current tool call."""
    agent = Agent(FunctionModel(stream_function=_tool_then_text))

    @agent.tool
    async def progress(ctx: RunContext[Any]) -> str:
        await ctx.emit(ProgressEvent(tool_call_id='explicit'))
        return 'ok'

    events = await _collect_events(agent)
    custom = [event for event in events if isinstance(event, CustomEvent)]
    assert custom == snapshot([ProgressEvent(tool_call_id='explicit')])


async def test_emit_from_capability_hook():
    """A `CustomEvent` emitted from a capability hook (workflow-side) reaches the stream, un-stamped."""

    async def only_text(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'done'

    @dataclass
    class EmitCapability(AbstractCapability[Any]):
        @property
        def _emits_app_events(self) -> bool:
            return True

        async def before_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            await ctx.emit(StartingEvent(payload='before request'))
            return request_context

    agent = Agent(FunctionModel(stream_function=only_text), capabilities=[EmitCapability()])

    events = await _collect_events(agent)
    custom = [event for event in events if isinstance(event, CustomEvent)]
    assert custom == snapshot([StartingEvent(payload='before request')])


async def test_agent_run_emit_event():
    """Code driving `agent.iter()` can inject events via `AgentRun.emit`."""

    async def only_text(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'done'

    agent = Agent(FunctionModel(stream_function=only_text))

    collected: list[AgentStreamEvent] = []
    async with agent.iter('go') as run:
        await run.emit(ExternalEvent(payload={'source': 'bus'}))
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        collected.append(event)

    custom = [event for event in collected if isinstance(event, CustomEvent)]
    assert custom == snapshot([ExternalEvent(payload={'source': 'bus'})])


async def test_agent_run_emit_event_after_end_rejected():
    """Emitting after the run has ended fails loudly instead of silently never delivering."""

    def only_text(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(FunctionModel(only_text))
    async with agent.iter('go') as run:
        async for _ in run:
            pass
        with pytest.raises(UserError, match='cannot be called after the run has ended'):
            await run.emit(ExternalEvent())


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
                await run.emit(ExternalEvent())
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
    async def validate(ctx: RunContext[Any], output: str) -> str:
        await ctx.emit(ValidatedEvent())
        return output

    events = await _collect_events(agent)
    custom = [event for event in events if isinstance(event, CustomEvent)]
    assert custom == snapshot([ValidatedEvent()])


async def test_custom_events_excluded_from_stream_output():
    """Pending custom events don't disturb `stream_output`, which only reflects model response events."""
    agent = Agent(FunctionModel(stream_function=_tool_then_text))

    @agent.tool
    async def progress(ctx: RunContext[Any]) -> str:
        await ctx.emit(ProgressEvent())
        return 'ok'

    outputs: list[str] = []
    async with agent.iter('go') as run:
        await run.emit(ExternalEvent())
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
    async def progress(ctx: RunContext[Any]) -> str:
        await ctx.emit(ProgressEvent(payload={'pct': 50}))
        return 'ok'

    events: list[AgentStreamEvent | AgentRunResultEvent[str]] = []
    async with agent.run_stream_events('go') as stream:
        async for event in stream:
            events.append(event)

    custom = [event for event in events if isinstance(event, CustomEvent)]
    assert custom == snapshot([ProgressEvent(payload={'pct': 50}, tool_call_id='call_1', tool_name='progress')])


async def test_surfaced_via_run_stream():
    """Custom events surface through the `run_stream` event stream handler."""
    events: list[AgentStreamEvent] = []

    async def event_stream_handler(ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    agent = Agent(FunctionModel(stream_function=_tool_then_text))

    @agent.tool
    async def progress(ctx: RunContext[Any]) -> str:
        await ctx.emit(ProgressEvent(payload={'pct': 50}))
        return 'ok'

    async with agent.run_stream('go', event_stream_handler=event_stream_handler) as result:
        assert await result.get_output() == 'done'

    custom = [event for event in events if isinstance(event, CustomEvent)]
    assert custom == snapshot([ProgressEvent(payload={'pct': 50}, tool_call_id='call_1', tool_name='progress')])


async def test_emit_without_buffer_raises():
    """A `RunContext` not backed by a running agent has nowhere to emit to."""
    ctx = RunContext[Any](deps=None, model=FunctionModel(stream_function=_tool_then_text), usage=None)  # type: ignore[arg-type]
    with pytest.raises(UserError, match='`emit` is only available during an agent run'):
        await ctx.emit(ProgressEvent())


def test_serialization_round_trip():
    """A `CustomEvent` round-trips through the `AgentStreamEvent` discriminated union."""
    adapter = pydantic.TypeAdapter[AgentStreamEvent](AgentStreamEvent)
    event = ProgressEvent(payload={'pct': 50, 'label': 'halfway'}, tool_call_id='call_1')
    dumped = adapter.dump_python(event)
    assert dumped == snapshot(
        {
            'name': 'progress',
            'payload': {'pct': 50, 'label': 'halfway'},
            'tool_call_id': 'call_1',
            'tool_name': None,
            'event_kind': 'custom',
        }
    )
    assert adapter.validate_python(dumped) == event


def test_custom_event_requires_name():
    """`name` has a static-only default (so typed subclasses don't require it); it can't end up empty."""
    with pytest.raises(ValueError, match='A custom event requires a `name`'):
        UnknownCustomEvent(name='', data={'x': 1})


def test_custom_event_base_not_instantiable():
    """`CustomEvent` is the family base; payloads are carried by typed subclasses."""
    with pytest.raises(TypeError, match='`CustomEvent` is a base class'):
        CustomEvent(name='x')


@dataclass(kw_only=True)
class UploadProgressEvent(CustomEvent):
    done: int
    total: int


@dataclass(kw_only=True)
class RenamedEvent(CustomEvent, name='sync_renamed'):
    label: str


def test_typed_subclass_round_trip():
    """A typed subclass round-trips through the union back to its own class, payload as its own fields."""
    adapter = pydantic.TypeAdapter[AgentStreamEvent](AgentStreamEvent)
    event = UploadProgressEvent(done=3, total=9)
    assert event.name == 'upload_progress'
    dumped = adapter.dump_python(event)
    assert dumped == snapshot(
        {
            'name': 'upload_progress',
            'tool_call_id': None,
            'tool_name': None,
            'event_kind': 'custom',
            'done': 3,
            'total': 9,
        }
    )
    revalidated = adapter.validate_python(dumped)
    assert isinstance(revalidated, UploadProgressEvent)
    assert revalidated == event


def test_typed_subclass_explicit_name():
    """A `name` class argument overrides the class-name-derived event name."""
    event = RenamedEvent(label='x')
    assert event.name == 'sync_renamed'


def test_typed_subclass_to_payload():
    """`to_payload` returns the subclass's own fields, excluding the envelope."""
    assert UploadProgressEvent(done=3, total=9).to_payload() == {'done': 3, 'total': 9}
    assert ProgressEvent(payload={'pct': 50}, tool_call_id='call_1').to_payload() == {'payload': {'pct': 50}}


def test_duplicate_event_name_rejected():
    """Registering a second event class under an existing name fails at class definition."""
    with pytest.raises(TypeError, match="Duplicate custom event name 'upload_progress'"):

        @dataclass(kw_only=True)
        class _ConflictingEvent(CustomEvent, name='upload_progress'):  # pyright: ignore[reportUnusedClass]
            pass


def test_instance_name_override_rejected():
    """A per-instance `name` override on a typed subclass would misroute (de)serialization."""
    with pytest.raises(ValueError, match="serializes under its registered name 'upload_progress'"):
        UploadProgressEvent(done=1, total=2, name='other')


def test_reserved_name_rejected():
    """The family schema's synthetic tags can't be claimed by an event class."""
    with pytest.raises(TypeError, match="Custom event name '__unknown__' is reserved"):

        @dataclass(kw_only=True)
        class ReservedEvent(CustomEvent, name='__unknown__'):  # pyright: ignore[reportUnusedClass]
            pass


def test_envelope_field_shadowing_rejected():
    """Payload fields can't shadow envelope fields like `data`, the untyped payload carrier."""
    with pytest.raises(TypeError, match='reserved for the event envelope: data'):

        @dataclass(kw_only=True)
        class ShadowingEvent(CustomEvent):  # pyright: ignore[reportUnusedClass]
            data: Any = None


def test_slotted_event_class():
    """`@dataclass(slots=True)` recreates the class; the recreated class keeps its registered name."""

    @dataclass(kw_only=True, slots=True)
    class SlottedCustomEvent(CustomEvent):
        value: int

    assert SlottedCustomEvent(value=1).name == 'slotted_custom'


def test_redefined_event_class_replaces_registration():
    """Re-executing the same class definition (notebook cell re-run, reload) replaces, not errors."""

    def define() -> CustomEvent:
        @dataclass(kw_only=True)
        class RedefinedEvent(CustomEvent):
            value: int

        return RedefinedEvent(value=1)

    first, second = define(), define()
    assert type(first) is not type(second)
    assert second.name == 'redefined'
    adapter = pydantic.TypeAdapter[AgentStreamEvent](AgentStreamEvent)
    assert type(adapter.validate_python({'event_kind': 'custom', 'name': 'redefined', 'value': 1})) is type(second)


def test_replay_isolation_keeps_the_canonical_event_class():
    """A durable runtime re-executing app modules doesn't hand the host a class it can't recognize.

    Temporal's workflow sandbox re-runs the module that defines an event class while sharing
    `pydantic_ai` with the host process, so without this the sandbox's copy would take over the
    registry and the host would decode payloads into a class its own `isinstance` checks miss.
    Instances of the copy still serialize exactly, because the family schema canonicalizes them.
    """

    def define() -> Any:
        @dataclass(kw_only=True)
        class IsolatedEvent(CustomEvent):
            value: int

        return IsolatedEvent

    host_cls = define()
    isolated = True
    set_replay_isolation_guard(lambda: isolated)
    try:
        copy_cls = define()
        assert copy_cls is not host_cls
        adapter = pydantic.TypeAdapter[AgentStreamEvent](AgentStreamEvent)

        # The copy serializes as the registered class, without a `PydanticSerializationUnexpectedValue`.
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            wire = adapter.dump_python(copy_cls(value=1), mode='json')
        assert wire == {
            'name': 'isolated',
            'tool_call_id': None,
            'tool_name': None,
            'event_kind': 'custom',
            'value': 1,
        }
        # And validates back into the class the host imported, so `isinstance` holds on both sides.
        assert isinstance(adapter.validate_python(wire), host_cls)

        # Outside the isolated re-execution, a redefinition still replaces the registration.
        isolated = False
        assert isinstance(adapter.validate_python(wire), host_cls)
        replacement = define()
        assert type(pydantic.TypeAdapter[AgentStreamEvent](AgentStreamEvent).validate_python(wire)) is replacement
    finally:
        set_replay_isolation_guard(lambda: False)
        CUSTOM_EVENT_TYPES.pop('isolated', None)


async def test_event_delivered_while_tool_still_running():
    """An emitted event reaches stream consumers while the emitting tool is still executing.

    The tool blocks until the handler has seen the event, so delivery that only happened at the
    tool's completion would deadlock (and trip the timeout) instead of passing.
    """
    received = asyncio.Event()
    agent = Agent(FunctionModel(stream_function=_tool_then_text))

    @agent.tool
    async def progress(ctx: RunContext[Any]) -> str:
        await ctx.emit(ProgressEvent(payload={'done': 1}))
        await asyncio.wait_for(received.wait(), timeout=5)
        return 'ok'

    async def handler(ctx: RunContext[Any], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in events:
            if isinstance(event, CustomEvent) and event.name == 'progress':
                received.set()

    await agent.run('go', event_stream_handler=handler)
    assert received.is_set()


async def test_typed_subclass_emitted_from_tool():
    """A typed subclass emitted from a tool is stamped like any custom event and keeps its type."""
    agent = Agent(FunctionModel(stream_function=_tool_then_text))

    @agent.tool
    async def progress(ctx: RunContext[Any]) -> str:
        await ctx.emit(UploadProgressEvent(done=1, total=2))
        return 'ok'

    events = await _collect_events(agent)
    custom = [event for event in events if isinstance(event, UploadProgressEvent)]
    assert custom == snapshot([UploadProgressEvent(done=1, total=2, tool_call_id='call_1', tool_name='progress')])


async def test_emit_returns_same_typed_instance():
    """`emit` returns the passed instance under its own type, so payload fields typecheck without casts."""
    agent = Agent(FunctionModel(stream_function=_tool_then_text))
    payloads: list[Any] = []

    @agent.tool
    async def progress(ctx: RunContext[Any]) -> str:
        original = ProgressEvent(payload={'pct': 50})
        event = await ctx.emit(original)
        # `event` is typed `ProgressEvent`, not bare `CustomEvent`: accessing the payload field
        # is the pyright-checked claim here.
        payloads.append(event.payload)
        assert event is original
        return 'ok'

    await _collect_events(agent)
    assert payloads == [{'pct': 50}]


def test_undecorated_subclass_rejected():
    """Forgetting `@dataclass` fails loudly at construction and validation instead of dropping payload."""

    class ForgotDecoratorEvent(CustomEvent):
        done: int

    try:
        with pytest.raises(ValueError, match='must be decorated with `@dataclass`'):
            ForgotDecoratorEvent()

        adapter = pydantic.TypeAdapter[AgentStreamEvent](AgentStreamEvent)
        with pytest.raises(pydantic.ValidationError, match='must be decorated with `@dataclass`'):
            adapter.validate_python({'event_kind': 'custom', 'name': 'forgot_decorator', 'done': 3})
    finally:
        # The broken class must not stay registered: adapters built by later tests would embed it.
        del CUSTOM_EVENT_TYPES['forgot_decorator']


def test_empty_derived_name_rejected():
    """A class name that derives an empty event name is rejected at definition, not at first use."""
    with pytest.raises(TypeError, match='derives an empty name'):

        class Event(CustomEvent):  # pyright: ignore[reportUnusedClass]
            pass


def test_post_init_cannot_corrupt_name():
    """A subclass `__post_init__` reassigning the registered `name` is caught by the re-run guard."""

    @dataclass(kw_only=True)
    class CorruptingEvent(CustomEvent):
        def __post_init__(self) -> None:
            self.name = 'corrupted'

    with pytest.raises(ValueError, match="registered name 'corrupting'"):
        CorruptingEvent()


def test_forward_referenced_payload_annotation():
    """An event payload field may reference a class defined later in the module (PEP 649 lazy annotations).

    Below Python 3.14, annotations in a no-`__future__` module evaluate eagerly at class creation, so
    the deferred reference only exists on 3.14+; the classes are built from source to keep this module
    importable everywhere.
    """
    if sys.version_info < (3, 14):
        pytest.skip('deferred (PEP 649) annotations require Python 3.14+')
    namespace: dict[str, Any] = {'CustomEvent': CustomEvent, 'dataclass': dataclass}
    try:
        exec(
            textwrap.dedent(
                """
                @dataclass(kw_only=True)
                class DeferredRefEvent(CustomEvent):
                    ref: DefinedLater

                @dataclass
                class DefinedLater:
                    value: int
                """
            ),
            namespace,
        )
        event = namespace['DeferredRefEvent'](ref=namespace['DefinedLater'](value=1))
        assert event.name == 'deferred_ref'
        assert event.ref.value == 1
    finally:
        # The class's annotation only resolves inside the exec namespace; unregister it so adapters
        # built by later tests don't try (and fail) to build a schema for it.
        CUSTOM_EVENT_TYPES.pop('deferred_ref', None)


async def test_event_stream_position_relative_to_framework_events():
    """A tool-emitted event lands between that call's tool-call and tool-result framework events."""
    agent = Agent(FunctionModel(stream_function=_tool_then_text))

    @agent.tool
    async def progress(ctx: RunContext[Any]) -> str:
        await ctx.emit(ProgressEvent())
        return 'ok'

    events = await _collect_events(agent)
    assert [event.event_kind for event in events] == snapshot(
        [
            'part_start',
            'part_end',
            'function_tool_call',
            'custom',
            'function_tool_result',
            'part_start',
            'final_result',
            'part_end',
        ]
    )


async def test_emission_before_model_retry_is_delivered():
    """An event emitted before the tool raises `ModelRetry` still reaches the stream, once per attempt."""
    attempts = 0
    agent = Agent(FunctionModel(stream_function=_tool_then_text))

    @agent.tool
    async def progress(ctx: RunContext[Any]) -> str:
        nonlocal attempts
        attempts += 1
        await ctx.emit(ProgressEvent(payload=attempts))
        if attempts == 1:
            raise ModelRetry('try again')
        return 'ok'

    events = await _collect_events(agent)
    assert [event.payload for event in events if isinstance(event, ProgressEvent)] == [1, 2]


async def test_emission_before_fatal_tool_error_is_delivered():
    """An event emitted before the tool raises a fatal error reaches consumers before the run fails."""
    agent = Agent(FunctionModel(stream_function=_tool_then_text))

    @agent.tool
    async def progress(ctx: RunContext[Any]) -> str:
        await ctx.emit(ProgressEvent(payload='before-crash'))
        raise RuntimeError('tool crashed')

    events: list[AgentStreamEvent] = []

    async def handler(ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    with pytest.raises(RuntimeError, match='tool crashed'):
        await agent.run('go', event_stream_handler=handler)
    assert [event.payload for event in events if isinstance(event, ProgressEvent)] == ['before-crash']


def test_unknown_event_name_with_payload_degrades():
    """An event dict with an unregistered name and payload fields validates as `UnknownCustomEvent`.

    Nothing is dropped: the payload rides in `data`, and re-serialization re-flattens it so a
    downstream consumer that has the defining module imported recovers the typed event.
    """
    adapter = pydantic.TypeAdapter[AgentStreamEvent](AgentStreamEvent)
    wire = {
        'event_kind': 'custom',
        'name': 'their_typed_event',
        'progress': 0.5,
        'stage': 'fetching',
    }
    with pytest.warns(UserWarning, match="Unknown event name 'their_typed_event'"):
        event = adapter.validate_python(wire)
    assert event == snapshot(UnknownCustomEvent(name='their_typed_event', data={'progress': 0.5, 'stage': 'fetching'}))
    assert isinstance(event, UnknownCustomEvent)
    assert event.to_payload() == {'progress': 0.5, 'stage': 'fetching'}

    redumped = adapter.dump_python(event)
    assert redumped == snapshot(
        {
            'progress': 0.5,
            'stage': 'fetching',
            'name': 'their_typed_event',
            'tool_call_id': None,
            'tool_name': None,
            'event_kind': 'custom',
        }
    )


def test_unknown_event_name_with_nested_data_preserved():
    """A wire event with an unregistered name keeps a nested `data` payload through the unknown envelope."""
    adapter = pydantic.TypeAdapter[AgentStreamEvent](AgentStreamEvent)
    wire = {'event_kind': 'custom', 'name': 'quick_status', 'data': {'stage': 'fetching'}}
    with pytest.warns(UserWarning, match="Unknown event name 'quick_status'"):
        event = adapter.validate_python(wire)
    assert event == snapshot(UnknownCustomEvent(name='quick_status', data={'stage': 'fetching'}))


def test_unknown_event_data_key_collision_round_trips():
    """An unknown wire dict carrying both payload fields and its own `data` key keeps both.

    The envelope nests the original `data` value inside the gathered payload, and re-serialization
    restores the exact original wire dict.
    """
    adapter = pydantic.TypeAdapter[AgentStreamEvent](AgentStreamEvent)
    wire = {'event_kind': 'custom', 'name': 'unseen_collision', 'data': 1, 'extra': 2}
    with pytest.warns(UserWarning, match="Unknown event name 'unseen_collision'"):
        event = adapter.validate_python(wire)
    assert event == snapshot(UnknownCustomEvent(name='unseen_collision', data={'extra': 2, 'data': 1}))
    redumped = adapter.dump_python(event)
    assert {k: v for k, v in redumped.items() if k in wire} == wire


def test_unknown_event_instance_revalidates():
    """An already-constructed unknown instance passes through validation unchanged, without warning."""
    adapter = pydantic.TypeAdapter[AgentStreamEvent](AgentStreamEvent)
    event = UnknownCustomEvent(name='unseen_instance', data={'x': 1})
    assert adapter.validate_python(event) == event


def test_unknown_event_non_mapping_data_round_trips():
    """A non-mapping `data` payload is kept under `data` on serialization instead of being flattened."""
    adapter = pydantic.TypeAdapter[AgentStreamEvent](AgentStreamEvent)
    event = UnknownCustomEvent(name='unseen_scalar', data=5)
    dumped = adapter.dump_python(event)
    assert dumped['data'] == 5
    with pytest.warns(UserWarning, match="Unknown event name 'unseen_scalar'"):
        assert adapter.validate_python(dumped) == event


def test_registration_after_adapter_not_seen():
    """The union is built per `TypeAdapter`: an adapter built before a class was registered degrades
    its events to `UnknownCustomEvent` (the import-order caveat), while a fresh adapter recovers them."""
    old_adapter = pydantic.TypeAdapter[AgentStreamEvent](AgentStreamEvent)

    @dataclass(kw_only=True)
    class LateEvent(CustomEvent, name='late_event'):
        value: int

    fresh_adapter = pydantic.TypeAdapter[AgentStreamEvent](AgentStreamEvent)
    wire = fresh_adapter.dump_python(LateEvent(value=1))
    with pytest.warns(UserWarning, match="Unknown event name 'late_event'"):
        degraded = old_adapter.validate_python(wire)
    assert isinstance(degraded, UnknownCustomEvent)
    assert degraded.data == {'value': 1}

    recovered = fresh_adapter.validate_python(old_adapter.dump_python(degraded))
    assert recovered == LateEvent(value=1)


def test_subclass_post_init_override_keeps_guards():
    """A subclass `__post_init__` that doesn't call `super()` cannot bypass the construction guards."""

    @dataclass(kw_only=True)
    class GuardedCustomEvent(CustomEvent, name='guarded_custom'):
        value: int = 0

        def __post_init__(self) -> None:
            self.value += 1

    with pytest.raises(ValueError, match='serializes under its registered name'):
        GuardedCustomEvent(name='other')
    assert GuardedCustomEvent().value == 1


async def test_iter_completed_or_buffered_plain_list_buffer():
    """A run revived from persisted graph state holds a plain `list` buffer, which can't signal
    appends, so task completion is awaited in plain completion order.

    Unit test: the revived-state path isn't reachable through the public API without a
    persistence backend.
    """
    from pydantic_ai._tool_execution import _iter_completed_or_buffered  # pyright: ignore[reportPrivateUsage]

    async def result() -> int:
        return 1

    items = [item async for item in _iter_completed_or_buffered({asyncio.create_task(result())}, [])]
    assert [item.result() for item in items if isinstance(item, asyncio.Task)] == [1]


async def test_iter_completed_or_buffered_drains_pre_buffered_events():
    """Events already buffered before iteration starts are yielded ahead of any task completion.

    Unit test: through the public API the buffer is drained at stream edges before tool execution
    starts, so the pre-drain branch only sees content when an event lands in the same loop tick.
    """
    from pydantic_ai._run_context import EventStreamBuffer
    from pydantic_ai._tool_execution import _iter_completed_or_buffered  # pyright: ignore[reportPrivateUsage]

    async def result() -> int:
        return 1

    buffer = EventStreamBuffer([ProgressEvent(payload='pre-buffered')])
    items = [item async for item in _iter_completed_or_buffered({asyncio.create_task(result())}, buffer)]
    assert isinstance(items[0], ProgressEvent)
    assert [item.result() for item in items if isinstance(item, asyncio.Task)] == [1]


async def test_output_function_emission_is_delivered():
    """An event emitted from an output function surfaces while output tool calls are processed.

    `end_strategy='exhaustive'` routes output calls through the completion-or-buffer race, the
    same live-delivery path tool calls use.
    """
    from pydantic_ai.output import ToolOutput

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        yield {0: DeltaToolCall(name='final_result', json_args='{"value": "ok"}', tool_call_id='call_out')}

    async def produce(ctx: RunContext[Any], value: str) -> str:
        await ctx.emit(ProgressEvent(payload='from output'))
        return value

    agent: Agent[Any, str] = Agent(
        FunctionModel(stream_function=stream), output_type=ToolOutput(produce), end_strategy='exhaustive'
    )
    events = await _collect_events(agent)
    assert any(isinstance(event, ProgressEvent) and event.payload == 'from output' for event in events)
