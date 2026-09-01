"""Tests for capability event listeners."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

from pydantic_ai import Agent, CapabilityEvent, CustomEvent, RunContext
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.capabilities import (
    AbstractCapability,
    Capability,
    CombinedCapability,
    Hooks,
    OnEventMethod,
    WrapperCapability,
    on_event,
)
from pydantic_ai.capabilities.on_event import collect_on_event_methods
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from .capability_models import simple_model_function, simple_stream_function

pytestmark = pytest.mark.anyio


@dataclass(kw_only=True)
class FileReadEvent(CapabilityEvent, namespace='on_event_files'):
    path: str


@dataclass(kw_only=True)
class DirectoryListedEvent(CapabilityEvent, namespace='on_event_files'):
    path: str


@dataclass(kw_only=True)
class ThingStartEvent(CapabilityEvent, namespace='on_event_decision', dispatch='inline'):
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True


@dataclass(kw_only=True)
class NestedEvent(CapabilityEvent, namespace='on_event_nested'):
    value: str


@dataclass(kw_only=True)
class OnEventNoteEvent(CustomEvent, name='on_event_note'):
    pass


@dataclass
class MarkerCapability(AbstractCapability[Any]):
    seen: list[str]

    @on_event(FileReadEvent, DirectoryListedEvent)
    async def traversal(self, ctx: RunContext[Any], event: FileReadEvent | DirectoryListedEvent) -> None:
        self.seen.append(f'traversal:{event.path}')

    @on_event
    async def any_event(self, ctx: RunContext[Any], event: AgentStreamEvent) -> None:
        self.seen.append(f'any:{event.event_kind}')


async def test_marker_filtering_order_and_direct_call() -> None:
    seen: list[str] = []
    capability = MarkerCapability(seen)
    ctx = RunContext[Any](
        deps=None,
        model=FunctionModel(stream_function=_tool_then_text),
        usage=None,  # type: ignore[arg-type]
    )

    await capability.on_event(ctx, event=FileReadEvent(path='a'))
    await capability.on_event(ctx, event=OnEventNoteEvent())
    await capability.traversal(ctx, event=DirectoryListedEvent(path='b'))

    assert seen == ['traversal:a', 'any:capability', 'any:custom', 'traversal:b']
    assert capability.has_on_event


def test_sync_marker_rejected() -> None:
    with pytest.raises(TypeError, match='only decorate async methods'):

        @on_event(FileReadEvent)  # pyright: ignore[reportArgumentType]
        def invalid(self: Any, ctx: RunContext[Any], event: FileReadEvent) -> None:
            pass


def test_dispatch_mode_is_inherited_and_validated() -> None:
    @dataclass(kw_only=True)
    class InlineBaseEvent(CapabilityEvent, namespace='on_event_inherited', dispatch='inline'):
        pass

    @dataclass(kw_only=True)
    class InlineChildEvent(InlineBaseEvent):
        pass

    assert InlineChildEvent.event_dispatch == 'inline'

    with pytest.raises(TypeError, match="`dispatch` must be either 'stream' or 'inline'"):

        class InvalidDispatchEvent(  # pyright: ignore[reportGeneralTypeIssues, reportUnusedClass]
            CapabilityEvent,
            namespace='on_event_invalid',
            dispatch='later',  # pyright: ignore[reportArgumentType]
        ):
            pass


# Static negative case: `@on_event(FileReadEvent)` rejects an event parameter typed as
# `DirectoryListedEvent` with `reportArgumentType` under pyright.


def _has_tool_return(messages: list[ModelMessage]) -> bool:
    return any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)


async def _tool_then_text(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
    if not _has_tool_return(messages):
        yield {0: DeltaToolCall(name='read_file', json_args='{}', tool_call_id='call_1')}
    else:
        yield 'do'
        yield 'ne'


async def test_listener_enqueue_reaches_next_model_request() -> None:
    seen_context = False

    async def model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        nonlocal seen_context
        if not _has_tool_return(messages):
            yield {0: DeltaToolCall(name='read_file', json_args='{}', tool_call_id='call_1')}
        else:
            seen_context = any(
                getattr(part, 'content', None) == 'AGENTS.md context' for message in messages for part in message.parts
            )
            yield 'done'

    files = Capability[Any](id='files')

    @files.tool
    async def read_file(ctx: RunContext[Any]) -> str:
        await ctx.emit(FileReadEvent(path='AGENTS.md'))
        return 'contents'

    @dataclass
    class RepoContext(AbstractCapability[Any]):
        @on_event(FileReadEvent)
        async def add_context(self, ctx: RunContext[Any], event: FileReadEvent) -> None:
            ctx.enqueue('AGENTS.md context')

    result = await Agent(FunctionModel(stream_function=model), capabilities=[files, RepoContext()]).run('go')
    assert result.output == 'done'
    assert seen_context


async def test_mutable_decision_event_is_inline() -> None:
    observed: list[tuple[bool, bool]] = []
    emitter = Capability[Any](id='emitter')

    @emitter.tool
    async def read_file(ctx: RunContext[Any]) -> str:
        stream_event = await ctx.emit(FileReadEvent(path='later'))
        event = await ctx.emit(ThingStartEvent())
        observed.append((stream_event.path == 'changed', event.cancelled))
        return 'cancelled' if event.cancelled else 'continued'

    @dataclass
    class Canceller(AbstractCapability[Any]):
        @on_event(FileReadEvent)
        async def change(self, ctx: RunContext[Any], event: FileReadEvent) -> None:
            event.path = 'changed'

        @on_event(ThingStartEvent)
        async def cancel(self, ctx: RunContext[Any], event: ThingStartEvent) -> None:
            event.cancel()

    await Agent(FunctionModel(stream_function=_tool_then_text), capabilities=[emitter, Canceller()]).run('go')
    assert observed == [(False, True)]


async def test_emitted_event_dispatches_before_tool_result() -> None:
    order: list[str] = []
    files = Capability[Any](id='files')

    @files.tool
    async def read_file(ctx: RunContext[Any]) -> str:
        await ctx.emit(FileReadEvent(path='AGENTS.md'))
        return 'contents'

    @dataclass
    class Listener(AbstractCapability[Any]):
        @on_event(FileReadEvent, FunctionToolResultEvent)
        async def record(self, ctx: RunContext[Any], event: FileReadEvent | FunctionToolResultEvent) -> None:
            order.append(type(event).__name__)

    await Agent(FunctionModel(stream_function=_tool_then_text), capabilities=[files, Listener()]).run('go')
    assert order == ['FileReadEvent', 'FunctionToolResultEvent']


async def test_framework_events_auto_enable_streaming() -> None:
    seen: list[AgentStreamEvent] = []

    @dataclass
    class Listener(AbstractCapability[Any]):
        @on_event
        async def record(self, ctx: RunContext[Any], event: AgentStreamEvent) -> None:
            seen.append(event)

    def read_file() -> str:
        return 'contents'

    await Agent(FunctionModel(stream_function=_tool_then_text), capabilities=[Listener()], tools=[read_file]).run('go')
    assert any(isinstance(event, FunctionToolCallEvent) for event in seen)
    assert any(isinstance(event, FunctionToolResultEvent) for event in seen)
    assert any(isinstance(event, PartStartEvent) for event in seen)
    assert any(isinstance(event, PartDeltaEvent) for event in seen)


async def test_inline_event_delivered_exactly_once_in_stream_events() -> None:
    seen: list[ThingStartEvent] = []

    @dataclass
    class Emitter(AbstractCapability[Any]):
        async def before_run(self, ctx: RunContext[Any]) -> None:
            await ctx.emit(ThingStartEvent())

    @dataclass
    class Listener(AbstractCapability[Any]):
        @on_event(ThingStartEvent)
        async def record(self, ctx: RunContext[Any], event: ThingStartEvent) -> None:
            seen.append(event)

    agent = Agent(FunctionModel(stream_function=_text_stream), capabilities=[Emitter(), Listener()])
    async with agent.run_stream_events('go') as stream:
        async for _ in stream:
            pass
    assert len(seen) == 1


async def _text_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield 'done'


async def test_nested_emit_from_inline_listener_is_cause_first() -> None:
    listener_log: list[str] = []
    stream_log: list[str] = []

    @dataclass
    class Listener(AbstractCapability[Any]):
        @on_event(ThingStartEvent)
        async def cause(self, ctx: RunContext[Any], event: ThingStartEvent) -> None:
            listener_log.append('cause')
            await ctx.emit(NestedEvent(value='effect'))

        @on_event(NestedEvent)
        async def nested(self, ctx: RunContext[Any], event: NestedEvent) -> None:
            listener_log.append(event.value)

    @dataclass
    class Emitter(AbstractCapability[Any]):
        async def before_run(self, ctx: RunContext[Any]) -> None:
            await ctx.emit(ThingStartEvent())

    agent = Agent(FunctionModel(stream_function=_text_stream), capabilities=[Emitter(), Listener()])
    async with agent.run_stream_events('go') as stream:
        async for event in stream:
            if isinstance(event, ThingStartEvent):
                stream_log.append('cause')
            elif isinstance(event, NestedEvent):
                stream_log.append(event.value)

    assert listener_log == ['cause', 'effect']
    assert stream_log == ['cause', 'effect']


async def test_deferred_listener_only_runs_when_loaded() -> None:
    seen: list[str] = []

    @dataclass
    class Listener(AbstractCapability[Any]):
        label: str

        @on_event(CustomEvent)
        async def record(self, ctx: RunContext[Any], event: CustomEvent) -> None:
            seen.append(self.label)

    unloaded = Listener('unloaded', id='unloaded', defer_loading=True)
    loaded = Listener('loaded', id='loaded', defer_loading=True)
    root = CombinedCapability([unloaded, loaded])
    ctx = RunContext[Any](
        deps=None,
        model=FunctionModel(stream_function=_tool_then_text),
        usage=None,  # type: ignore[arg-type]
        root_capability=root,
        capabilities={'unloaded': unloaded, 'loaded': loaded},
        loaded_capability_ids={'loaded'},
    )
    await root.on_event(ctx, event=OnEventNoteEvent())
    assert seen == ['loaded']


@dataclass
class RecorderCapability(AbstractCapability[Any]):
    order: list[str]
    label: str

    @on_event(ThingStartEvent)
    async def record(self, ctx: RunContext[Any], event: ThingStartEvent) -> None:
        self.order.append(self.label)


@dataclass
class ThingStartEmitter(AbstractCapability[Any]):
    async def before_run(self, ctx: RunContext[Any]) -> None:
        await ctx.emit(ThingStartEvent())


async def test_listener_order_follows_capability_composition_order() -> None:
    """Listeners across composed capabilities run in composition order; swapping the composition swaps the order."""
    order: list[str] = []
    first = RecorderCapability(order, 'first', id='first')
    second = RecorderCapability(order, 'second', id='second')

    await Agent(FunctionModel(stream_function=_text_stream), capabilities=[ThingStartEmitter(), first, second]).run(
        'go'
    )
    assert order == ['first', 'second']

    order.clear()
    await Agent(FunctionModel(stream_function=_text_stream), capabilities=[ThingStartEmitter(), second, first]).run(
        'go'
    )
    assert order == ['second', 'first']


async def test_combined_subclass_marked_listeners_run_after_children() -> None:
    """A `CombinedCapability` subclass's own marked listeners dispatch after its children's.

    Dispatched on the container directly: `Agent(capabilities=[...])` splats a nested combined
    container into its leaves during normalization, so a subclass's own listener surface is only
    reached when dispatch starts at the container itself (a custom root or manual dispatch).
    """
    order: list[str] = []

    @dataclass
    class Child(AbstractCapability[Any]):
        @on_event(OnEventNoteEvent)
        async def record(self, ctx: RunContext[Any], event: OnEventNoteEvent) -> None:
            order.append('child')

    @dataclass
    class Team(CombinedCapability[Any]):
        @on_event(OnEventNoteEvent)
        async def record_team(self, ctx: RunContext[Any], event: OnEventNoteEvent) -> None:
            order.append('team')

    child = Child(id='child')
    team = Team([child])
    ctx = RunContext[Any](
        deps=None,
        model=FunctionModel(stream_function=_tool_then_text),
        usage=None,  # type: ignore[arg-type]
        root_capability=team,
        capabilities={'child': child},
    )
    await team.on_event(ctx, event=OnEventNoteEvent())
    assert order == ['child', 'team']


async def test_emitter_reference_reflects_inline_decisions() -> None:
    """Attribution stamps in place: the emitter's own reference to an inline event sees listener decisions."""
    alias_saw: list[bool] = []
    emitter = Capability[Any](id='emitter')

    @emitter.tool
    async def read_file(ctx: RunContext[Any]) -> str:
        event = ThingStartEvent()
        returned = await ctx.emit(event)
        alias_saw.append(event.cancelled)
        assert returned is event
        return 'ok'

    @dataclass
    class Canceller(AbstractCapability[Any]):
        @on_event(ThingStartEvent)
        async def cancel(self, ctx: RunContext[Any], event: ThingStartEvent) -> None:
            event.cancel()

    await Agent(FunctionModel(stream_function=_tool_then_text), capabilities=[emitter, Canceller()]).run('go')
    assert alias_saw == [True]


async def test_reemitted_inline_event_dispatched_once_per_emit() -> None:
    """Re-emitting the instance `emit` returned delivers to listeners exactly once per emission."""
    count = 0
    emitter = Capability[Any](id='emitter')

    @emitter.tool
    async def read_file(ctx: RunContext[Any]) -> str:
        event = await ctx.emit(ThingStartEvent())
        await ctx.emit(event)
        return 'ok'

    @dataclass
    class Counter(AbstractCapability[Any]):
        @on_event(ThingStartEvent)
        async def count_up(self, ctx: RunContext[Any], event: ThingStartEvent) -> None:
            nonlocal count
            count += 1

    await Agent(FunctionModel(stream_function=_tool_then_text), capabilities=[emitter, Counter()]).run('go')
    assert count == 2


async def test_stream_consumers_observe_settled_inline_events() -> None:
    """A stream consumer never sees an inline decision event before its listeners have settled it."""
    emitter = Capability[Any](id='emitter')

    @emitter.tool
    async def read_file(ctx: RunContext[Any]) -> str:
        await ctx.emit(ThingStartEvent())
        return 'ok'

    @dataclass
    class SlowCanceller(AbstractCapability[Any]):
        @on_event(ThingStartEvent)
        async def cancel(self, ctx: RunContext[Any], event: ThingStartEvent) -> None:
            # Yield the event loop first so a concurrent stream consumer could drain the buffered
            # event mid-dispatch; without settlement it would observe `cancelled=False`.
            await asyncio.sleep(0.02)
            event.cancel()

    observed: list[bool] = []
    agent = Agent(FunctionModel(stream_function=_tool_then_text), capabilities=[emitter, SlowCanceller()])
    async with agent.run_stream_events('go') as stream:
        async for event in stream:
            if isinstance(event, ThingStartEvent):
                observed.append(event.cancelled)
    assert observed == [True]


async def test_stream_listener_exception_fails_run() -> None:
    """Listeners are fail-closed: an exception from a stream-dispatched listener fails the run."""
    emitter = Capability[Any](id='emitter')

    @emitter.tool
    async def read_file(ctx: RunContext[Any]) -> str:
        await ctx.emit(FileReadEvent(path='a'))
        return 'ok'

    @dataclass
    class Boom(AbstractCapability[Any]):
        @on_event(FileReadEvent)
        async def boom(self, ctx: RunContext[Any], event: FileReadEvent) -> None:
            raise RuntimeError('listener failed')

    agent = Agent(FunctionModel(stream_function=_tool_then_text), capabilities=[emitter, Boom()])
    with pytest.raises(RuntimeError, match='listener failed'):
        await agent.run('go')


async def test_inline_listener_exception_propagates_to_emitter() -> None:
    """An inline listener's exception surfaces from `emit`, where the emitter can recover."""
    caught: list[str] = []
    emitter = Capability[Any](id='emitter')

    @emitter.tool
    async def read_file(ctx: RunContext[Any]) -> str:
        try:
            await ctx.emit(ThingStartEvent())
        except RuntimeError as e:
            caught.append(str(e))
        return 'ok'

    @dataclass
    class Boom(AbstractCapability[Any]):
        @on_event(ThingStartEvent)
        async def boom(self, ctx: RunContext[Any], event: ThingStartEvent) -> None:
            raise RuntimeError('inline listener failed')

    result = await Agent(FunctionModel(stream_function=_tool_then_text), capabilities=[emitter, Boom()]).run('go')
    assert result.output == 'done'
    assert caught == ['inline listener failed']


async def test_zero_listeners_does_not_enable_streaming() -> None:
    calls: list[str] = []

    async def function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('function')
        return ModelResponse(parts=[TextPart(content='done')])

    # Asserted never called.
    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:  # pragma: no cover
        calls.append('stream')
        yield 'done'

    await Agent(FunctionModel(function=function, stream_function=stream), capabilities=[AbstractCapability()]).run('go')
    assert calls == ['function']


def test_marked_method_named_on_event_rejected() -> None:
    """`on_event` is the dispatcher that invokes the marked listeners; a marker can't replace it."""
    # Python < 3.12 wraps exceptions raised by `__set_name__` in a `RuntimeError`.
    with pytest.raises((TypeError, RuntimeError)) as exc_info:

        class BadCapability(AbstractCapability[Any]):  # pyright: ignore[reportUnusedClass]
            @on_event(FileReadEvent)
            async def on_event(  # pragma: no cover  # pyright: ignore[reportIncompatibleMethodOverride]
                self, ctx: RunContext[Any], event: FileReadEvent
            ) -> None: ...

    error: BaseException = exc_info.value
    if isinstance(error, RuntimeError):
        assert error.__cause__ is not None
        error = error.__cause__
    assert isinstance(error, TypeError)
    assert "cannot decorate a method named 'on_event'" in str(error)


async def test_combined_capability_subclass_own_listeners() -> None:
    """A `CombinedCapability` subclass's own marked listeners dispatch after its children's.

    Direct dispatch: combining such a subclass under another `CombinedCapability` splats its
    children into the outer container, so its own listeners matter when it is the dispatch root.
    """
    received: list[str] = []

    class Child(AbstractCapability[Any]):
        @on_event(FileReadEvent)
        async def _on_read(self, ctx: RunContext[Any], event: FileReadEvent) -> None:
            received.append('child')

    class Harness(CombinedCapability[Any]):
        @on_event(FileReadEvent)
        async def _on_read(self, ctx: RunContext[Any], event: FileReadEvent) -> None:
            received.append('combined')

    harness = Harness(capabilities=[Child()])
    assert harness.has_on_event
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id='run')
    await harness.on_event(ctx, event=FileReadEvent(path='a.txt'))
    assert received == ['child', 'combined']


def test_combined_capability_subclass_listeners_alone_enable_dispatch() -> None:
    """A subclass's own listeners count toward `has_on_event` even with no listening children."""

    class Harness(CombinedCapability[Any]):
        @on_event(FileReadEvent)
        # Never dispatched.
        async def _on_read(self, ctx: RunContext[Any], event: FileReadEvent) -> None: ...  # pragma: no cover

    assert Harness(capabilities=[AbstractCapability()]).has_on_event
    assert not CombinedCapability[Any](capabilities=[AbstractCapability()]).has_on_event


async def test_hooks_subclass_marked_listeners_dispatch() -> None:
    """A `Hooks` subclass's own marked listeners are detected and dispatched.

    `Hooks.has_on_event` reports registered hook functions; it must not mask the base
    capability surface a subclass uses.
    """
    received: list[FileReadEvent] = []

    class ListeningHooks(Hooks[Any]):
        @on_event(FileReadEvent)
        async def _on_read(self, ctx: RunContext[Any], event: FileReadEvent) -> None:
            received.append(event)

    hooks = ListeningHooks()
    assert hooks.has_on_event

    files = Capability[Any](id='files')

    @files.tool
    async def read_file(ctx: RunContext[Any]) -> str:
        await ctx.emit(FileReadEvent(path='hook.txt'))
        return 'contents'

    await Agent(FunctionModel(stream_function=_tool_then_text), capabilities=[files, hooks]).run('go')
    assert received == [
        FileReadEvent(path='hook.txt', capability_id='files', tool_call_id='call_1', tool_name='read_file')
    ]


def test_marker_class_access_returns_descriptor() -> None:
    assert isinstance(MarkerCapability.traversal, OnEventMethod)


def test_bare_sync_marker_rejected() -> None:
    with pytest.raises(TypeError, match='only decorate async methods'):

        @on_event  # pyright: ignore[reportArgumentType, reportCallIssue, reportUntypedFunctionDecorator]
        def invalid(self: Any, ctx: RunContext[Any], event: AgentStreamEvent) -> None:
            pass


def test_subclass_override_unmarks_inherited_listener() -> None:
    """A subclass overriding a marked method with a plain method removes the marker."""

    @dataclass
    class Quiet(MarkerCapability):
        async def traversal(  # pragma: no cover  # pyright: ignore[reportIncompatibleVariableOverride]
            self, ctx: RunContext[Any], event: FileReadEvent | DirectoryListedEvent
        ) -> None: ...

    assert [method.func.__name__ for method in collect_on_event_methods(Quiet)] == ['any_event']


async def test_wrapper_delegates_on_event() -> None:
    """A wrapped capability's listeners still receive events."""
    received: list[FileReadEvent] = []

    @dataclass
    class Listener(AbstractCapability[Any]):
        @on_event(FileReadEvent)
        async def _on_read(self, ctx: RunContext[Any], event: FileReadEvent) -> None:
            received.append(event)

    files = Capability[Any](id='files')

    @files.tool
    async def read_file(ctx: RunContext[Any]) -> str:
        await ctx.emit(FileReadEvent(path='wrapped.txt'))
        return 'contents'

    wrapper = WrapperCapability(wrapped=Listener())
    assert wrapper.has_on_event
    await Agent(FunctionModel(stream_function=_tool_then_text), capabilities=[files, wrapper]).run('go')
    assert [event.path for event in received] == ['wrapped.txt']


async def test_inline_event_without_listeners_returns_defaults() -> None:
    """An inline decision event with no listeners anywhere returns with its fields untouched."""
    emitter = Capability[Any](id='emitter')
    outcomes: list[bool] = []

    @emitter.tool
    async def read_file(ctx: RunContext[Any]) -> str:
        event = await ctx.emit(ThingStartEvent())
        outcomes.append(event.cancelled)
        return 'done'

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if _has_tool_return(messages):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='read_file', args='{}', tool_call_id='call_1')])

    await Agent(FunctionModel(function=model), capabilities=[emitter]).run('go')
    assert outcomes == [False]


async def test_wrapper_subclass_markers_dispatch_over_non_listening_wrapped() -> None:
    """A wrapper subclass's own marked listeners dispatch even when the wrapped capability has none."""
    received: list[str] = []

    @dataclass
    class ListeningWrapper(WrapperCapability[Any]):
        @on_event(FileReadEvent)
        async def _on_read(self, ctx: RunContext[Any], event: FileReadEvent) -> None:
            received.append(event.path)

    files = Capability[Any](id='files')

    @files.tool
    async def read_file(ctx: RunContext[Any]) -> str:
        await ctx.emit(FileReadEvent(path='wrapped.txt'))
        return 'contents'

    wrapper = ListeningWrapper(wrapped=AbstractCapability())
    assert wrapper.has_on_event
    await Agent(FunctionModel(stream_function=_tool_then_text), capabilities=[files, wrapper]).run('go')
    assert received == ['wrapped.txt']


# --- Legacy `hooks.on.event` replacement semantics (deprecated toward `hooks.on.run_event_stream`) ---


@dataclass(kw_only=True)
class ReplacementEvent(CustomEvent, name='replacement'):
    payload: Any = None


async def test_hooks_on_event_legacy_replacement_warns_and_transforms() -> None:
    hooks = Hooks()

    @hooks.on.event
    async def replace(ctx: RunContext[Any], event: AgentStreamEvent) -> AgentStreamEvent | None:
        if isinstance(event, PartStartEvent):
            return ReplacementEvent()

    events: list[Any] = []
    agent = Agent(
        FunctionModel(simple_model_function, stream_function=simple_stream_function),
        capabilities=[hooks],
    )
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match='returning a replacement event from `hooks.on.event` is deprecated; '
        'use `hooks.on.run_event_stream` to transform the stream',
    ):
        async with agent.run_stream_events('hello') as stream:
            events = [event async for event in stream]
    assert any(isinstance(event, CustomEvent) and event.name == 'replacement' for event in events)


async def test_hooks_on_event_legacy_replacements_compose() -> None:
    """A second replacing callback sees the first's replacement, and the last replacement wins."""
    hooks = Hooks()

    @hooks.on.event
    async def replace_first(ctx: RunContext[Any], event: AgentStreamEvent) -> AgentStreamEvent | None:
        if isinstance(event, PartStartEvent):
            return ReplacementEvent(payload='first')

    seen_by_second: list[Any] = []

    @hooks.on.event
    async def replace_second(ctx: RunContext[Any], event: AgentStreamEvent) -> AgentStreamEvent | None:
        if isinstance(event, ReplacementEvent):
            seen_by_second.append(event.payload)
            return ReplacementEvent(payload=f'{event.payload}+second')

    seen_by_third: list[Any] = []

    @hooks.on.event
    async def observe_third(ctx: RunContext[Any], event: AgentStreamEvent) -> None:
        if isinstance(event, ReplacementEvent):
            seen_by_third.append(event.payload)

    agent = Agent(
        FunctionModel(simple_model_function, stream_function=simple_stream_function),
        capabilities=[hooks],
    )
    with pytest.warns(PydanticAIDeprecationWarning, match='returning a replacement event'):
        async with agent.run_stream_events('hello') as stream:
            events = [event async for event in stream]
    assert 'first' in seen_by_second
    assert 'first+second' in seen_by_third
    assert any(isinstance(event, ReplacementEvent) and event.payload == 'first+second' for event in events), (
        'the composed replacement should reach the stream'
    )


async def test_hooks_on_event_legacy_replacement_of_inline_event_chains_without_stream_rewrite() -> None:
    """Replacing an inline decision event chains to later callbacks but never rewrites the stream."""

    @dataclass(kw_only=True)
    class InlineDecisionEvent(CapabilityEvent, namespace='capabilities_inline_replace', dispatch='inline'):
        cancelled: bool = False

    emitter = Capability[Any](id='emitter')
    emitted: list[InlineDecisionEvent] = []

    @emitter.tool
    async def decide(ctx: RunContext[Any]) -> str:
        emitted.append(await ctx.emit(InlineDecisionEvent()))
        return 'done'

    hooks = Hooks[Any]()

    @hooks.on.event
    async def replace(ctx: RunContext[Any], event: AgentStreamEvent) -> AgentStreamEvent | None:
        if isinstance(event, InlineDecisionEvent):
            return ReplacementEvent(payload='inline-replaced')

    seen_after: list[str] = []

    @hooks.on.event
    async def observe(ctx: RunContext[Any], event: AgentStreamEvent) -> None:
        if isinstance(event, ReplacementEvent):
            seen_after.append(str(event.payload))

    async def call_decide(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            yield 'done'
        else:
            yield {0: DeltaToolCall(name='decide', json_args='{}', tool_call_id='call_1')}

    agent = Agent(FunctionModel(stream_function=call_decide), capabilities=[emitter, hooks])
    with pytest.warns(PydanticAIDeprecationWarning, match='returning a replacement event'):
        async with agent.run_stream_events('hello') as stream:
            events = [event async for event in stream]
    assert seen_after == ['inline-replaced']
    # The inline event still reaches the stream itself; the replacement is not stored.
    assert any(isinstance(event, InlineDecisionEvent) for event in events)
