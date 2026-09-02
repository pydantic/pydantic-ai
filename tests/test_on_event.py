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
    WrapperCapability,
    on_event,
)
from pydantic_ai.capabilities._on_event import (
    _OnEventMethod,  # pyright: ignore[reportPrivateUsage]
    collect_on_event_methods,
)
from pydantic_ai.exceptions import UserError
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
from pydantic_ai.models import ModelRequestContext
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
class ThingStartEvent(CapabilityEvent, namespace='on_event_decision', dispatch='immediate'):
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


def test_dispatch_mode_is_inherited_and_validated() -> None:
    @dataclass(kw_only=True)
    class ImmediateBaseEvent(CapabilityEvent, namespace='on_event_inherited', dispatch='immediate'):
        pass

    @dataclass(kw_only=True)
    class ImmediateChildEvent(ImmediateBaseEvent):
        pass

    assert ImmediateChildEvent.event_dispatch == 'immediate'

    with pytest.raises(UserError, match="`dispatch` must be either 'stream' or 'immediate'"):

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


async def test_mutable_decision_event_is_immediate() -> None:
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


async def test_immediate_event_delivered_exactly_once_in_stream_events() -> None:
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


async def test_nested_emit_from_immediate_listener_is_cause_first() -> None:
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


async def test_emitter_reference_reflects_immediate_decisions() -> None:
    """Attribution stamps in place: the emitter's own reference to an immediately dispatched event sees listener decisions."""
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


async def test_reemitted_immediate_event_dispatched_once_per_emit() -> None:
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


async def test_stream_consumers_observe_settled_immediate_events() -> None:
    """A stream consumer never sees an immediately dispatched decision event before its listeners have settled it."""
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


async def test_immediate_listener_exception_propagates_to_emitter() -> None:
    """An immediately dispatched listener's exception surfaces from `emit`, where the emitter can recover."""
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
            raise RuntimeError('immediate listener failed')

    result = await Agent(FunctionModel(stream_function=_tool_then_text), capabilities=[emitter, Boom()]).run('go')
    assert result.output == 'done'
    assert caught == ['immediate listener failed']


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
    with pytest.raises(RuntimeError) as exc_info:  # `UserError` is a `RuntimeError`

        class BadCapability(AbstractCapability[Any]):  # pyright: ignore[reportUnusedClass]
            @on_event(FileReadEvent)
            async def on_event(  # pragma: no cover  # pyright: ignore[reportIncompatibleMethodOverride]
                self, ctx: RunContext[Any], event: FileReadEvent
            ) -> None: ...

    error: BaseException = exc_info.value
    if type(error) is RuntimeError:
        assert error.__cause__ is not None
        error = error.__cause__
    assert isinstance(error, UserError)
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
    assert isinstance(MarkerCapability.traversal, _OnEventMethod)


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


async def test_immediate_event_without_listeners_returns_defaults() -> None:
    """An immediately dispatched decision event with no listeners anywhere returns with its fields untouched."""
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


async def test_hooks_on_event_legacy_replacements_compose_across_capabilities() -> None:
    """The replacement chain spans separate `Hooks` capabilities, not just one capability's callbacks.

    The stream-wrapper implementation this replaced composed capability wrappers by nesting, so a
    second capability transformed what the first produced. Dispatching through `on_event` hands every
    capability the same original event, so without picking up the recorded replacement the last
    capability to run would silently drop the first's. Replacements chain in capability order, the
    order every other hook and `@on_event` observer already runs in.
    """
    first_hooks = Hooks()

    @first_hooks.on.event
    async def replace_in_first(ctx: RunContext[Any], event: AgentStreamEvent) -> AgentStreamEvent | None:
        if isinstance(event, PartStartEvent):
            return ReplacementEvent(payload='first')

    seen_by_second: list[Any] = []
    second_hooks = Hooks()

    @second_hooks.on.event
    async def replace_in_second(ctx: RunContext[Any], event: AgentStreamEvent) -> AgentStreamEvent | None:
        if isinstance(event, ReplacementEvent):
            seen_by_second.append(event.payload)
            return ReplacementEvent(payload=f'{event.payload}+second')

    agent = Agent(
        FunctionModel(simple_model_function, stream_function=simple_stream_function),
        capabilities=[first_hooks, second_hooks],
    )
    with pytest.warns(PydanticAIDeprecationWarning, match='returning a replacement event'):
        async with agent.run_stream_events('hello') as stream:
            events = [event async for event in stream]
    assert 'first' in seen_by_second, "the second capability should see the first capability's replacement"
    assert any(isinstance(event, ReplacementEvent) and event.payload == 'first+second' for event in events), (
        'both replacements should survive to the stream'
    )


async def test_hooks_on_event_legacy_replacement_of_immediate_event_chains_without_stream_rewrite() -> None:
    """Replacing an immediately dispatched decision event chains to later callbacks but never rewrites the stream."""

    @dataclass(kw_only=True)
    class ImmediateDecisionEvent(CapabilityEvent, namespace='capabilities_immediate_replace', dispatch='immediate'):
        cancelled: bool = False

    emitter = Capability[Any](id='emitter')
    emitted: list[ImmediateDecisionEvent] = []

    @emitter.tool
    async def decide(ctx: RunContext[Any]) -> str:
        emitted.append(await ctx.emit(ImmediateDecisionEvent()))
        return 'done'

    hooks = Hooks[Any]()

    @hooks.on.event
    async def replace(ctx: RunContext[Any], event: AgentStreamEvent) -> AgentStreamEvent | None:
        if isinstance(event, ImmediateDecisionEvent):
            return ReplacementEvent(payload='immediate-replaced')

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
    assert seen_after == ['immediate-replaced']
    # The immediately dispatched event still reaches the stream itself; the replacement is not stored.
    assert any(isinstance(event, ImmediateDecisionEvent) for event in events)


# --- `listens_to`: the gate that keeps a capability out of dispatch for events it never wanted ---


def test_listens_to_reports_marked_types_and_bare_markers() -> None:
    """Typed markers narrow to their classes; a bare marker widens the capability to everything."""

    @dataclass
    class Typed(AbstractCapability[Any]):
        @on_event(FileReadEvent)
        async def _files(self, ctx: RunContext[Any], event: FileReadEvent) -> None: ...  # pragma: no cover

    @dataclass
    class Bare(AbstractCapability[Any]):
        @on_event
        async def _everything(self, ctx: RunContext[Any], event: AgentStreamEvent) -> None: ...  # pragma: no cover

    typed, bare, silent = Typed(), Bare(), AbstractCapability[Any]()
    assert typed.listens_to(FileReadEvent(path='a'))
    assert not typed.listens_to(DirectoryListedEvent(path='a'))
    assert bare.listens_to(DirectoryListedEvent(path='a'))
    # Nothing registered at all: `listens_to` agrees with `has_on_event` rather than defaulting to True.
    assert not silent.listens_to(FileReadEvent(path='a'))
    assert not silent.has_on_event


def test_listens_to_widens_for_an_overridden_on_event() -> None:
    """An override's dispatch isn't knowable here, so it opts in to everything unless it says otherwise."""

    @dataclass
    class Dynamic(AbstractCapability[Any]):
        async def on_event(self, ctx: RunContext[Any], *, event: AgentStreamEvent) -> None: ...  # pragma: no cover

    @dataclass
    class NarrowedDynamic(Dynamic):
        def listens_to(self, event: AgentStreamEvent) -> bool:
            return isinstance(event, FileReadEvent)

    assert Dynamic().listens_to(DirectoryListedEvent(path='a'))
    assert NarrowedDynamic().listens_to(FileReadEvent(path='a'))
    assert not NarrowedDynamic().listens_to(DirectoryListedEvent(path='a'))


def test_listens_to_composes_through_containers_and_wrappers() -> None:
    """A container or wrapper reports the union of what it holds, so one broad child widens the whole."""

    @dataclass
    class Typed(AbstractCapability[Any]):
        @on_event(FileReadEvent)
        async def _files(self, ctx: RunContext[Any], event: FileReadEvent) -> None: ...  # pragma: no cover

    combined = CombinedCapability[Any](capabilities=[Typed(), AbstractCapability[Any]()])
    assert combined.listens_to(FileReadEvent(path='a'))
    assert not combined.listens_to(DirectoryListedEvent(path='a'))

    wrapper = WrapperCapability[Any](wrapped=Typed())
    assert wrapper.listens_to(FileReadEvent(path='a'))
    assert not wrapper.listens_to(DirectoryListedEvent(path='a'))

    hooks = Hooks()

    @hooks.on.event(DirectoryListedEvent)
    async def _dirs(ctx: RunContext[Any], event: DirectoryListedEvent) -> None: ...  # pragma: no cover

    assert not hooks.listens_to(FileReadEvent(path='a'))
    assert hooks.listens_to(DirectoryListedEvent(path='a'))
    # One broad member is enough to widen everything above it.
    assert CombinedCapability[Any](capabilities=[Typed(), hooks]).listens_to(DirectoryListedEvent(path='a'))


def test_bare_hook_callback_widens_hooks() -> None:
    """A bare `hooks.on.event` takes every event, so `Hooks` can't narrow."""
    hooks = Hooks()

    @hooks.on.event
    async def _everything(ctx: RunContext[Any], event: AgentStreamEvent) -> None: ...  # pragma: no cover

    assert hooks.listens_to(FileReadEvent(path='a'))


async def test_unlistened_events_never_reach_a_capability() -> None:
    """The point of the gate: a capability isn't woken for event classes it didn't name."""
    dispatched: list[str] = []

    @dataclass
    class Narrow(AbstractCapability[Any]):
        # Declared before the `on_event` override below: inside a class body that name would
        # otherwise resolve to the method rather than to the decorator.
        @on_event(OnEventNoteEvent)
        async def _notes(self, ctx: RunContext[Any], event: OnEventNoteEvent) -> None:
            dispatched.append(f'note:{event.name}')

        async def on_event(self, ctx: RunContext[Any], *, event: AgentStreamEvent) -> None:
            dispatched.append(event.event_kind)
            await super().on_event(ctx, event=event)

        def listens_to(self, event: AgentStreamEvent) -> bool:
            return isinstance(event, CustomEvent)

    @dataclass
    class Emitter(AbstractCapability[Any]):
        @property
        def _emits_app_events(self) -> bool:
            return True

        async def before_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            await ctx.emit(OnEventNoteEvent())
            return request_context

    agent = Agent(FunctionModel(stream_function=simple_stream_function), capabilities=[Emitter(), Narrow()])
    async with agent.run_stream_events('hello') as stream:
        events = [event async for event in stream]

    # Plenty of model events flowed past, but only the custom event was ever handed to `Narrow`.
    assert len({event.event_kind for event in events}) > 1
    assert dispatched == ['custom', 'note:on_event_note']


async def test_hooks_filters_non_matching_callbacks_once_dispatch_is_entered() -> None:
    """A broad callback opens the gate; the per-entry filter still keeps others from firing.

    `listens_to` decides whether `Hooks` is entered at all, but once it is, each registered callback
    is still matched individually — otherwise a bare callback would drag every typed sibling along.
    """
    hooks = Hooks()
    seen: list[str] = []

    @hooks.on.event
    async def everything(ctx: RunContext[Any], event: AgentStreamEvent) -> None:
        seen.append(f'any:{event.event_kind}')

    @hooks.on.event(FileReadEvent)
    async def only_reads(ctx: RunContext[Any], event: FileReadEvent) -> None:
        seen.append(f'read:{event.path}')

    emitter = Capability[Any](id='emitter')

    @emitter.tool
    async def read_file(ctx: RunContext[Any]) -> str:
        await ctx.emit(DirectoryListedEvent(path='dir'))
        await ctx.emit(FileReadEvent(path='file'))
        return 'ok'

    await Agent(FunctionModel(stream_function=_tool_then_text), capabilities=[emitter, hooks]).run('go')

    # The directory event reached the bare callback but was filtered out of the typed one.
    assert 'any:capability' in seen
    assert seen.count('read:file') == 1
    assert not any(entry == 'read:dir' for entry in seen)
