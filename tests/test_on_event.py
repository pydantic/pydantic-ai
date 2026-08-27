"""Tests for capability event listeners."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest

from pydantic_ai import Agent, CapabilityEvent, CustomEvent, RunContext
from pydantic_ai.capabilities import AbstractCapability, Capability, CombinedCapability, on_event
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

pytestmark = pytest.mark.anyio


@dataclass(kw_only=True)
class FileReadEvent(CapabilityEvent, namespace='on_event_files'):
    path: str


@dataclass(kw_only=True)
class DirectoryListedEvent(CapabilityEvent, namespace='on_event_files'):
    path: str


@dataclass(kw_only=True)
class BeforeThingEvent(CapabilityEvent, namespace='on_event_decision'):
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True


@dataclass(kw_only=True)
class NestedEvent(CapabilityEvent, namespace='on_event_nested'):
    value: str


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
    await capability.on_event(ctx, event=CustomEvent(name='custom'))
    await capability.traversal(ctx, event=DirectoryListedEvent(path='b'))

    assert seen == ['traversal:a', 'any:capability', 'any:custom', 'traversal:b']
    assert capability.has_on_event


def test_sync_marker_rejected() -> None:
    with pytest.raises(TypeError, match='only decorate async methods'):

        @on_event(FileReadEvent)  # pyright: ignore[reportArgumentType]
        def invalid(self: Any, ctx: RunContext[Any], event: FileReadEvent) -> None:
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
        await ctx.emit_event(FileReadEvent(path='AGENTS.md'))
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
    observed: list[bool] = []
    emitter = Capability[Any](id='emitter')

    @emitter.tool
    async def read_file(ctx: RunContext[Any]) -> str:
        event = await ctx.emit_event(BeforeThingEvent())
        observed.append(event.cancelled)
        return 'cancelled' if event.cancelled else 'continued'

    @dataclass
    class Canceller(AbstractCapability[Any]):
        @on_event(BeforeThingEvent)
        async def cancel(self, ctx: RunContext[Any], event: BeforeThingEvent) -> None:
            event.cancel()

    await Agent(FunctionModel(stream_function=_tool_then_text), capabilities=[emitter, Canceller()]).run('go')
    assert observed == [True]


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


async def test_emitted_event_delivered_exactly_once_in_stream_events() -> None:
    seen: list[CustomEvent] = []

    @dataclass
    class Emitter(AbstractCapability[Any]):
        _emits_app_events: ClassVar[bool] = True

        async def before_run(self, ctx: RunContext[Any]) -> None:
            await ctx.emit_event(CustomEvent(name='once'))

    @dataclass
    class Listener(AbstractCapability[Any]):
        @on_event(CustomEvent)
        async def record(self, ctx: RunContext[Any], event: CustomEvent) -> None:
            seen.append(event)

    async def model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        async for text in _text_stream():
            yield text

    agent = Agent(FunctionModel(stream_function=model), capabilities=[Emitter(), Listener()])
    async with agent.run_stream_events('go') as stream:
        async for _ in stream:
            pass
    assert [event.name for event in seen] == ['once']


async def _text_stream() -> AsyncIterator[str]:
    yield 'done'


async def test_capability_order_and_reentrant_emission() -> None:
    order: list[str] = []

    @dataclass
    class Listener(AbstractCapability[Any]):
        label: str

        @on_event(CustomEvent)
        async def custom(self, ctx: RunContext[Any], event: CustomEvent) -> None:
            order.append(f'{self.label}:{event.name}')
            if self.label == 'first' and event.name == 'outer':
                await ctx.emit_event(NestedEvent(value='inner'))

        @on_event(NestedEvent)
        async def nested(self, ctx: RunContext[Any], event: NestedEvent) -> None:
            order.append(f'{self.label}:{event.value}')

    root = CombinedCapability([Listener('first', id='first'), Listener('second', id='second')])
    buffer: list[AgentStreamEvent] = []
    ctx = RunContext[Any](
        deps=None,
        model=FunctionModel(stream_function=_tool_then_text),
        usage=None,  # type: ignore[arg-type]
        root_capability=root,
        capabilities={'first': root.capabilities[0], 'second': root.capabilities[1]},
        _event_stream_buffer=buffer,
    )
    await ctx.emit_event(CustomEvent(name='outer'))
    assert order == ['first:outer', 'first:inner', 'second:inner', 'second:outer']


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
    await root.on_event(ctx, event=CustomEvent(name='test'))
    assert seen == ['loaded']


async def test_zero_listeners_does_not_enable_streaming() -> None:
    calls: list[str] = []

    async def function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('function')
        return ModelResponse(parts=[TextPart(content='done')])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        calls.append('stream')
        yield 'done'

    await Agent(FunctionModel(function=function, stream_function=stream), capabilities=[AbstractCapability()]).run('go')
    assert calls == ['function']
