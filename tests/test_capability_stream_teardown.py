"""Regression tests for capability event-stream teardown."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from typing import Any, cast

import anyio
import pytest

from pydantic_ai import Agent, RunContext, _utils
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, Hooks, WrapperCapability
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, PartStartEvent, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_graph import End

pytestmark = pytest.mark.anyio


class _TrackingStream(AsyncIterator[AgentStreamEvent]):
    def __init__(
        self,
        name: str,
        closed: list[str],
        stream: AsyncIterable[AgentStreamEvent] | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.name = name
        self.closed = closed
        self.stream = stream
        self.close_error = close_error
        self._iterator = aiter(stream) if stream is not None else None

    def __aiter__(self) -> _TrackingStream:
        return self

    async def __anext__(self) -> AgentStreamEvent:
        if self._iterator is not None:
            return await anext(self._iterator)
        return PartStartEvent(index=0, part=TextPart(content='event'))

    async def aclose(self) -> None:
        self.closed.append(self.name)
        if self.close_error is not None:
            raise self.close_error


@dataclass
class _TrackingCapability(AbstractCapability[Any]):
    name: str
    closed: list[str]
    close_error: BaseException | None = None

    def wrap_run_event_stream(
        self,
        ctx: RunContext[Any],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        return _TrackingStream(self.name, self.closed, stream, self.close_error)


@dataclass
class _BlockingCapability(AbstractCapability[Any]):
    pull_started: anyio.Event
    torn_down: anyio.Event

    def wrap_run_event_stream(
        self,
        ctx: RunContext[Any],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        return _BlockingStream(self.pull_started, self.torn_down)


@dataclass
class _BlockingStream(AsyncIterator[AgentStreamEvent]):
    pull_started: anyio.Event
    torn_down: anyio.Event

    async def __anext__(self) -> AgentStreamEvent:
        self.pull_started.set()
        return cast(AgentStreamEvent, await anyio.sleep_forever())

    async def aclose(self) -> None:
        self.torn_down.set()


@dataclass
class _CloseTrackingCapability(AbstractCapability[Any]):
    torn_down: anyio.Event
    held_streams: list[AsyncIterator[AgentStreamEvent]]
    checkpoint_on_close: bool = False

    def wrap_run_event_stream(
        self,
        ctx: RunContext[Any],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        wrapped = _CloseTrackingStream(aiter(stream), self.torn_down, self.checkpoint_on_close)
        self.held_streams.append(wrapped)
        return wrapped


@dataclass
class _CloseTrackingStream(AsyncIterator[AgentStreamEvent]):
    stream: AsyncIterator[AgentStreamEvent]
    torn_down: anyio.Event
    checkpoint_on_close: bool

    async def __anext__(self) -> AgentStreamEvent:
        return await anext(self.stream)

    async def aclose(self) -> None:
        if self.checkpoint_on_close:
            await anyio.sleep(0)
        self.torn_down.set()


@dataclass
class _PlainIteratorRootCapability(CombinedCapability[Any]):
    torn_down: anyio.Event
    held_streams: list[AsyncIterator[AgentStreamEvent]]

    def wrap_run_event_stream(
        self,
        ctx: RunContext[Any],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        wrapped = _CloseTrackingStream(aiter(stream), self.torn_down, checkpoint_on_close=False)
        self.held_streams.append(wrapped)
        return wrapped


async def _streaming_model(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
    yield 'first'


async def _model_request_stream(agent: Agent[Any, Any]):
    async with agent.iter('hello') as agent_run:
        node = agent_run.next_node
        while not Agent.is_model_request_node(node):
            assert not isinstance(node, End)
            node = await agent_run.next(node)
        async with node.stream(agent_run.ctx) as stream:
            yield stream


def _run_context() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


async def test_combined_closes_every_stream_when_wrapper_close_raises() -> None:
    closed: list[str] = []
    close_error = RuntimeError('retaining close failed')
    source = _TrackingStream('source', closed)
    capability = CombinedCapability(
        [
            _TrackingCapability('outer', closed),
            _TrackingCapability('retaining-raiser', closed, close_error),
            _TrackingCapability('inner', closed),
        ]
    )

    stream = aiter(capability.wrap_run_event_stream(_run_context(), stream=source))
    await anext(stream)
    with pytest.raises(RuntimeError, match='retaining close failed') as exc_info:
        await _utils.aclose_if_supported(stream)

    assert exc_info.value is close_error
    assert closed == ['outer', 'retaining-raiser', 'inner', 'source']


async def test_combined_groups_multiple_close_errors_after_closing_every_stream() -> None:
    closed: list[str] = []
    outer_error = RuntimeError('outer close failed')
    inner_error = ValueError('inner close failed')
    source = _TrackingStream('source', closed)
    capability = CombinedCapability(
        [
            _TrackingCapability('outer', closed, outer_error),
            _TrackingCapability('inner', closed, inner_error),
        ]
    )

    stream = aiter(capability.wrap_run_event_stream(_run_context(), stream=source))
    await anext(stream)
    with pytest.raises(_utils.BaseExceptionGroup) as exc_info:
        await _utils.aclose_if_supported(stream)

    assert exc_info.value.exceptions == (outer_error, inner_error)
    assert closed == ['outer', 'inner', 'source']


async def test_wrapper_capability_closes_original_stream_when_wrapper_retains_it() -> None:
    closed: list[str] = []
    close_error = RuntimeError('wrapped close failed')
    source = _TrackingStream('source', closed)
    capability = WrapperCapability(wrapped=_TrackingCapability('wrapped', closed, close_error))

    stream = aiter(capability.wrap_run_event_stream(_run_context(), stream=source))
    await anext(stream)
    with pytest.raises(RuntimeError, match='wrapped close failed') as exc_info:
        await _utils.aclose_if_supported(stream)

    assert exc_info.value is close_error
    assert closed == ['wrapped', 'source']


async def test_hooks_close_every_stream_when_outer_hook_retains_its_input() -> None:
    closed: list[str] = []
    close_error = RuntimeError('outer close failed')
    source = _TrackingStream('source', closed)
    hooks = Hooks()

    @hooks.on.run_event_stream
    def outer(ctx: RunContext[Any], *, stream: AsyncIterable[AgentStreamEvent]) -> AsyncIterable[AgentStreamEvent]:
        return _TrackingStream('outer', closed, stream, close_error)

    @hooks.on.run_event_stream
    def inner(ctx: RunContext[Any], *, stream: AsyncIterable[AgentStreamEvent]) -> AsyncIterable[AgentStreamEvent]:
        return _TrackingStream('inner', closed, stream)

    stream = aiter(hooks.wrap_run_event_stream(_run_context(), stream=source))
    await anext(stream)
    with pytest.raises(RuntimeError, match='outer close failed') as exc_info:
        await _utils.aclose_if_supported(stream)

    assert exc_info.value is close_error
    assert closed == ['outer', 'inner', 'source']


async def test_agent_stream_closes_custom_async_iterator() -> None:
    torn_down = anyio.Event()
    held_streams: list[AsyncIterator[AgentStreamEvent]] = []
    agent = Agent(FunctionModel(stream_function=_streaming_model))
    # `Agent` normally owns a `CombinedCapability`, whose async-generator wrapper would hide the
    # custom iterator type this regression needs to exercise. Replacing only the root capability
    # keeps the real `AgentStream` construction and graph teardown path intact.
    agent._root_capability = _PlainIteratorRootCapability([], torn_down, held_streams)  # pyright: ignore[reportPrivateUsage]

    with anyio.fail_after(5):
        async for stream in _model_request_stream(agent):
            async for _event in stream:  # pragma: no branch
                break

    assert held_streams
    assert torn_down.is_set()


async def test_agent_stream_cancels_parked_pull_and_closes_capability() -> None:
    pull_started = anyio.Event()
    torn_down = anyio.Event()
    capability = _BlockingCapability(pull_started, torn_down)
    agent = Agent(FunctionModel(stream_function=_streaming_model), capabilities=[capability])

    async def consume(stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for _ in stream:
            pass

    with anyio.fail_after(5):
        async with anyio.create_task_group() as task_group:
            async for stream in _model_request_stream(agent):
                task_group.start_soon(consume, stream)
                await pull_started.wait()

    assert torn_down.is_set()


async def test_agent_stream_close_is_shielded_from_cancellation() -> None:
    torn_down = anyio.Event()
    held_streams: list[AsyncIterator[AgentStreamEvent]] = []
    capability = _CloseTrackingCapability(torn_down, held_streams, checkpoint_on_close=True)
    agent = Agent(FunctionModel(stream_function=_streaming_model), capabilities=[capability])

    with anyio.fail_after(5):
        with anyio.CancelScope() as scope:
            async for stream in _model_request_stream(agent):
                async for _event in stream:  # pragma: no branch
                    scope.cancel()
                    break

    assert held_streams
    assert torn_down.is_set()
