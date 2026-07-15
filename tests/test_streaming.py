from __future__ import annotations as _annotations

import asyncio
import contextvars
import datetime
import gc
import json
import re
import sys
import threading
import warnings
import weakref
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Awaitable, Callable, Generator
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime as _datetime, timezone
from types import TracebackType
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel
from pydantic_core import ErrorDetails

from pydantic_ai import (
    Agent,
    AgentRunResult,
    AgentRunResultEvent,
    AgentStreamEvent,
    ExternalToolset,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ImageUrl,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelRequestContext,
    ModelResponse,
    ModelResponsePart,
    NativeToolCallPart,
    NativeToolReturnPart,
    OutputToolCallEvent,
    OutputToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    RunContext,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UnexpectedModelBehavior,
    UserError,
    UserPromptPart,
    _utils,
    capture_run_messages,
    models,
)
from pydantic_ai._agent_graph import GraphAgentState
from pydantic_ai._output import TextOutputProcessor, TextOutputSchema
from pydantic_ai._sync_stream import (
    SyncStreamBridge,
    _finalize_loop,  # pyright: ignore[reportPrivateUsage]
    _request_exit,  # pyright: ignore[reportPrivateUsage]
    _run_task_to_completion,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.agent import AgentRun
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, WrapModelRequestHandler
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry
from pydantic_ai.models.function import AgentInfo, BuiltinToolCallsReturns, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel, TestStreamedResponse as ModelTestStreamedResponse
from pydantic_ai.models.wrapper import CompletedStreamedResponse
from pydantic_ai.output import NativeOutput, PromptedOutput, TextOutput, ToolOutput
from pydantic_ai.result import AgentStream, FinalResult, RunUsage, StreamedRunResult, StreamedRunResultSync
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDefinition, ToolDenied
from pydantic_ai.usage import RequestUsage
from pydantic_graph import End

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, IsInt, IsNow, IsStr, message_part

pytestmark = pytest.mark.anyio


class Foo(BaseModel):
    a: int
    b: str


async def test_streamed_text_response():
    m = TestModel()

    test_agent = Agent(m)
    assert test_agent.name is None

    @test_agent.tool_plain
    async def ret_a(x: str) -> str:
        return f'{x}-apple'

    async with test_agent.run_stream('Hello') as result:
        assert test_agent.name == 'test_agent'
        assert isinstance(result.run_id, str)
        assert not result.is_complete
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr())],
                    usage=RequestUsage(input_tokens=51),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                    provider_name='test',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='ret_a', content='a-apple', timestamp=IsNow(tz=timezone.utc), tool_call_id=IsStr()
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )
        assert result.usage == snapshot(
            RunUsage(
                requests=2,
                input_tokens=103,
                output_tokens=5,
                tool_calls=1,
            )
        )
        response = await result.get_output()
        assert response == snapshot('{"ret_a":"a-apple"}')
        assert result.is_complete
        assert result.timestamp == IsNow(tz=timezone.utc)
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr())],
                    usage=RequestUsage(input_tokens=51),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                    provider_name='test',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='ret_a', content='a-apple', timestamp=IsNow(tz=timezone.utc), tool_call_id=IsStr()
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"ret_a":"a-apple"}')],
                    usage=RequestUsage(input_tokens=52, output_tokens=11),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                    provider_name='test',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )
        assert result.usage == snapshot(
            RunUsage(
                requests=2,
                input_tokens=103,
                output_tokens=11,
                tool_calls=1,
            )
        )


def test_streamed_text_sync_response():
    m = TestModel()

    test_agent = Agent(m)
    assert test_agent.name is None

    @test_agent.tool_plain
    async def ret_a(x: str) -> str:
        return f'{x}-apple'

    result = test_agent.run_stream_sync('Hello')
    assert test_agent.name == 'test_agent'
    assert isinstance(result.run_id, str)
    assert not result.is_complete
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=51),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='test',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='ret_a', content='a-apple', timestamp=IsNow(tz=timezone.utc), tool_call_id=IsStr()
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result.new_messages() == result.all_messages()
    assert result.usage == snapshot(
        RunUsage(
            requests=2,
            input_tokens=103,
            output_tokens=5,
            tool_calls=1,
        )
    )
    response = result.get_output()
    assert response == snapshot('{"ret_a":"a-apple"}')
    assert result.is_complete
    assert result.timestamp == IsNow(tz=timezone.utc)
    assert result.response == snapshot(
        ModelResponse(
            parts=[TextPart(content='{"ret_a":"a-apple"}')],
            usage=RequestUsage(input_tokens=52, output_tokens=11),
            model_name='test',
            timestamp=IsDatetime(),
            provider_name='test',
        )
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=51),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='test',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='ret_a', content='a-apple', timestamp=IsNow(tz=timezone.utc), tool_call_id=IsStr()
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"ret_a":"a-apple"}')],
                usage=RequestUsage(input_tokens=52, output_tokens=11),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='test',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result.usage == snapshot(
        RunUsage(
            requests=2,
            input_tokens=103,
            output_tokens=11,
            tool_calls=1,
        )
    )


async def test_run_stream_sync_rejects_running_event_loop():
    """`run_stream_sync` drives the caller's event loop, so it must refuse to run inside an existing one.

    This is an in-process loop-state guard that a provider cassette cannot exercise.
    """
    agent = Agent(TestModel())
    with pytest.raises(RuntimeError, match=r'from within an async context or a running event loop; use `run_stream`'):
        agent.run_stream_sync('Hello')


def test_run_stream_sync_works_with_disabled_threads():
    """`run_stream_sync` does not need a worker thread.

    VCR cannot observe whether the synchronous bridge starts a thread.
    """
    agent = Agent(TestModel())
    with _utils.disable_threads():
        with agent.run_stream_sync('Hello') as result:
            assert result.get_output()


def _interrupt_next_loop_run(
    bridge: SyncStreamBridge[Any], monkeypatch: pytest.MonkeyPatch
) -> Callable[[Awaitable[Any]], Any]:
    """Raise `KeyboardInterrupt` from the next blocking event-loop drive."""
    loop = bridge._loop  # pyright: ignore[reportPrivateUsage]
    original_run_until_complete = loop.run_until_complete
    calls = 0

    def interrupt_first_run(awaitable: Awaitable[Any]) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt
        return original_run_until_complete(awaitable)

    monkeypatch.setattr(loop, 'run_until_complete', interrupt_first_run)
    return original_run_until_complete


def test_run_stream_sync_tears_down_on_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch):
    """A Ctrl-C while driving the event loop cancels the run instead of leaking tasks or sockets (#5975).

    VCR cannot reproduce the required in-process interrupt timing or inspect pending tasks.
    """
    agent = Agent(TestModel())
    result = agent.run_stream_sync('Hello')
    bridge = result._bridge  # pyright: ignore[reportPrivateUsage]
    assert bridge._finalizer.alive  # pyright: ignore[reportPrivateUsage]

    _interrupt_next_loop_run(bridge, monkeypatch)

    # Enter the `with` block too, so its `__exit__` also calls `shutdown()`. The interrupt teardown
    # already ran it once, so this exercises the idempotent (already-disarmed) shutdown path.
    with pytest.raises(KeyboardInterrupt):
        with result:
            result.get_output()

    # The run was torn down as part of handling the interrupt, leaving no pending owner task for GC.
    assert not bridge._finalizer.alive  # pyright: ignore[reportPrivateUsage]
    assert bridge._owner_task.done()  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(RuntimeError, match='already closed'):
        bridge.call(lambda: None)


def test_sync_stream_bridge_call_propagates_keyboard_interrupt():
    """An interrupt raised by a bridge call propagates after the context manager exits.

    VCR cannot inject a synchronous interrupt into the bridge call lifecycle.
    """

    @asynccontextmanager
    async def stream_context() -> AsyncGenerator[object]:
        yield object()

    bridge = SyncStreamBridge(stream_context(), async_alternative='`async_method`')

    def interrupt() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        bridge.call(interrupt)

    assert bridge._owner_task.done()  # pyright: ignore[reportPrivateUsage]


def test_sync_stream_bridge_rejects_streaming_after_shutdown_before_creating_pump(monkeypatch: pytest.MonkeyPatch):
    """A closed bridge rejects iteration without creating another loop-bound task.

    VCR cannot inspect the bridge's in-process task lifecycle.
    """

    @asynccontextmanager
    async def stream_context() -> AsyncGenerator[object]:
        yield object()

    bridge = SyncStreamBridge(stream_context(), async_alternative='`async_method`')
    bridge.shutdown()
    source = MagicMock()
    task_context = MagicMock(side_effect=AssertionError('must not create a pump task'))
    monkeypatch.setattr(bridge, '_task_context', task_context)

    with pytest.raises(RuntimeError, match='already closed'):
        next(bridge.stream_sync(source))

    source.assert_not_called()
    task_context.assert_not_called()


def test_sync_stream_bridge_interrupt_without_pump_preserves_original_error():
    """An interrupt after call completion propagates and leaves the event loop reusable.

    Callback ordering is deterministic: completing the call queues `run_until_complete()`'s stop callback,
    then the already-queued interrupt escapes before that stop callback runs. Shutdown must consume the stale
    callback so it cannot stop the next unrelated loop drive. VCR cannot control this in-process ordering.
    """
    cleanup_complete = False

    @asynccontextmanager
    async def stream_context() -> AsyncGenerator[object]:
        nonlocal cleanup_complete
        try:
            yield object()
        finally:
            cleanup_complete = True

    bridge = SyncStreamBridge(stream_context(), async_alternative='`async_method`')
    loop = bridge._loop  # pyright: ignore[reportPrivateUsage]
    original_error = KeyboardInterrupt('original')

    def interrupt() -> None:
        raise original_error

    def finish_then_interrupt() -> None:
        # Finishing the call queues its `run_until_complete()` stop callback behind this interrupt.
        loop.call_soon(interrupt)

    with pytest.raises(KeyboardInterrupt) as exc_info:
        bridge.call(finish_then_interrupt)

    assert exc_info.value is original_error
    assert cleanup_complete
    assert bridge._owner_task.done()  # pyright: ignore[reportPrivateUsage]
    # A leftover stop callback would stop this drive before `sleep(0)` completes and raise `RuntimeError`.
    assert loop.run_until_complete(asyncio.sleep(0)) is None


def test_sync_stream_bridge_task_drain_retries_multiple_early_stops(monkeypatch: pytest.MonkeyPatch):
    """Task draining tolerates multiple early stops without depending on their exception text.

    VCR cannot replace the local event-loop driver or inject stale stop callbacks.
    """
    loop = asyncio.new_event_loop()
    task = loop.create_task(asyncio.sleep(0))
    original_run_until_complete = loop.run_until_complete
    calls = 0

    def stop_twice(awaitable: Awaitable[object]) -> object:
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise RuntimeError(f'custom loop early stop {calls}')
        return original_run_until_complete(awaitable)

    with monkeypatch.context() as context:
        context.setattr(loop, 'run_until_complete', stop_twice)
        _run_task_to_completion(loop, task)

    assert calls == 3
    assert task.done()
    loop.close()


def test_sync_stream_bridge_task_drain_propagates_error_after_completion(monkeypatch: pytest.MonkeyPatch):
    """A loop-driver error after the waiter completes is propagated instead of retried.

    VCR cannot replace the local event-loop driver or exercise this defensive cleanup branch.
    """
    loop = asyncio.new_event_loop()
    task = loop.create_task(asyncio.sleep(0))
    original_run_until_complete = loop.run_until_complete
    error = RuntimeError('loop driver failed after completing the waiter')

    def finish_then_fail(awaitable: Awaitable[object]) -> object:
        original_run_until_complete(awaitable)
        raise error

    with monkeypatch.context() as context:
        context.setattr(loop, 'run_until_complete', finish_then_fail)
        with pytest.raises(RuntimeError) as exc_info:
            _run_task_to_completion(loop, task)

    assert exc_info.value is error
    assert task.done()
    loop.close()


def test_sync_stream_bridge_task_drain_propagates_error_while_loop_is_running(monkeypatch: pytest.MonkeyPatch):
    """A persistent loop-driver error is propagated instead of retried indefinitely.

    VCR cannot replace the local event-loop driver or simulate another thread driving its loop.
    """
    loop = asyncio.new_event_loop()
    task = loop.create_task(asyncio.sleep(0))
    original_run_until_complete = loop.run_until_complete
    error = RuntimeError('event loop is already running')

    def fail_to_drive_loop(awaitable: Awaitable[object]) -> object:
        raise error

    with monkeypatch.context() as context:
        context.setattr(loop, 'run_until_complete', fail_to_drive_loop)
        context.setattr(loop, 'is_running', lambda: True)
        with pytest.raises(RuntimeError) as exc_info:
            _run_task_to_completion(loop, task)

    assert exc_info.value is error
    pending_tasks = asyncio.all_tasks(loop)
    for pending_task in pending_tasks:
        pending_task.cancel()
    original_run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
    loop.close()


def test_sync_stream_bridge_interrupt_drains_pump_before_owner_exit():
    """A stale loop-stop callback cannot let owner cleanup overtake a cancelled stream pump.

    VCR cannot control event-loop callback ordering or observe task cleanup order.
    """
    source_cleaned = False
    owner_saw_source_cleaned: bool | None = None

    @asynccontextmanager
    async def stream_context() -> AsyncGenerator[object]:
        nonlocal owner_saw_source_cleaned
        try:
            yield object()
        finally:
            owner_saw_source_cleaned = source_cleaned

    async def source() -> AsyncIterator[str]:
        nonlocal source_cleaned
        try:
            yield 'first'
            await asyncio.Event().wait()
        finally:
            # Keep pump cleanup pending long enough for a stale stop callback to interrupt its first drain.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            source_cleaned = True

    bridge = SyncStreamBridge(stream_context(), async_alternative='`async_method`')
    stream = bridge.stream_sync(source)
    assert next(stream) == 'first'

    loop = bridge._loop  # pyright: ignore[reportPrivateUsage]

    def interrupt() -> None:
        raise KeyboardInterrupt

    def finish_then_interrupt() -> None:
        # The call task finishes and queues its `run_until_complete()` stop callback behind this interrupt.
        # The interrupt escapes first, leaving that stop callback queued for the pump drain in shutdown.
        loop.call_soon(interrupt)

    with pytest.raises(KeyboardInterrupt):
        bridge.call(finish_then_interrupt)

    assert source_cleaned
    assert owner_saw_source_cleaned is True
    cast(Generator[str, None, None], stream).close()


def test_sync_stream_bridge_rejects_use_from_another_thread():
    """Normal use never moves the caller-owned event loop to another thread.

    VCR cannot exercise or observe the bridge's in-process thread affinity.
    """

    @asynccontextmanager
    async def stream_context() -> AsyncGenerator[object]:
        yield object()

    bridge = SyncStreamBridge(stream_context(), async_alternative='`async_method`')
    errors: list[RuntimeError] = []

    def use_bridge() -> None:
        try:
            bridge.call(lambda: None)
        except RuntimeError as exc:
            errors.append(exc)

    thread = threading.Thread(target=use_bridge)
    thread.start()
    thread.join()

    assert len(errors) == 1
    assert str(errors[0]) == 'A synchronous stream must be used and closed on the thread where it was created.'
    bridge.shutdown()


def test_sync_stream_bridge_defers_iterator_close_from_another_thread():
    """Closing a suspended iterator queues its pump cleanup back to the owner thread.

    VCR cannot exercise or observe the bridge's in-process thread affinity.
    """
    owner_thread_id = threading.get_ident()
    cleanup_thread_id: int | None = None

    @asynccontextmanager
    async def stream_context() -> AsyncGenerator[object]:
        yield object()

    async def source() -> AsyncIterator[str]:
        nonlocal cleanup_thread_id
        try:
            yield 'first'
            await asyncio.Event().wait()
        finally:
            cleanup_thread_id = threading.get_ident()

    bridge = SyncStreamBridge(stream_context(), async_alternative='`async_method`')
    stream = bridge.stream_sync(source)
    assert next(stream) == 'first'

    def close_stream() -> None:
        cast(Generator[str, None, None], stream).close()

    thread = threading.Thread(target=close_stream)
    thread.start()
    thread.join()

    assert cleanup_thread_id is None

    bridge.call(asyncio.sleep, 0)
    bridge.call(asyncio.sleep, 0)
    assert cleanup_thread_id == owner_thread_id
    assert not bridge._pump_tasks  # pyright: ignore[reportPrivateUsage]
    bridge.shutdown()


def test_sync_stream_bridge_rejects_iterator_resume_from_another_thread():
    """Resuming a suspended iterator cannot move its pump cleanup to another thread.

    VCR cannot exercise or observe the bridge's in-process thread affinity.
    """
    owner_thread_id = threading.get_ident()
    cleanup_thread_id: int | None = None

    @asynccontextmanager
    async def stream_context() -> AsyncGenerator[object]:
        yield object()

    async def source() -> AsyncIterator[str]:
        nonlocal cleanup_thread_id
        try:
            yield 'first'
            await asyncio.Event().wait()
        finally:
            cleanup_thread_id = threading.get_ident()

    bridge = SyncStreamBridge(stream_context(), async_alternative='`async_method`')
    stream = bridge.stream_sync(source)
    assert next(stream) == 'first'
    errors: list[RuntimeError] = []

    def resume_stream() -> None:
        try:
            next(stream)
        except RuntimeError as exc:
            errors.append(exc)

    thread = threading.Thread(target=resume_stream)
    thread.start()
    thread.join()

    assert len(errors) == 1
    assert str(errors[0]) == 'A synchronous stream must be used and closed on the thread where it was created.'
    assert cleanup_thread_id is None

    bridge.shutdown()
    assert cleanup_thread_id == owner_thread_id


def test_sync_stream_bridge_defers_iterator_gc_from_another_thread(monkeypatch: pytest.MonkeyPatch):
    """Foreign-thread iterator GC queues cleanup without emitting an unraisable exception.

    VCR cannot control garbage collection or observe the bridge's in-process thread affinity.
    """
    owner_thread_id = threading.get_ident()
    cleanup_thread_id: int | None = None
    unraisable: list[object] = []
    monkeypatch.setattr(sys, 'unraisablehook', unraisable.append)

    @asynccontextmanager
    async def stream_context() -> AsyncGenerator[object]:
        yield object()

    async def source() -> AsyncIterator[str]:
        nonlocal cleanup_thread_id
        try:
            yield 'first'
            await asyncio.Event().wait()
        finally:
            cleanup_thread_id = threading.get_ident()

    bridge = SyncStreamBridge(stream_context(), async_alternative='`async_method`')
    stream = bridge.stream_sync(source)
    assert next(stream) == 'first'
    holder = [stream]
    del stream

    def drop_stream() -> None:
        holder.clear()
        gc.collect()

    thread = threading.Thread(target=drop_stream)
    thread.start()
    thread.join()

    assert not unraisable
    assert cleanup_thread_id is None

    bridge.call(asyncio.sleep, 0)
    bridge.call(asyncio.sleep, 0)
    assert cleanup_thread_id == owner_thread_id
    assert not bridge._pump_tasks  # pyright: ignore[reportPrivateUsage]
    bridge.shutdown()


def test_sync_stream_bridge_closes_iterator_gc_after_shutdown(monkeypatch: pytest.MonkeyPatch):
    """Foreign-thread iterator GC closes its receive stream after wrapper shutdown.

    VCR cannot control garbage collection or observe the bridge's in-process stream resources.
    """
    original_loop = _utils.get_event_loop()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    cleanup_thread_id: int | None = None
    unraisable: list[object] = []
    monkeypatch.setattr(sys, 'unraisablehook', unraisable.append)

    @asynccontextmanager
    async def stream_context() -> AsyncGenerator[object]:
        yield object()

    async def source() -> AsyncIterator[str]:
        nonlocal cleanup_thread_id
        try:
            yield 'first'
            await asyncio.Event().wait()
        finally:
            cleanup_thread_id = threading.get_ident()

    try:
        bridge = SyncStreamBridge(stream_context(), async_alternative='`async_method`')
        stream = bridge.stream_sync(source)
        assert next(stream) == 'first'
        bridge.shutdown()
        assert cleanup_thread_id == threading.get_ident()
        holder = [stream]
        del stream

        def drop_stream() -> None:
            holder.clear()
            gc.collect()

        with warnings.catch_warnings():
            warnings.simplefilter('error', ResourceWarning)
            thread = threading.Thread(target=drop_stream)
            thread.start()
            thread.join()
            loop.close()
            gc.collect()
    finally:
        if not loop.is_closed():  # pragma: no cover
            loop.close()
        asyncio.set_event_loop(original_loop)

    assert not unraisable


def test_sync_stream_bridge_rejects_use_while_another_loop_runs():
    """A sync bridge never nests its owner loop inside another running event loop.

    This is an in-process loop-state guard that a provider cassette cannot exercise.
    """

    @asynccontextmanager
    async def stream_context() -> AsyncGenerator[object]:
        yield object()

    bridge = SyncStreamBridge(stream_context(), async_alternative='`async_method`')

    async def use_bridge() -> None:
        with pytest.raises(RuntimeError, match='while an event loop is running'):
            bridge.call(lambda: None)

    asyncio.run(use_bridge())
    bridge.shutdown()


def test_sync_stream_bridge_init_interrupt_cleans_owner():
    """An interrupt during context entry cancels the owner task without leaking its ready future.

    VCR cannot inject the interrupt timing or inspect the bridge's pending futures.
    """
    loop = _utils.get_event_loop()

    def interrupt() -> None:
        raise KeyboardInterrupt

    @asynccontextmanager
    async def stream_context() -> AsyncGenerator[object]:
        loop.call_soon(interrupt)
        await asyncio.sleep(1)
        yield object()  # pragma: no cover

    with pytest.raises(KeyboardInterrupt):
        SyncStreamBridge(stream_context(), async_alternative='`async_method`')


def test_sync_stream_bridge_init_interrupt_after_entry_exits_context():
    """An interrupt after context entry still runs the context manager's cleanup.

    VCR cannot inject an interrupt between context entry and result delivery.
    """
    original_loop = _utils.get_event_loop()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    exited = False

    def interrupt() -> None:
        raise KeyboardInterrupt

    @asynccontextmanager
    async def stream_context() -> AsyncGenerator[object]:
        nonlocal exited
        loop.call_soon(interrupt)
        try:
            yield object()
        finally:
            exited = True

    try:
        with pytest.raises(KeyboardInterrupt):
            SyncStreamBridge(stream_context(), async_alternative='`async_method`')
        assert loop.run_until_complete(asyncio.sleep(0)) is None
    finally:
        loop.close()
        asyncio.set_event_loop(original_loop)

    assert exited


@pytest.mark.parametrize('error_type', [KeyboardInterrupt, SystemExit])
def test_sync_stream_bridge_init_propagates_base_exception(error_type: type[BaseException]):
    """A base exception from `__aenter__` escapes immediately instead of hanging the event loop.

    VCR cannot inject an in-process base exception into the context-manager entry protocol.
    """
    original_loop = _utils.get_event_loop()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    error = error_type('entry failed')
    forced_stop = False

    class FailingContextManager:
        async def __aenter__(self) -> object:
            raise error

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            pytest.fail('`__aexit__` must not be called when `__aenter__` fails')  # pragma: no cover

    def force_stop() -> None:  # pragma: no cover
        nonlocal forced_stop
        forced_stop = True
        loop.stop()

    stop_handle = loop.call_later(1, force_stop)
    try:
        with pytest.raises(error_type) as exc_info:
            SyncStreamBridge(FailingContextManager(), async_alternative='`async_method`')
        assert exc_info.value is error
        assert not forced_stop
        assert loop.run_until_complete(asyncio.sleep(0)) is None
    finally:
        stop_handle.cancel()
        loop.close()
        asyncio.set_event_loop(original_loop)


def test_sync_stream_bridge_exit_error_propagates():
    """An error raised after context-manager cleanup is complete propagates to the sync caller.

    VCR cannot make the in-process context manager fail during its exit protocol.
    """

    @asynccontextmanager
    async def stream_context() -> AsyncGenerator[object]:
        try:
            yield object()
        finally:
            raise RuntimeError('exit failed')

    bridge = SyncStreamBridge(stream_context(), async_alternative='`async_method`')

    with pytest.raises(RuntimeError, match='exit failed'):
        bridge.shutdown()

    assert bridge._owner_task.done()  # pyright: ignore[reportPrivateUsage]


def test_sync_stream_bridge_owner_cancellation_can_be_suppressed():
    """Owner cancellation follows the async context manager's suppression semantics.

    VCR cannot control task cancellation or inspect the context manager's exception arguments.
    """
    exit_type: type[BaseException] | None = None

    class SuppressingContextManager:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> bool:
            nonlocal exit_type
            exit_type = exc_type
            return True

    bridge = SyncStreamBridge(SuppressingContextManager(), async_alternative='`async_method`')
    bridge._owner_task.cancel()  # pyright: ignore[reportPrivateUsage]
    bridge._loop.run_until_complete(bridge._owner_task)  # pyright: ignore[reportPrivateUsage]

    assert exit_type is asyncio.CancelledError
    bridge.shutdown()


def test_sync_stream_bridge_shutdown_accepts_prior_exit_request():
    """Shutdown remains idempotent if cleanup was already requested on the owner loop.

    VCR cannot place the bridge into this internal cleanup state.
    """

    @asynccontextmanager
    async def stream_context() -> AsyncGenerator[object]:
        yield object()

    bridge = SyncStreamBridge(stream_context(), async_alternative='`async_method`')
    bridge._exit_requested.set_result((None, None, None))  # pyright: ignore[reportPrivateUsage]
    bridge.shutdown()

    assert bridge._owner_task.done()  # pyright: ignore[reportPrivateUsage]


def test_sync_stream_bridge_finalizes_while_owner_loop_is_running():
    """The non-context-manager fallback can request cleanup from its running owner loop.

    VCR cannot control garbage collection or inspect owner-task completion.
    """
    result = Agent(TestModel()).run_stream_sync('Hello')
    bridge = result._bridge  # pyright: ignore[reportPrivateUsage]
    loop = bridge._loop  # pyright: ignore[reportPrivateUsage]
    owner_task = bridge._owner_task  # pyright: ignore[reportPrivateUsage]
    bridge_ref = weakref.ref(bridge)
    holder = [result]
    del result, bridge

    async def drop_result() -> None:
        holder.clear()
        gc.collect()
        await asyncio.sleep(0)

    loop.run_until_complete(drop_result())
    loop.run_until_complete(owner_task)

    assert bridge_ref() is None
    assert owner_task.done()


def test_sync_stream_bridge_finalizes_with_unclosed_iterator():
    """Dropping an active iterator releases its pump before the wrapper's GC cleanup runs.

    VCR cannot control garbage collection or inspect the in-process pump tasks.
    """
    result = Agent(TestModel(custom_output_text='The cat sat on the mat.')).run_stream_sync('Hello')
    bridge = result._bridge  # pyright: ignore[reportPrivateUsage]
    owner_task = bridge._owner_task  # pyright: ignore[reportPrivateUsage]
    stream = result.stream_text(delta=True, debounce_by=None)
    assert next(stream)
    pump_tasks = tuple(bridge._pump_tasks)  # pyright: ignore[reportPrivateUsage]
    bridge_ref = weakref.ref(bridge)

    del stream, result, bridge
    gc.collect()

    assert bridge_ref() is None
    assert owner_task.done()
    assert all(task.done() for task in pump_tasks)


def test_sync_stream_bridge_running_loop_finalizer_drains_pumps():
    """Finalization on a running loop cancels pumps before releasing the owner task.

    VCR cannot trigger this finalizer state or observe task cancellation order.
    """

    async def run() -> None:
        loop = asyncio.get_running_loop()
        exit_requested: asyncio.Future[
            tuple[type[BaseException] | None, BaseException | None, TracebackType | None]
        ] = loop.create_future()

        async def wait_for_exit() -> None:
            await exit_requested

        async def wait_forever() -> None:
            await asyncio.Event().wait()

        owner_task = loop.create_task(wait_for_exit())
        pump_task = loop.create_task(wait_forever())
        pump_tasks = {pump_task}

        _finalize_loop(loop, owner_task, exit_requested, pump_tasks, threading.get_ident())
        await owner_task

        assert pump_task.cancelled()
        assert not pump_tasks

        # A queued duplicate request is harmless after both cleanup signals are complete.
        await _request_exit(owner_task, exit_requested, pump_tasks)

    asyncio.run(run())


def test_sync_stream_bridge_gc_request_retrieves_owner_exit_error():
    """Best-effort GC cleanup consumes an error raised while the owner context exits.

    VCR cannot inspect whether an in-process task exception was retrieved.
    """

    async def run() -> None:
        loop = asyncio.get_running_loop()
        exit_requested: asyncio.Future[
            tuple[type[BaseException] | None, BaseException | None, TracebackType | None]
        ] = loop.create_future()

        async def fail_on_exit() -> None:
            await exit_requested
            raise RuntimeError('exit failed')

        owner_task = loop.create_task(fail_on_exit())
        await _request_exit(owner_task, exit_requested, set())

        assert owner_task.done()
        assert isinstance(owner_task.exception(), RuntimeError)

    asyncio.run(run())


def test_sync_stream_bridge_finalizes_while_another_loop_is_running():
    """The non-context-manager fallback requests cleanup if another event loop is active.

    VCR cannot control garbage collection while a second event loop is running.
    """
    result = Agent(TestModel()).run_stream_sync('Hello')
    bridge = result._bridge  # pyright: ignore[reportPrivateUsage]
    owner_loop = bridge._loop  # pyright: ignore[reportPrivateUsage]
    owner_task = bridge._owner_task  # pyright: ignore[reportPrivateUsage]
    bridge_ref = weakref.ref(bridge)
    holder = [result]
    del result, bridge

    async def drop_result() -> None:
        holder.clear()
        gc.collect()

    try:
        asyncio.run(drop_result())
        owner_loop.run_until_complete(owner_task)
    finally:
        asyncio.set_event_loop(owner_loop)

    assert bridge_ref() is None
    assert owner_task.done()


def test_sync_stream_bridge_finalizes_on_another_thread():
    """The GC fallback never moves the stopped caller-owned loop to the finalizer thread.

    VCR cannot control the finalizer thread or observe event-loop thread affinity.
    """
    result = Agent(TestModel()).run_stream_sync('Hello')
    bridge = result._bridge  # pyright: ignore[reportPrivateUsage]
    owner_loop = bridge._loop  # pyright: ignore[reportPrivateUsage]
    owner_task = bridge._owner_task  # pyright: ignore[reportPrivateUsage]
    bridge_ref = weakref.ref(bridge)
    holder = [result]
    del result, bridge

    def drop_result() -> None:
        holder.clear()
        gc.collect()

    thread = threading.Thread(target=drop_result)
    thread.start()
    thread.join()

    assert bridge_ref() is None
    assert not owner_task.done()
    owner_loop.run_until_complete(owner_task)
    assert owner_task.done()


def test_sync_stream_bridge_finalizer_ignores_closed_loop():
    """The GC fallback cannot drive a loop that its owner has already closed.

    Once a caller closes the owner loop, async cleanup cannot safely run in-process. A provider cassette
    cannot observe this local lifecycle guard. A pending Future stands in for the owner task so the guard
    can be exercised without knowingly leaking a real task.
    """
    loop = asyncio.new_event_loop()
    exit_requested: asyncio.Future[tuple[type[BaseException] | None, BaseException | None, TracebackType | None]] = (
        loop.create_future()
    )
    owner_task = cast(asyncio.Task[None], loop.create_future())
    assert not owner_task.done()
    loop.close()

    _finalize_loop(loop, owner_task, exit_requested, set(), threading.get_ident())

    assert not owner_task.done()
    assert not exit_requested.done()


def test_run_stream_sync_keyboard_interrupt_closes_open_stream(monkeypatch: pytest.MonkeyPatch):
    """A Ctrl-C mid-stream tears down the still-open model stream instead of leaking it (#5975).

    The model stream is still open when the interrupt lands. Teardown must close it, which we observe
    via its `finally`.
    """
    stream_closed = threading.Event()

    async def stream_function(_messages: list[ModelMessage], _: AgentInfo) -> AsyncIterator[str]:
        try:
            # The final-result event lets `run_stream_sync` return here with the generator suspended at
            # this `yield` — i.e. the model stream (and its notional connection) still open. The `finally`
            # runs only when teardown closes it.
            yield 'The cat sat on the mat.'
        finally:
            stream_closed.set()

    agent = Agent(FunctionModel(stream_function=stream_function))
    result = agent.run_stream_sync('Hello')
    bridge = result._bridge  # pyright: ignore[reportPrivateUsage]
    assert not stream_closed.is_set()  # the model stream is open and producing

    _interrupt_next_loop_run(bridge, monkeypatch)

    with pytest.raises(KeyboardInterrupt):
        with result:
            result.get_output()

    # The interrupt teardown ran the still-open model stream's `finally`, so the connection was closed
    # rather than left pending on the loop until GC.
    assert stream_closed.wait(timeout=5)


def test_run_stream_sync_keyboard_interrupt_mid_iteration_closes_receive_stream(monkeypatch: pytest.MonkeyPatch):
    """A Ctrl-C *while iterating* a sync stream closes its receive stream too, leaking nothing (#5975).

    Without cleanup, the orphaned `MemoryObjectReceiveStream` warns from `__del__` at GC, which pytest
    escalates to an error.
    """
    agent = Agent(TestModel(custom_output_text='The cat sat on the mat.'))
    with agent.run_stream_sync('Hello') as result:
        bridge = result._bridge  # pyright: ignore[reportPrivateUsage]
        stream = result.stream_text(delta=True, debounce_by=None)
        assert next(stream)  # pump running, receive stream open

        _interrupt_next_loop_run(bridge, monkeypatch)

        # The interrupt propagates through the `stream_sync` generator's `finally`, which closes the receive stream.
        with pytest.raises(KeyboardInterrupt):
            next(stream)

    del stream
    gc.collect()  # surface any unclosed `MemoryObjectReceiveStream` now, not at session teardown


def test_run_stream_sync_early_break_tears_down_pump():
    """Abandoning a sync stream early unblocks and closes the pump without surfacing an error."""
    agent = Agent(TestModel(custom_output_text='The cat sat on the mat.'))
    with agent.run_stream_sync('Hello') as result:
        stream = result.stream_text(delta=True, debounce_by=None)
        assert next(stream)  # pull one chunk while the pump still has more to send
        # `stream_text` is typed `Iterator` but is a generator at runtime; closing it abandons the stream,
        # closing the receive end the pump is sending into.
        cast(Generator[str, None, None], stream).close()


def test_sync_stream_bridge_early_close_cancels_waiting_pump():
    """Closing a sync iterator cancels a pump waiting indefinitely on its source.

    VCR cannot inspect the pump task or deterministically hold its source pending.
    """
    source_closed = False
    wait_forever = asyncio.Event()

    @asynccontextmanager
    async def stream_context() -> AsyncGenerator[object]:
        yield object()

    async def source() -> AsyncIterator[str]:
        nonlocal source_closed
        try:
            yield 'first'
            await wait_forever.wait()
            yield 'unreachable'  # pragma: no cover
        finally:
            source_closed = True

    bridge = SyncStreamBridge(stream_context(), async_alternative='`async_method`')
    stream = bridge.stream_sync(source)
    assert next(stream) == 'first'
    cast(Generator[str, None, None], stream).close()

    assert source_closed
    assert not bridge._pump_tasks  # pyright: ignore[reportPrivateUsage]
    bridge.shutdown()


@pytest.mark.parametrize('error_type', [KeyboardInterrupt, SystemExit])
def test_sync_stream_bridge_pump_propagates_base_exception_without_hanging(error_type: type[BaseException]):
    """A base exception from a completed pump escapes without stranding the caller's event loop.

    VCR cannot inject an in-process base exception into the iterator pump task.
    """
    error = error_type('pump failed')
    forced_stop = False

    @asynccontextmanager
    async def stream_context() -> AsyncGenerator[object]:
        yield object()

    async def source() -> AsyncIterator[str]:
        yield 'first'
        raise error

    bridge = SyncStreamBridge(stream_context(), async_alternative='`async_method`')
    loop = bridge._loop  # pyright: ignore[reportPrivateUsage]
    stream = bridge.stream_sync(source)

    def force_stop() -> None:  # pragma: no cover
        nonlocal forced_stop
        forced_stop = True
        loop.stop()

    stop_handle = loop.call_later(1, force_stop)
    try:
        with pytest.raises(error_type) as exc_info:
            while True:
                next(stream)
        assert exc_info.value is error
        assert not forced_stop
        assert bridge._owner_task.done()  # pyright: ignore[reportPrivateUsage]
        assert loop.run_until_complete(asyncio.sleep(0)) is None
    finally:
        stop_handle.cancel()


def test_run_stream_sync_preserves_capability_contextvars():
    """Tasks driven by the sync bridge inherit context set inside the agent run.

    VCR cannot observe in-process context-variable propagation between tasks.
    """
    run_context: contextvars.ContextVar[str] = contextvars.ContextVar('run_context')

    @dataclass
    class Setter(AbstractCapability[Any]):
        async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
            token = run_context.set('from-wrap-run')
            try:
                return await handler()
            finally:
                run_context.reset(token)

    @dataclass
    class Reader(AbstractCapability[Any]):
        seen: list[str | None] = field(default_factory=lambda: [])

        async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
            self.seen.append(run_context.get(None))
            return node

        async def after_node_run(self, ctx: RunContext[Any], *, node: Any, result: Any) -> Any:
            self.seen.append(run_context.get(None))
            return result

    reader = Reader()
    agent = Agent(TestModel(), capabilities=[Setter(), reader])
    with agent.run_stream_sync('Hello') as result:
        assert result.get_output()

    assert reader.seen
    assert all(value == 'from-wrap-run' for value in reader.seen)


def test_run_stream_sync_preserves_current_caller_contextvars():
    """Run-owned context merges into caller values changed after constructing the sync result.

    VCR cannot observe caller context changes made between construction and consumption.
    """
    caller_context: contextvars.ContextVar[str] = contextvars.ContextVar('caller_context')
    seen: list[str | None] = []
    agent = Agent(TestModel(custom_output_text='done'))

    @agent.output_validator
    def validate_output(output: str) -> str:
        seen.append(caller_context.get(None))
        return output

    construction_token = caller_context.set('construction')
    try:
        result = agent.run_stream_sync('Hello')
        consumption_token = caller_context.set('consumption')
        try:
            with result:
                assert result.get_output() == 'done'
        finally:
            caller_context.reset(consumption_token)
    finally:
        caller_context.reset(construction_token)

    assert seen == ['consumption']


async def test_run_stream_early_break_during_debounce_closes_cleanly():
    """Breaking out of a debounced `stream_text()` mid-chunk must not raise from stream teardown.

    `stream_text()`/`stream_output()` debounce via `group_by_temporal`, which prefetches the next item in
    a background task. Abandoning the stream with an early `break` while that prefetch is parked in an
    in-flight `anext` on the model source used to make the run's `aclose()` raise
    `RuntimeError: aclose(): asynchronous generator is already running`; `PeekableAsyncStream` now
    serializes source access so `aclose()` waits for the prefetch to release the source first.
    """

    async def stream_function(_messages: list[ModelMessage], _: AgentInfo) -> AsyncIterator[str]:
        while True:  # `while True` (not a bounded loop) so teardown mid-loop leaves no uncovered exit branch
            yield 'chunk '
            await asyncio.sleep(0.2)  # keep a chunk in-flight (prefetched) when we break

    agent = Agent(FunctionModel(stream_function=stream_function))
    # Consume one chunk (default debounce spawns the prefetch task), then abandon the still-suspended
    # stream by leaving the `async with`. Tearing down while the prefetch is mid-`anext` must not raise.
    # (A single `anext` rather than `async for ...: break` avoids an uncovered loop-exit branch; keeping
    # `stream` referenced stops it being finalized early, which would cancel the prefetch and hide the bug.)
    async with agent.run_stream('hello') as result:
        stream = result.stream_text(delta=True)
        assert await anext(stream)


def test_run_stream_sync_rejects_already_entered_result():
    """Passing an already-entered `StreamedRunResult` (the old constructor arg) raises a clear error."""
    with pytest.raises(TypeError, match='now takes the `run_stream\\(\\)` context manager'):
        StreamedRunResultSync(cast(Any, object.__new__(StreamedRunResult)))


async def test_streamed_structured_response():
    m = TestModel()

    agent = Agent(m, output_type=tuple[str, str], name='fig_jam')

    async with agent.run_stream('') as result:
        assert agent.name == 'fig_jam'
        assert not result.is_complete
        response = await result.get_output()
        assert response == snapshot(('a', 'a'))
        assert result.is_complete
    assert result.response == snapshot(
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='final_result',
                    args={'response': ['a', 'a']},
                    tool_call_id='pyd_ai_tool_call_id__final_result',
                )
            ],
            usage=RequestUsage(input_tokens=50),
            model_name='test',
            timestamp=IsDatetime(),
            provider_name='test',
        )
    )


async def test_structured_response_iter():
    async def text_stream(_messages: list[ModelMessage], agent_info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        assert agent_info.output_tools is not None
        assert len(agent_info.output_tools) == 1
        name = agent_info.output_tools[0].name
        json_data = json.dumps({'response': [1, 2, 3, 4]})
        yield {0: DeltaToolCall(name=name)}
        yield {0: DeltaToolCall(json_args=json_data[:15])}
        yield {0: DeltaToolCall(json_args=json_data[15:])}

    agent = Agent(FunctionModel(stream_function=text_stream), output_type=list[int])

    chunks: list[list[int]] = []
    async with agent.run_stream('') as result:
        async for structured_response in result.stream_response(debounce_by=None):
            response_data = await result.validate_response_output(
                structured_response, allow_partial=structured_response.state == 'incomplete'
            )
            chunks.append(response_data)

    assert chunks == snapshot([[1], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])

    async with agent.run_stream('Hello') as result:
        with pytest.raises(UserError, match=r'stream_text\(\) can only be used with text responses'):
            async for _ in result.stream_text():
                pass


async def test_streamed_text_stream():
    m = TestModel(custom_output_text='The cat sat on the mat.')

    agent = Agent(m)

    async with agent.run_stream('Hello') as result:
        # typehint to test (via static typing) that the stream type is correctly inferred
        chunks: list[str] = [c async for c in result.stream_text()]
        # one chunk with `stream_text()` due to group_by_temporal
        assert chunks == snapshot(['The cat sat on the mat.'])
        assert result.is_complete

    async with agent.run_stream('Hello') as result:
        # typehint to test (via static typing) that the stream type is correctly inferred
        chunks: list[str] = [c async for c in result.stream_output()]
        # two chunks with `stream()` due to not-final vs. final
        assert chunks == snapshot(['The cat sat on the mat.', 'The cat sat on the mat.'])
        assert result.is_complete

    async with agent.run_stream('Hello') as result:
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(
            [
                'The ',
                'The cat ',
                'The cat sat ',
                'The cat sat on ',
                'The cat sat on the ',
                'The cat sat on the mat.',
            ]
        )

    async with agent.run_stream('Hello') as result:
        # with stream_text, there is no need to do partial validation, so we only get the final message once:
        assert [c async for c in result.stream_text(delta=False, debounce_by=None)] == snapshot(
            ['The ', 'The cat ', 'The cat sat ', 'The cat sat on ', 'The cat sat on the ', 'The cat sat on the mat.']
        )

    async with agent.run_stream('Hello') as result:
        assert [c async for c in result.stream_text(delta=True, debounce_by=None)] == snapshot(
            ['The ', 'cat ', 'sat ', 'on ', 'the ', 'mat.']
        )

    def upcase(text: str) -> str:
        return text.upper()

    async with agent.run_stream('Hello', output_type=TextOutput(upcase)) as result:
        assert [c async for c in result.stream_output(debounce_by=None)] == snapshot(
            [
                'THE ',
                'THE CAT ',
                'THE CAT SAT ',
                'THE CAT SAT ON ',
                'THE CAT SAT ON THE ',
                'THE CAT SAT ON THE MAT.',
                'THE CAT SAT ON THE MAT.',
            ]
        )

    async with agent.run_stream('Hello') as result:
        assert [c async for c in result.stream_response(debounce_by=None)] == snapshot(
            [
                ModelResponse(
                    parts=[TextPart(content='The ')],
                    usage=RequestUsage(input_tokens=51, output_tokens=1),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                    provider_name='test',
                    state='incomplete',
                ),
                ModelResponse(
                    parts=[TextPart(content='The cat ')],
                    usage=RequestUsage(input_tokens=51, output_tokens=2),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                    provider_name='test',
                    state='incomplete',
                ),
                ModelResponse(
                    parts=[TextPart(content='The cat sat ')],
                    usage=RequestUsage(input_tokens=51, output_tokens=3),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                    provider_name='test',
                    state='incomplete',
                ),
                ModelResponse(
                    parts=[TextPart(content='The cat sat on ')],
                    usage=RequestUsage(input_tokens=51, output_tokens=4),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                    provider_name='test',
                    state='incomplete',
                ),
                ModelResponse(
                    parts=[TextPart(content='The cat sat on the ')],
                    usage=RequestUsage(input_tokens=51, output_tokens=5),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                    provider_name='test',
                    state='incomplete',
                ),
                ModelResponse(
                    parts=[TextPart(content='The cat sat on the mat.')],
                    usage=RequestUsage(input_tokens=51, output_tokens=7),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                    provider_name='test',
                    state='incomplete',
                ),
                ModelResponse(
                    parts=[TextPart(content='The cat sat on the mat.')],
                    usage=RequestUsage(input_tokens=51, output_tokens=7),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                    provider_name='test',
                    state='incomplete',
                ),
                ModelResponse(
                    parts=[TextPart(content='The cat sat on the mat.')],
                    usage=RequestUsage(input_tokens=51, output_tokens=7),
                    model_name='test',
                    timestamp=IsDatetime(),
                    provider_name='test',
                    state='complete',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )


def test_streamed_text_stream_sync():
    m = TestModel(custom_output_text='The cat sat on the mat.')

    agent = Agent(m)

    result = agent.run_stream_sync('Hello')
    # typehint to test (via static typing) that the stream type is correctly inferred
    chunks: list[str] = [c for c in result.stream_text()]
    # one chunk with `stream_text()` due to group_by_temporal
    assert chunks == snapshot(['The cat sat on the mat.'])
    assert result.is_complete

    result = agent.run_stream_sync('Hello')
    # typehint to test (via static typing) that the stream type is correctly inferred
    chunks: list[str] = [c for c in result.stream_output()]
    # two chunks with `stream()` due to not-final vs. final
    assert chunks == snapshot(['The cat sat on the mat.', 'The cat sat on the mat.'])
    assert result.is_complete

    result = agent.run_stream_sync('Hello')
    assert [c for c in result.stream_text(debounce_by=None)] == snapshot(
        [
            'The ',
            'The cat ',
            'The cat sat ',
            'The cat sat on ',
            'The cat sat on the ',
            'The cat sat on the mat.',
        ]
    )

    result = agent.run_stream_sync('Hello')
    # with stream_text, there is no need to do partial validation, so we only get the final message once:
    assert [c for c in result.stream_text(delta=False, debounce_by=None)] == snapshot(
        ['The ', 'The cat ', 'The cat sat ', 'The cat sat on ', 'The cat sat on the ', 'The cat sat on the mat.']
    )

    result = agent.run_stream_sync('Hello')
    assert [c for c in result.stream_text(delta=True, debounce_by=None)] == snapshot(
        ['The ', 'cat ', 'sat ', 'on ', 'the ', 'mat.']
    )

    def upcase(text: str) -> str:
        return text.upper()

    result = agent.run_stream_sync('Hello', output_type=TextOutput(upcase))
    assert [c for c in result.stream_output(debounce_by=None)] == snapshot(
        [
            'THE ',
            'THE CAT ',
            'THE CAT SAT ',
            'THE CAT SAT ON ',
            'THE CAT SAT ON THE ',
            'THE CAT SAT ON THE MAT.',
            'THE CAT SAT ON THE MAT.',
        ]
    )

    result = agent.run_stream_sync('Hello')
    assert [c for c in result.stream_response(debounce_by=None)] == snapshot(
        [
            ModelResponse(
                parts=[TextPart(content='The ')],
                usage=RequestUsage(input_tokens=51, output_tokens=1),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='test',
                state='incomplete',
            ),
            ModelResponse(
                parts=[TextPart(content='The cat ')],
                usage=RequestUsage(input_tokens=51, output_tokens=2),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='test',
                state='incomplete',
            ),
            ModelResponse(
                parts=[TextPart(content='The cat sat ')],
                usage=RequestUsage(input_tokens=51, output_tokens=3),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='test',
                state='incomplete',
            ),
            ModelResponse(
                parts=[TextPart(content='The cat sat on ')],
                usage=RequestUsage(input_tokens=51, output_tokens=4),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='test',
                state='incomplete',
            ),
            ModelResponse(
                parts=[TextPart(content='The cat sat on the ')],
                usage=RequestUsage(input_tokens=51, output_tokens=5),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='test',
                state='incomplete',
            ),
            ModelResponse(
                parts=[TextPart(content='The cat sat on the mat.')],
                usage=RequestUsage(input_tokens=51, output_tokens=7),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='test',
                state='incomplete',
            ),
            ModelResponse(
                parts=[TextPart(content='The cat sat on the mat.')],
                usage=RequestUsage(input_tokens=51, output_tokens=7),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='test',
                state='incomplete',
            ),
            ModelResponse(
                parts=[TextPart(content='The cat sat on the mat.')],
                usage=RequestUsage(input_tokens=51, output_tokens=7),
                model_name='test',
                timestamp=IsDatetime(),
                provider_name='test',
                state='complete',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_plain_response():
    call_index = 0

    async def text_stream(_messages: list[ModelMessage], _: AgentInfo) -> AsyncIterator[str]:
        nonlocal call_index

        call_index += 1
        yield 'hello '
        yield 'world'

    agent = Agent(FunctionModel(stream_function=text_stream), output_type=tuple[str, str])

    with pytest.raises(UnexpectedModelBehavior, match=r'Exceeded maximum output retries \(1\)'):
        async with agent.run_stream(''):
            pass

    assert call_index == 2


async def test_stream_output_type_union_data_before_kind():
    """A valid union envelope streamed with `data` before `kind` must not crash mid-stream.

    While `kind` is still a partial trailing string (e.g. `'App'`), envelope validation must
    fail (so the chunk is skipped) rather than reach the union processor's `kind` lookup.
    Streaming manifestation of https://github.com/pydantic/pydantic-ai/issues/5844.
    """

    class Apple(BaseModel):
        color: str

    class Banana(BaseModel):
        length: float

    async def text_stream(_messages: list[ModelMessage], _: AgentInfo) -> AsyncIterator[str]:
        # `data` first, so that `kind` is the trailing partial string while streaming.
        for char in '{"result": {"data": {"color": "red"}, "kind": "Apple"}}':
            yield char

    agent = Agent(FunctionModel(stream_function=text_stream), output_type=PromptedOutput([Apple, Banana]))

    async with agent.run_stream('What fruit is it?') as result:
        async for _ in result.stream_output(debounce_by=None):
            pass
        assert await result.get_output() == snapshot(Apple(color='red'))


async def test_call_tool():
    async def stream_structured_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) == 1:
            assert agent_info.function_tools is not None
            assert len(agent_info.function_tools) == 1
            name = agent_info.function_tools[0].name
            part = message_part(messages, UserPromptPart)
            json_string = json.dumps({'x': part.content})
            yield {0: DeltaToolCall(name=name)}
            yield {0: DeltaToolCall(json_args=json_string[:3])}
            yield {0: DeltaToolCall(json_args=json_string[3:])}
        else:
            part = message_part(messages, ToolReturnPart, message_index=-1)
            assert agent_info.output_tools is not None
            assert len(agent_info.output_tools) == 1
            name = agent_info.output_tools[0].name
            json_data = json.dumps({'response': [part.content, 2]})
            yield {0: DeltaToolCall(name=name)}
            yield {0: DeltaToolCall(json_args=json_data[:5])}
            yield {0: DeltaToolCall(json_args=json_data[5:])}

    agent = Agent(FunctionModel(stream_function=stream_structured_function), output_type=tuple[str, int])

    @agent.tool_plain
    async def ret_a(x: str) -> str:
        assert x == 'hello'
        return f'{x} world'

    async with agent.run_stream('hello') as result:
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='ret_a', args='{"x": "hello"}', tool_call_id=IsStr())],
                    usage=RequestUsage(input_tokens=50, output_tokens=5),
                    model_name='function::stream_structured_function',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='ret_a',
                            content='hello world',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )
        assert await result.get_output() == snapshot(('hello world', 2))
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='ret_a', args='{"x": "hello"}', tool_call_id=IsStr())],
                    usage=RequestUsage(input_tokens=50, output_tokens=5),
                    model_name='function::stream_structured_function',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='ret_a',
                            content='hello world',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"response": ["hello world", 2]}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=7),
                    model_name='function::stream_structured_function',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )


async def test_empty_response():
    async def stream_structured_function(
        messages: list[ModelMessage], _: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) == 1:
            yield {}
        else:
            yield 'ok here is text'

    agent = Agent(FunctionModel(stream_function=stream_structured_function))

    async with agent.run_stream('hello') as result:
        response = await result.get_output()
        assert response == snapshot('ok here is text')
        messages = result.all_messages()

    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='hello',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[],
                usage=RequestUsage(input_tokens=50),
                model_name='function::stream_structured_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Please return text.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='ok here is text')],
                usage=RequestUsage(input_tokens=50, output_tokens=4),
                model_name='function::stream_structured_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_run_stream_allows_none_output_empty_response():
    """`run_stream()` with `output_type=str | None` should return `None` on an empty model response."""

    async def empty_stream(_messages: list[ModelMessage], _: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        yield {}

    agent = Agent(FunctionModel(stream_function=empty_stream), output_type=str | None)

    async with agent.run_stream('hello') as result:
        assert await result.get_output() is None
        assert result.is_complete
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[],
                    usage=RequestUsage(input_tokens=50),
                    model_name='function::empty_stream',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )


async def test_call_tool_wrong_name():
    async def stream_structured_function(_messages: list[ModelMessage], _: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        yield {0: DeltaToolCall(name='foobar', json_args='{}')}

    agent = Agent(
        FunctionModel(stream_function=stream_structured_function),
        output_type=tuple[str, int],
        retries={'tools': 0, 'output': 0},
    )

    @agent.tool_plain
    async def ret_a(x: str) -> str:  # pragma: no cover
        return x

    with capture_run_messages() as messages:
        with pytest.raises(UnexpectedModelBehavior, match=r"Tool 'foobar' exceeded max retries count of 0"):
            async with agent.run_stream('hello'):
                pass

    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='foobar', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=50, output_tokens=1),
                model_name='function::stream_structured_function',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_invalid_output_tool_args_get_output():
    """Regression test for https://github.com/pydantic/pydantic-ai/issues/3638."""

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        assert info.output_tools is not None and len(info.output_tools) == 1
        yield {0: DeltaToolCall(name=info.output_tools[0].name)}
        yield {0: DeltaToolCall(json_args='{"response": ["hello", "not_an_int"]}')}

    agent = Agent(FunctionModel(stream_function=stream_fn), output_type=tuple[str, int])

    with pytest.raises(UnexpectedModelBehavior, match='retries are not supported in `run_stream'):
        async with agent.run_stream('hello') as result:
            await result.get_output()


async def test_invalid_output_tool_args_stream_output():
    """Regression test for https://github.com/pydantic/pydantic-ai/issues/3638."""

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        assert info.output_tools is not None and len(info.output_tools) == 1
        yield {0: DeltaToolCall(name=info.output_tools[0].name)}
        yield {0: DeltaToolCall(json_args='{"response": ["hello", "not_an_int"]}')}

    agent = Agent(FunctionModel(stream_function=stream_fn), output_type=tuple[str, int])

    with pytest.raises(UnexpectedModelBehavior, match='retries are not supported in `run_stream'):
        async with agent.run_stream('hello') as result:
            async for _ in result.stream_output(debounce_by=None):
                pass


class TestPartialOutput:
    """Tests for `ctx.partial_output` flag in output validators and output functions."""

    # NOTE: When changing tests in this class:
    # 1. Follow the existing order
    # 2. Update tests in `tests/test_agent.py::TestPartialOutput` as well

    async def test_output_validator_text(self):
        """Test that output validators receive correct value for `partial_output` with text output."""
        call_log: list[tuple[str, bool]] = []

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            for chunk in ['Hello', ' ', 'world', '!']:
                yield chunk

        agent = Agent(FunctionModel(stream_function=sf))

        @agent.output_validator
        def validate_output(ctx: RunContext, output: str) -> str:
            call_log.append((output, ctx.partial_output))
            return output

        async with agent.run_stream('test') as result:
            text_parts = [text_part async for text_part in result.stream_text(debounce_by=None)]

        assert text_parts[-1] == 'Hello world!'
        assert call_log == snapshot(
            [
                ('Hello', True),
                ('Hello ', True),
                ('Hello world', True),
                ('Hello world!', True),
                ('Hello world!', False),
            ]
        )

    async def test_output_validator_structured(self):
        """Test that output validators receive correct value for `partial_output` with structured output."""
        call_log: list[tuple[Foo, bool]] = []

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            assert info.output_tools is not None
            yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args='{"a": 42')}
            yield {0: DeltaToolCall(json_args=', "b": "f')}
            yield {0: DeltaToolCall(json_args='oo"}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=Foo)

        @agent.output_validator
        def validate_output(ctx: RunContext, output: Foo) -> Foo:
            call_log.append((output, ctx.partial_output))
            return output

        async with agent.run_stream('test') as result:
            outputs = [output async for output in result.stream_output(debounce_by=None)]

        assert outputs[-1] == Foo(a=42, b='foo')
        assert call_log == snapshot(
            [
                (Foo(a=42, b='f'), True),
                (Foo(a=42, b='foo'), True),
                (Foo(a=42, b='foo'), False),
            ]
        )

    async def test_output_function_text(self):
        """Test that output functions receive correct value for `partial_output` with text output."""
        call_log: list[tuple[str, bool]] = []

        def process_output(ctx: RunContext, text: str) -> str:
            call_log.append((text, ctx.partial_output))
            return text.upper()

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            for chunk in ['Hello', ' ', 'world', '!']:
                yield chunk

        agent = Agent(FunctionModel(stream_function=sf), output_type=TextOutput(process_output))

        async with agent.run_stream('test') as result:
            outputs = [output async for output in result.stream_output(debounce_by=None)]

        assert outputs[-1] == 'HELLO WORLD!'
        assert call_log == snapshot(
            [
                ('Hello', True),
                ('Hello ', True),
                ('Hello world', True),
                ('Hello world!', True),
                ('Hello world!', False),
            ]
        )

    async def test_output_function_structured(self):
        """Test that output functions receive correct value for `partial_output` with structured output."""
        call_log: list[tuple[Foo, bool]] = []

        def process_foo(ctx: RunContext, foo: Foo) -> Foo:
            call_log.append((foo, ctx.partial_output))
            return Foo(a=foo.a * 2, b=foo.b.upper())

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            assert info.output_tools is not None
            yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args='{"a": 21')}
            yield {0: DeltaToolCall(json_args=', "b": "f')}
            yield {0: DeltaToolCall(json_args='oo"}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=process_foo)

        async with agent.run_stream('test') as result:
            outputs = [output async for output in result.stream_output(debounce_by=None)]

        assert outputs[-1] == Foo(a=42, b='FOO')
        assert call_log == snapshot(
            [
                (Foo(a=21, b='f'), True),
                (Foo(a=21, b='foo'), True),
                (Foo(a=21, b='foo'), False),
            ]
        )

    async def test_output_function_structured_get_output(self):
        """Test that output functions receive correct value for `partial_output` with `get_output()`.

        When using only `get_output()` without streaming, the output processor is called only once
        with `partial_output=False` (final validation), since the user doesn't see partial results.
        """
        call_log: list[tuple[Foo, bool]] = []

        def process_foo(ctx: RunContext, foo: Foo) -> Foo:
            call_log.append((foo, ctx.partial_output))
            return Foo(a=foo.a * 2, b=foo.b.upper())

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            assert info.output_tools is not None
            yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args='{"a": 21, "b": "foo"}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=ToolOutput(process_foo, name='my_output'))

        async with agent.run_stream('test') as result:
            output = await result.get_output()

        assert output == Foo(a=42, b='FOO')
        assert call_log == snapshot([(Foo(a=21, b='foo'), False)])

    async def test_output_function_structured_stream_output_only(self):
        """Test that output functions receive correct value for `partial_output` with `stream_output()`.

        When using only `stream_output()`, the LAST yielded output should have `partial_output=False` (final validation).
        """
        call_log: list[tuple[Foo, bool]] = []

        def process_foo(ctx: RunContext, foo: Foo) -> Foo:
            call_log.append((foo, ctx.partial_output))
            return Foo(a=foo.a * 2, b=foo.b.upper())

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            assert info.output_tools is not None
            yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args='{"a": 21, "b": "foo"}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=ToolOutput(process_foo, name='my_output'))

        async with agent.run_stream('test') as result:
            outputs = [output async for output in result.stream_output()]

        assert outputs[-1] == Foo(a=42, b='FOO')
        assert call_log == snapshot(
            [
                (Foo(a=21, b='foo'), True),
                (Foo(a=21, b='foo'), False),
            ],
        )

    async def test_stream_output_partial_then_final_validation(self):
        """Test that stream_output() calls validators with partial_output=True during streaming, then False at the end.

        This verifies the critical invariant: output validators/functions are called multiple times with
        partial_output=True as chunks arrive, followed by exactly one call with partial_output=False
        for final validation. The final yield may have the same content as the last partial yield,
        but the validation semantics differ (partial validation may accept incomplete data).
        """
        call_log: list[tuple[Foo, bool]] = []

        def process_foo(ctx: RunContext, foo: Foo) -> Foo:
            call_log.append((foo, ctx.partial_output))
            return Foo(a=foo.a * 2, b=foo.b.upper())

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            assert info.output_tools is not None
            yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args='{"a": 21')}
            yield {0: DeltaToolCall(json_args=', "b": "f')}
            yield {0: DeltaToolCall(json_args='oo"}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=ToolOutput(process_foo, name='my_output'))

        async with agent.run_stream('test') as result:
            outputs = [output async for output in result.stream_output(debounce_by=None)]

        assert outputs[-1] == Foo(a=42, b='FOO')

        # Verify the pattern: multiple True calls, exactly one False call at the end
        partial_output_flags = [partial for _, partial in call_log]
        assert partial_output_flags[-1] is False, 'Last call must have partial_output=False'
        assert all(flag is True for flag in partial_output_flags[:-1]), (
            'All calls except last must have partial_output=True'
        )
        assert len([f for f in partial_output_flags if f is False]) == 1, 'Exactly one partial_output=False call'

        # The full call log shows progressive partial outputs followed by final validation
        assert call_log == snapshot(
            [
                (Foo(a=21, b='f'), True),
                (Foo(a=21, b='foo'), True),
                (Foo(a=21, b='foo'), False),  # Final validation - same content, different validation mode
            ]
        )

    # NOTE: When changing tests in this class:
    # 1. Follow the existing order
    # 2. Update tests in `tests/test_agent.py::TestPartialOutput` as well


class TestStreamingCachedOutput:
    async def test_output_function_structured_double_stream_output(self):
        """Test that calling `stream_output()` twice works correctly.

        The first `stream_output()` should do validations and cache the result.
        The second `stream_output()` should return cached results without re-validation.
        """
        call_log: list[tuple[Foo, bool]] = []

        def process_foo(ctx: RunContext, foo: Foo) -> Foo:
            call_log.append((foo, ctx.partial_output))
            return Foo(a=foo.a * 2, b=foo.b.upper())

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            assert info.output_tools is not None
            yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args='{"a": 21, "b": "foo"}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=ToolOutput(process_foo, name='my_output'))

        async with agent.run_stream('test') as result:
            outputs1 = [output async for output in result.stream_output()]
            outputs2 = [output async for output in result.stream_output()]

        assert outputs1[-1] == outputs2[-1] == Foo(a=42, b='FOO')
        assert call_log == snapshot(
            [
                (Foo(a=21, b='foo'), True),
                (Foo(a=21, b='foo'), False),
            ],
        )

    async def test_output_validator_text_double_stream_text(self):
        """Test that calling `stream_text()` twice works correctly with output validator.

        The first `stream_text()` should do validations and cache the result.
        The second `stream_text()` should return cached results without re-validation.
        """
        call_log: list[tuple[str, bool]] = []

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            for chunk in ['Hello', ' ', 'world', '!']:
                yield chunk

        agent = Agent(FunctionModel(stream_function=sf))

        @agent.output_validator
        def validate_output(ctx: RunContext, output: str) -> str:
            call_log.append((output, ctx.partial_output))
            return output

        async with agent.run_stream('test') as result:
            text_parts1 = [text async for text in result.stream_text(debounce_by=None)]
            text_parts2 = [text async for text in result.stream_text(debounce_by=None)]

        assert text_parts1[-1] == text_parts2[-1] == 'Hello world!'
        assert call_log == snapshot(
            [
                ('Hello', True),
                ('Hello ', True),
                ('Hello world', True),
                ('Hello world!', True),
                ('Hello world!', False),
            ],
        )

    async def test_output_function_structured_double_get_output(self):
        """Test that calling `get_output()` twice works correctly.

        The first `get_output()` should do validation and cache the result.
        The second `get_output()` should return cached results without re-validation.
        """
        call_log: list[tuple[Foo, bool]] = []

        def process_foo(ctx: RunContext, foo: Foo) -> Foo:
            call_log.append((foo, ctx.partial_output))
            return Foo(a=foo.a * 2, b=foo.b.upper())

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            assert info.output_tools is not None
            yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args='{"a": 21, "b": "foo"}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=ToolOutput(process_foo, name='my_output'))

        async with agent.run_stream('test') as result:
            output1 = await result.get_output()
            output2 = await result.get_output()

        assert output1 == output2 == Foo(a=42, b='FOO')
        assert call_log == snapshot([(Foo(a=21, b='foo'), False)])

    async def test_cached_output_mutation_does_not_affect_cache(self):
        """Test that mutating a returned cached output does not affect the cached value.

        When the same output is retrieved multiple times from cache, each call should return
        a deep copy, so mutations to one don't affect subsequent retrievals.
        """

        def process_foo(ctx: RunContext, foo: Foo) -> Foo:
            return Foo(a=foo.a * 2, b=foo.b.upper())

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            assert info.output_tools is not None
            yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args='{"a": 21, "b": "foo"}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=ToolOutput(process_foo, name='my_output'))

        async with agent.run_stream('test') as result:
            # Get the first output and mutate it
            output1 = await result.get_output()
            output1.a = 999
            output1.b = 'MUTATED'

            # Get the second output - should not be affected by mutation
            output2 = await result.get_output()

        # First output should have been mutated
        assert output1 == Foo(a=999, b='MUTATED')
        # Second output should be the original cached value (not mutated)
        assert output2 == Foo(a=42, b='FOO')


class OutputType(BaseModel):
    """Result type used by multiple tests."""

    value: str


class TestMultipleToolCalls:
    """Tests for scenarios where multiple tool calls are made in a single response."""

    # NOTE: When changing tests in this class:
    # 1. Follow the existing order
    # 2. Update tests in `tests/test_agent.py::TestMultipleToolCalls` as well

    async def test_early_strategy_stops_after_first_final_result(self):
        """Test that 'early' strategy stops processing regular tools after first final result."""
        tool_called: list[str] = []

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('final_result', '{"value": "final"}')}
            yield {2: DeltaToolCall('regular_tool', '{"x": 1}')}
            yield {3: DeltaToolCall('another_tool', '{"y": 2}')}
            yield {4: DeltaToolCall('deferred_tool', '{"x": 3}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=OutputType, end_strategy='early')

        @agent.tool_plain
        def regular_tool(x: int) -> int:  # pragma: no cover
            """A regular tool that should not be called."""
            tool_called.append('regular_tool')
            return x

        @agent.tool_plain
        def another_tool(y: int) -> int:  # pragma: no cover
            """Another tool that should not be called."""
            tool_called.append('another_tool')
            return y

        async def defer(ctx: RunContext, tool_def: ToolDefinition) -> ToolDefinition | None:
            return replace(tool_def, kind='external')

        @agent.tool_plain(prepare=defer)
        def deferred_tool(x: int) -> int:  # pragma: no cover
            return x + 1

        async with agent.run_stream('test early strategy') as result:
            response = await result.get_output()
            assert response.value == snapshot('final')
            messages = result.all_messages()

        # Verify no tools were called after final result
        assert tool_called == []

        # Verify we got tool returns for all calls
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test early strategy', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='final_result', args='{"value": "final"}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='regular_tool', args='{"x": 1}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='another_tool', args='{"y": 2}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='deferred_tool', args='{"x": 3}', tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=13),
                    model_name='function::sf',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        ),
                        ToolReturnPart(
                            tool_name='regular_tool',
                            content='Tool not executed - a final result was already processed.',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        ),
                        ToolReturnPart(
                            tool_name='another_tool',
                            content='Tool not executed - a final result was already processed.',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        ),
                        ToolReturnPart(
                            tool_name='deferred_tool',
                            content='Tool not executed - a final result was already processed.',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    @pytest.mark.parametrize('output_mode', ['native', 'prompted'])
    async def test_early_strategy_prefers_structured_text_output_over_tool_calls(self, output_mode: str):
        """Under 'early', valid native/prompted output text streamed alongside function tool calls is the
        final result, so the function tools are skipped — matching the non-streaming behavior."""
        tool_called: list[str] = []

        async def sf(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            yield '{"value": "final"}'
            yield {1: DeltaToolCall('regular_tool', '{"x": 1}')}

        output_type = NativeOutput(OutputType) if output_mode == 'native' else PromptedOutput(OutputType)
        agent = Agent(FunctionModel(stream_function=sf), output_type=output_type, end_strategy='early')

        @agent.tool_plain
        def regular_tool(x: int) -> int:  # pragma: no cover
            tool_called.append('regular_tool')
            return x

        async with agent.run_stream('test early structured output') as result:
            output = await result.get_output()
            messages = result.all_messages()

        assert output == OutputType(value='final')
        assert tool_called == []
        assert isinstance(messages[-1], ModelRequest)
        skipped = messages[-1].parts[0]
        assert isinstance(skipped, ToolReturnPart)
        assert skipped.tool_name == 'regular_tool'
        assert skipped.content == 'Tool not executed - a final result was already processed.'

    async def test_non_early_strategy_runs_tools_alongside_structured_text_output(self):
        """Under 'graceful', function tools streamed alongside structured text output still run. (In streaming
        the text output is committed the instant it streams, so it remains the final result — unlike the
        non-streaming graceful case, which continues the run and ends on the post-tool output.)"""
        tool_called: list[str] = []

        async def sf(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            yield '{"value": "final"}'
            yield {1: DeltaToolCall('regular_tool', '{"x": 1}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=NativeOutput(OutputType), end_strategy='graceful')

        @agent.tool_plain
        def regular_tool(x: int) -> int:
            tool_called.append('regular_tool')
            return x

        async with agent.run_stream('test graceful structured output') as result:
            output = await result.get_output()

        assert output == OutputType(value='final')
        assert tool_called == ['regular_tool']

    async def test_early_strategy_does_not_call_additional_output_tools(self):
        """Test that 'early' strategy does not execute additional output tool functions."""
        output_tools_called: list[str] = []

        def process_first(output: OutputType) -> OutputType:
            """Process first output."""
            output_tools_called.append('first')
            return output

        def process_second(output: OutputType) -> OutputType:  # pragma: no cover
            """Process second output."""
            output_tools_called.append('second')
            return output

        async def stream_function(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('first_output', '{"value": "first"}')}
            yield {2: DeltaToolCall('second_output', '{"value": "second"}')}

        agent = Agent(
            FunctionModel(stream_function=stream_function),
            output_type=[
                ToolOutput(process_first, name='first_output'),
                ToolOutput(process_second, name='second_output'),
            ],
            end_strategy='early',
        )

        async with agent.run_stream('test early output tools') as result:
            response = await result.get_output()

        # Verify the result came from the first output tool
        assert isinstance(response, OutputType)
        assert response.value == 'first'

        # Verify only the first output tool was called
        assert output_tools_called == ['first']

        # Verify we got tool returns in the correct order
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test early output tools', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='first_output', args='{"value": "first"}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='second_output', args='{"value": "second"}', tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=8),
                    model_name='function::stream_function',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='first_output',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='second_output',
                            content='Output tool not used - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_early_strategy_uses_first_final_result(self):
        """Test that 'early' strategy uses the first final result and ignores subsequent ones."""

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('final_result', '{"value": "first"}')}
            yield {2: DeltaToolCall('final_result', '{"value": "second"}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=OutputType, end_strategy='early')

        async with agent.run_stream('test multiple final results') as result:
            response = await result.get_output()
            assert response.value == snapshot('first')
            messages = result.all_messages()

        # Verify we got appropriate tool returns
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test multiple final results', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='final_result', args='{"value": "first"}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='final_result', args='{"value": "second"}', tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=8),
                    model_name='function::sf',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        ),
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Output tool not used - a final result was already processed.',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_early_strategy_with_final_result_in_middle(self):
        """Test that 'early' strategy stops at first final result, regardless of position."""
        tool_called: list[str] = []

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('regular_tool', '{"x": 1}')}
            yield {2: DeltaToolCall('final_result', '{"value": "final"}')}
            yield {3: DeltaToolCall('another_tool', '{"y": 2}')}
            yield {4: DeltaToolCall('unknown_tool', '{"value": "???"}')}
            yield {5: DeltaToolCall('deferred_tool', '{"x": 5}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=OutputType, end_strategy='early')

        @agent.tool_plain
        def regular_tool(x: int) -> int:  # pragma: no cover
            """A regular tool that should not be called."""
            tool_called.append('regular_tool')
            return x

        @agent.tool_plain
        def another_tool(y: int) -> int:  # pragma: no cover
            """A tool that should not be called."""
            tool_called.append('another_tool')
            return y

        async def defer(ctx: RunContext, tool_def: ToolDefinition) -> ToolDefinition | None:
            return replace(tool_def, kind='external')

        @agent.tool_plain(prepare=defer)
        def deferred_tool(x: int) -> int:  # pragma: no cover
            return x + 1

        async with agent.run_stream('test early strategy with final result in middle') as result:
            response = await result.get_output()
            assert response.value == snapshot('final')
            messages = result.all_messages()

        # Verify no tools were called
        assert tool_called == []

        # Verify we got appropriate tool returns
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='test early strategy with final result in middle',
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='regular_tool',
                            args='{"x": 1}',
                            tool_call_id=IsStr(),
                        ),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"value": "final"}',
                            tool_call_id=IsStr(),
                        ),
                        ToolCallPart(
                            tool_name='another_tool',
                            args='{"y": 2}',
                            tool_call_id=IsStr(),
                        ),
                        ToolCallPart(
                            tool_name='unknown_tool',
                            args='{"value": "???"}',
                            tool_call_id=IsStr(),
                        ),
                        ToolCallPart(
                            tool_name='deferred_tool',
                            args='{"x": 5}',
                            tool_call_id=IsStr(),
                        ),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=17),
                    model_name='function::sf',
                    timestamp=IsNow(tz=datetime.timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='regular_tool',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='another_tool',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        ),
                        ToolReturnPart(
                            content='Tool not executed - a final result was already processed.',
                            tool_name='unknown_tool',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='deferred_tool',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_early_strategy_with_external_tool_call(self):
        """Test that early strategy handles external tool calls correctly.

        Streaming and non-streaming modes differ in how they choose the final result:
        - Streaming: First tool call (in response order) that can produce a final result (output or deferred)
        - Non-streaming: First output tool (if none called, all deferred tools become final result)

        See https://github.com/pydantic/pydantic-ai/issues/3636#issuecomment-3618800480 for details.
        """
        tool_called: list[str] = []

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('external_tool')}
            yield {2: DeltaToolCall('final_result', '{"value": "final"}')}
            yield {3: DeltaToolCall('regular_tool', '{"x": 1}')}

        agent = Agent(
            FunctionModel(stream_function=sf),
            output_type=[OutputType, DeferredToolRequests],
            toolsets=[
                ExternalToolset(
                    tool_defs=[
                        ToolDefinition(
                            name='external_tool',
                            kind='external',
                        )
                    ]
                )
            ],
            end_strategy='early',
        )

        @agent.tool_plain
        def regular_tool(x: int) -> int:  # pragma: no cover
            """A regular tool that should not be called."""
            tool_called.append('regular_tool')
            return x

        async with agent.run_stream('test early strategy with external tool call') as result:
            response = await result.get_output()
            assert response == snapshot(
                DeferredToolRequests(
                    calls=[
                        ToolCallPart(
                            tool_name='external_tool',
                            tool_call_id=IsStr(),
                        )
                    ]
                )
            )
            messages = result.all_messages()

        # Verify no tools were called
        assert tool_called == []

        # Verify we got appropriate tool returns
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='test early strategy with external tool call',
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='external_tool', tool_call_id=IsStr()),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"value": "final"}',
                            tool_call_id=IsStr(),
                        ),
                        ToolCallPart(
                            tool_name='regular_tool',
                            args='{"x": 1}',
                            tool_call_id=IsStr(),
                        ),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=7),
                    model_name='function::sf',
                    timestamp=IsNow(tz=datetime.timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Output tool not used - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='regular_tool',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_early_strategy_with_deferred_tool_call(self):
        """Test that early strategy handles deferred tool calls correctly."""
        tool_called: list[str] = []

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('deferred_tool')}
            yield {2: DeltaToolCall('regular_tool', '{"x": 1}')}

        agent = Agent(
            FunctionModel(stream_function=sf),
            output_type=[str, DeferredToolRequests],
            end_strategy='early',
        )

        @agent.tool_plain
        def deferred_tool() -> int:
            raise CallDeferred

        @agent.tool_plain
        def regular_tool(x: int) -> int:
            tool_called.append('regular_tool')
            return x

        async with agent.run_stream('test early strategy with external tool call') as result:
            response = await result.get_output()
            assert response == snapshot(
                DeferredToolRequests(calls=[ToolCallPart(tool_name='deferred_tool', tool_call_id=IsStr())])
            )
            messages = result.all_messages()

        # Verify regular tool was called
        assert tool_called == ['regular_tool']

        # Verify we got appropriate tool returns
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='test early strategy with external tool call',
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='deferred_tool', tool_call_id=IsStr()),
                        ToolCallPart(
                            tool_name='regular_tool',
                            args='{"x": 1}',
                            tool_call_id=IsStr(),
                        ),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=3),
                    model_name='function::sf',
                    timestamp=IsNow(tz=datetime.timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='regular_tool',
                            content=1,
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_early_strategy_does_not_apply_to_tool_calls_without_final_tool(self):
        """Test that 'early' strategy does not apply to tool calls when no output tool is called."""
        tool_called: list[str] = []
        agent = Agent(TestModel(), output_type=OutputType, end_strategy='early')

        @agent.tool_plain
        def regular_tool(x: int) -> int:
            """A regular tool that should be called."""
            tool_called.append('regular_tool')
            return x

        async with agent.run_stream('test early strategy with regular tool calls') as result:
            response = await result.get_output()
            assert response.value == snapshot('a')
            messages = result.all_messages()

        # Verify the regular tool was executed
        assert tool_called == ['regular_tool']

        # Verify we got appropriate tool returns
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='test early strategy with regular tool calls',
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='regular_tool',
                            args={'x': 0},
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=57),
                    model_name='test',
                    timestamp=IsNow(tz=datetime.timezone.utc),
                    provider_name='test',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='regular_tool',
                            content=0,
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args={'value': 'a'},
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=58, output_tokens=4),
                    model_name='test',
                    timestamp=IsNow(tz=datetime.timezone.utc),
                    provider_name='test',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_graceful_strategy_executes_function_tools_but_skips_output_tools(self):
        """Test that 'graceful' strategy executes function tools but skips remaining output tools."""
        tool_called: list[str] = []

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('final_result', '{"value": "first"}')}
            yield {2: DeltaToolCall('regular_tool', '{"x": 42}')}
            yield {3: DeltaToolCall('another_tool', '{"y": 2}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=OutputType, end_strategy='graceful')

        @agent.tool_plain
        def regular_tool(x: int) -> int:
            """A regular tool that should be called."""
            tool_called.append('regular_tool')
            return x

        @agent.tool_plain
        def another_tool(y: int) -> int:
            """Another tool that should be called."""
            tool_called.append('another_tool')
            return y

        async with agent.run_stream('test graceful strategy') as result:
            response = await result.get_output()
            assert response.value == snapshot('first')
            messages = result.all_messages()

        # Verify all function tools were called
        assert sorted(tool_called) == sorted(['regular_tool', 'another_tool'])

        # Verify we got tool returns in the correct order
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test graceful strategy', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='final_result', args='{"value": "first"}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='regular_tool', args='{"x": 42}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='another_tool', args='{"y": 2}', tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=10),
                    model_name='function::sf',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        ),
                        ToolReturnPart(
                            tool_name='regular_tool',
                            content=42,
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='another_tool', content=2, tool_call_id=IsStr(), timestamp=IsNow(tz=timezone.utc)
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_graceful_strategy_does_not_call_additional_output_tools(self):
        """Test that 'graceful' strategy does not execute additional output tool functions."""
        output_tools_called: list[str] = []

        def process_first(output: OutputType) -> OutputType:
            """Process first output."""
            output_tools_called.append('first')
            return output

        def process_second(output: OutputType) -> OutputType:  # pragma: no cover
            """Process second output."""
            output_tools_called.append('second')
            return output

        async def stream_function(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('first_output', '{"value": "first"}')}
            yield {2: DeltaToolCall('second_output', '{"value": "second"}')}

        agent = Agent(
            FunctionModel(stream_function=stream_function),
            output_type=[
                ToolOutput(process_first, name='first_output'),
                ToolOutput(process_second, name='second_output'),
            ],
            end_strategy='graceful',
        )

        async with agent.run_stream('test graceful output tools') as result:
            response = await result.get_output()

        # Verify the result came from the first output tool
        assert isinstance(response, OutputType)
        assert response.value == 'first'

        # Verify only the first output tool was called
        assert output_tools_called == ['first']

        # Verify we got tool returns in the correct order
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test graceful output tools', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='first_output', args='{"value": "first"}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='second_output', args='{"value": "second"}', tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=8),
                    model_name='function::stream_function',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='first_output',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='second_output',
                            content='Output tool not used - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_graceful_strategy_uses_first_final_result(self):
        """Test that 'graceful' strategy uses the first final result and ignores subsequent ones."""

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('final_result', '{"value": "first"}')}
            yield {2: DeltaToolCall('final_result', '{"value": "second"}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=OutputType, end_strategy='graceful')

        async with agent.run_stream('test multiple final results') as result:
            response = await result.get_output()
            assert response.value == snapshot('first')
            messages = result.all_messages()

        # Verify we got appropriate tool returns
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test multiple final results', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='final_result', args='{"value": "first"}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='final_result', args='{"value": "second"}', tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=8),
                    model_name='function::sf',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        ),
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Output tool not used - a final result was already processed.',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_graceful_strategy_with_final_result_in_middle(self):
        """Test that 'graceful' strategy executes function tools but skips output and deferred tools."""
        tool_called: list[str] = []

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('regular_tool', '{"x": 1}')}
            yield {2: DeltaToolCall('final_result', '{"value": "final"}')}
            yield {3: DeltaToolCall('another_tool', '{"y": 2}')}
            yield {4: DeltaToolCall('unknown_tool', '{"value": "???"}')}
            yield {5: DeltaToolCall('deferred_tool', '{"x": 5}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=OutputType, end_strategy='graceful')

        @agent.tool_plain
        def regular_tool(x: int) -> int:
            """A regular tool that should be called."""
            tool_called.append('regular_tool')
            return x

        @agent.tool_plain
        def another_tool(y: int) -> int:
            """Another tool that should be called."""
            tool_called.append('another_tool')
            return y

        async def defer(ctx: RunContext, tool_def: ToolDefinition) -> ToolDefinition | None:
            return replace(tool_def, kind='external')

        @agent.tool_plain(prepare=defer)
        def deferred_tool(x: int) -> int:  # pragma: no cover
            tool_called.append('deferred_tool')
            return x + 1

        async with agent.run_stream('test graceful strategy with final result in middle') as result:
            response = await result.get_output()
            assert response.value == snapshot('final')
            messages = result.all_messages()

        # Verify function tools were called but deferred tools were not
        assert sorted(tool_called) == sorted(['regular_tool', 'another_tool'])

        # Verify we got appropriate tool returns
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='test graceful strategy with final result in middle',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='regular_tool',
                            args='{"x": 1}',
                            tool_call_id=IsStr(),
                        ),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"value": "final"}',
                            tool_call_id=IsStr(),
                        ),
                        ToolCallPart(
                            tool_name='another_tool',
                            args='{"y": 2}',
                            tool_call_id=IsStr(),
                        ),
                        ToolCallPart(
                            tool_name='unknown_tool',
                            args='{"value": "???"}',
                            tool_call_id=IsStr(),
                        ),
                        ToolCallPart(
                            tool_name='deferred_tool',
                            args='{"x": 5}',
                            tool_call_id=IsStr(),
                        ),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=17),
                    model_name='function::sf',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='regular_tool',
                            content=1,
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='another_tool',
                            content=2,
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        RetryPromptPart(
                            content="Unknown tool name: 'unknown_tool'. Available tools: 'another_tool', 'deferred_tool', 'final_result', 'regular_tool'",
                            tool_name='unknown_tool',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='deferred_tool',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_exhaustive_strategy_executes_all_tools(self):
        """Test that 'exhaustive' strategy executes all tools while using first final result."""
        tool_called: list[str] = []

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('regular_tool', '{"x": 42}')}
            yield {2: DeltaToolCall('final_result', '{"value": "first"}')}
            yield {3: DeltaToolCall('another_tool', '{"y": 2}')}
            yield {4: DeltaToolCall('final_result', '{"value": "second"}')}
            yield {5: DeltaToolCall('unknown_tool', '{"value": "???"}')}
            yield {6: DeltaToolCall('deferred_tool', '{"x": 4}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=OutputType, end_strategy='exhaustive')

        @agent.tool_plain
        def regular_tool(x: int) -> int:
            """A regular tool that should be called."""
            tool_called.append('regular_tool')
            return x

        @agent.tool_plain
        def another_tool(y: int) -> int:
            """Another tool that should be called."""
            tool_called.append('another_tool')
            return y

        async def defer(ctx: RunContext, tool_def: ToolDefinition) -> ToolDefinition | None:
            return replace(tool_def, kind='external')

        @agent.tool_plain(prepare=defer)
        def deferred_tool(x: int) -> int:  # pragma: no cover
            return x + 1

        async with agent.run_stream('test exhaustive strategy') as result:
            response = await result.get_output()
            assert response.value == snapshot('first')
            messages = result.all_messages()

        # Verify the result came from the first final tool
        assert response.value == 'first'

        # Verify all regular tools were called
        assert sorted(tool_called) == sorted(['regular_tool', 'another_tool'])

        # Verify we got tool returns in the correct order
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test exhaustive strategy', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='regular_tool', args='{"x": 42}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='final_result', args='{"value": "first"}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='another_tool', args='{"y": 2}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='final_result', args='{"value": "second"}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='unknown_tool', args='{"value": "???"}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='deferred_tool', args='{"x": 4}', tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=21),
                    model_name='function::sf',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='regular_tool',
                            content=42,
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        ),
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        ),
                        ToolReturnPart(
                            tool_name='another_tool',
                            content=2,
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Output tool processed, but its value will not be the final result of the agent run.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        RetryPromptPart(
                            content="Unknown tool name: 'unknown_tool'. Available tools: 'another_tool', 'deferred_tool', 'final_result', 'regular_tool'",
                            tool_name='unknown_tool',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='deferred_tool',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_exhaustive_strategy_calls_all_output_tools(self):
        """Test that 'exhaustive' strategy executes all output tool functions."""
        output_tools_called: list[str] = []

        def process_first(output: OutputType) -> OutputType:
            """Process first output."""
            output_tools_called.append('first')
            return output

        def process_second(output: OutputType) -> OutputType:
            """Process second output."""
            output_tools_called.append('second')
            return output

        async def stream_function(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('first_output', '{"value": "first"}')}
            yield {2: DeltaToolCall('second_output', '{"value": "second"}')}

        agent = Agent(
            FunctionModel(stream_function=stream_function),
            output_type=[
                ToolOutput(process_first, name='first_output'),
                ToolOutput(process_second, name='second_output'),
            ],
            end_strategy='exhaustive',
        )

        async with agent.run_stream('test exhaustive output tools') as result:
            response = await result.get_output()

        # Verify the result came from the first output tool
        assert isinstance(response, OutputType)
        assert response.value == 'first'

        # Verify both output tools were called
        assert sorted(output_tools_called) == ['first', 'second']

        # Verify we got tool returns in the correct order
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test exhaustive output tools', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='first_output', args='{"value": "first"}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='second_output', args='{"value": "second"}', tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=8),
                    model_name='function::stream_function',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='first_output',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='second_output',
                            content='Output tool processed, but its value will not be the final result of the agent run.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    @pytest.mark.xfail(reason='See https://github.com/pydantic/pydantic-ai/issues/3393')
    async def test_exhaustive_strategy_invalid_first_valid_second_output(self):
        """Test that exhaustive strategy uses the second valid output when the first is invalid."""
        output_tools_called: list[str] = []

        def process_first(output: OutputType) -> OutputType:
            """Process first output - will be invalid."""
            output_tools_called.append('first')
            raise ModelRetry('First output validation failed')

        def process_second(output: OutputType) -> OutputType:
            """Process second output - will be valid."""
            output_tools_called.append('second')
            return output

        async def stream_function(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('first_output', '{"value": "invalid"}')}
            yield {2: DeltaToolCall('second_output', '{"value": "valid"}')}

        agent = Agent(
            FunctionModel(stream_function=stream_function),
            output_type=[
                ToolOutput(process_first, name='first_output'),
                ToolOutput(process_second, name='second_output'),
            ],
            end_strategy='exhaustive',
        )

        async with agent.run_stream('test invalid first valid second') as result:
            response = await result.get_output()

        # Verify the result came from the second output tool (first was invalid)
        assert isinstance(response, OutputType)
        assert response.value == snapshot('valid')

        # Verify both output tools were called
        assert sorted(output_tools_called) == ['first', 'second']

        # Verify we got appropriate messages
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test invalid first valid second', timestamp=IsNow(tz=timezone.utc))],
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='first_output', args='{"value": "invalid"}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='second_output', args='{"value": "valid"}', tool_call_id=IsStr()),
                    ],
                    model_name='function:stream_function:',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='First output validation failed',
                            tool_name='first_output',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='second_output',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_exhaustive_strategy_valid_first_invalid_second_output(self):
        """Test that exhaustive strategy uses the first valid output even when the second is invalid."""
        output_tools_called: list[str] = []

        def process_first(output: OutputType) -> OutputType:
            """Process first output - will be valid."""
            output_tools_called.append('first')
            return output

        def process_second(output: OutputType) -> OutputType:
            """Process second output - will be invalid."""
            output_tools_called.append('second')
            raise ModelRetry('Second output validation failed')

        async def stream_function(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('first_output', '{"value": "valid"}')}
            yield {2: DeltaToolCall('second_output', '{"value": "invalid"}')}

        agent = Agent(
            FunctionModel(stream_function=stream_function),
            output_type=[
                ToolOutput(process_first, name='first_output'),
                ToolOutput(process_second, name='second_output'),
            ],
            end_strategy='exhaustive',
            retries={'output': 0},  # No retries - model must succeed first try
        )

        async with agent.run_stream('test valid first invalid second') as result:
            response = await result.get_output()

        # Verify the result came from the first output tool (second was invalid, but we ignore it)
        assert isinstance(response, OutputType)
        assert response.value == snapshot('valid')

        # Verify both output tools were called
        assert sorted(output_tools_called) == ['first', 'second']

        # Verify we got appropriate messages
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test valid first invalid second', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='first_output', args='{"value": "valid"}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='second_output', args='{"value": "invalid"}', tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=8),
                    model_name='function::stream_function',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='first_output',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='second_output',
                            content='Output tool not used - output function execution failed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_exhaustive_strategy_with_tool_retry_and_final_result(self):
        """Test that exhaustive strategy doesn't increment retries when `final_result` exists and `ToolRetryError` occurs."""
        output_tools_called: list[str] = []

        def process_first(output: OutputType) -> OutputType:
            """Process first output - will be valid."""
            output_tools_called.append('first')
            return output

        def process_second(output: OutputType) -> OutputType:
            """Process second output - will raise ModelRetry."""
            output_tools_called.append('second')
            raise ModelRetry('Second output validation failed')

        async def stream_function(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('first_output', '{"value": "valid"}')}
            yield {2: DeltaToolCall('second_output', '{"value": "invalid"}')}

        agent = Agent(
            FunctionModel(stream_function=stream_function),
            output_type=[
                ToolOutput(process_first, name='first_output'),
                ToolOutput(process_second, name='second_output'),
            ],
            end_strategy='exhaustive',
            retries={'output': 1},  # Allow 1 retry so ToolRetryError is raised
        )

        async with agent.run_stream('test exhaustive with tool retry') as result:
            response = await result.get_output()

        # Verify the result came from the first output tool
        assert isinstance(response, OutputType)
        assert response.value == 'valid'

        # Verify both output tools were called
        assert sorted(output_tools_called) == ['first', 'second']

        # Verify we got appropriate messages
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='test exhaustive with tool retry', timestamp=IsNow(tz=datetime.timezone.utc)
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='first_output', args='{"value": "valid"}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='second_output', args='{"value": "invalid"}', tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=8),
                    model_name='function::stream_function',
                    timestamp=IsNow(tz=datetime.timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='first_output',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        ),
                        RetryPromptPart(
                            content='Second output validation failed',
                            tool_name='second_output',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    @pytest.mark.xfail(reason='See https://github.com/pydantic/pydantic-ai/issues/3638')
    async def test_exhaustive_raises_unexpected_model_behavior(self):
        """Test that exhaustive strategy raises `UnexpectedModelBehavior` when all outputs have validation errors."""

        def process_output(output: OutputType) -> OutputType:  # pragma: no cover
            """A tool that should not be called."""
            assert False

        async def stream_function(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            # Missing 'value' field will cause validation error
            yield {1: DeltaToolCall('output_tool', '{"invalid_field": "invalid"}')}

        agent = Agent(
            FunctionModel(stream_function=stream_function),
            output_type=[
                ToolOutput(process_output, name='output_tool'),
            ],
            end_strategy='exhaustive',
        )

        with pytest.raises(UnexpectedModelBehavior, match='Exceeded maximum output retries \\(1\\)'):
            async with agent.run_stream('test') as result:
                await result.get_output()

    @pytest.mark.xfail(reason='See https://github.com/pydantic/pydantic-ai/issues/3638')
    async def test_multiple_final_result_are_validated_correctly(self):
        """Tests that if multiple final results are returned, but one fails validation, the other is used."""

        async def stream_function(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('final_result', '{"bad_value": "first"}')}
            yield {2: DeltaToolCall('final_result', '{"value": "second"}')}

        agent = Agent(FunctionModel(stream_function=stream_function), output_type=OutputType, end_strategy='early')

        async with agent.run_stream('test multiple final results') as result:
            response = await result.get_output()
            messages = result.new_messages()

        # Verify the result came from the second final tool
        assert response.value == snapshot('second')

        # Verify we got appropriate tool returns
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test multiple final results', timestamp=IsNow(tz=timezone.utc))],
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='final_result', args='{"bad_value": "first"}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='final_result', args='{"value": "second"}', tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=8),
                    model_name='function::stream_function',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content=[
                                ErrorDetails(
                                    type='missing',
                                    loc=('value',),
                                    msg='Field required',
                                    input={'bad_value': 'first'},
                                )
                            ],
                            tool_name='final_result',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        ),
                    ],
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_sequential_tool_is_a_per_tool_barrier(self):
        """A `sequential=True` tool runs alone; other tools parallelize around it (streaming path)."""
        active = 0
        barrier_ran_alone = True

        async def stream_function(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if len(messages) == 1:
                yield {0: DeltaToolCall(name='parallel_a')}
                yield {1: DeltaToolCall(name='parallel_b')}
                yield {2: DeltaToolCall(name='barrier')}
                yield {3: DeltaToolCall(name='parallel_c')}
            else:
                yield 'done'

        agent = Agent(FunctionModel(stream_function=stream_function))

        async def track() -> str:
            nonlocal active
            active += 1
            await asyncio.sleep(0.02)
            active -= 1
            return 'ok'

        @agent.tool_plain
        async def parallel_a() -> str:
            return await track()

        @agent.tool_plain
        async def parallel_b() -> str:
            return await track()

        @agent.tool_plain(sequential=True)
        async def barrier() -> str:
            nonlocal barrier_ran_alone
            if active != 0:
                barrier_ran_alone = False  # pragma: no cover
            await asyncio.sleep(0.02)
            return 'barrier'

        @agent.tool_plain
        async def parallel_c() -> str:
            return await track()

        async with agent.run_stream('test') as result:
            await result.get_output()

        assert barrier_ran_alone

    async def test_outer_cancellation_cancels_pending_tools(self):
        """Outer cancellation during streamed tool execution cancels still-pending tool tasks."""
        first_done = asyncio.Event()
        pending_started = asyncio.Event()
        pending_cancelled = asyncio.Event()

        async def stream_function(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if len(messages) == 1:
                yield {0: DeltaToolCall(name='fast_tool')}
                yield {1: DeltaToolCall(name='slow_tool')}
            else:
                yield 'done'  # pragma: no cover

        agent = Agent(FunctionModel(stream_function=stream_function))

        @agent.tool_plain
        async def fast_tool() -> str:
            first_done.set()
            return 'done'

        @agent.tool_plain
        async def slow_tool() -> str:
            pending_started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                pending_cancelled.set()
                raise
            return 'done'  # pragma: no cover

        async def run() -> None:
            async with agent.run_stream('test') as result:
                await result.get_output()  # pragma: no cover

        task = asyncio.create_task(run())
        await asyncio.wait_for(first_done.wait(), timeout=1)
        await asyncio.wait_for(pending_started.wait(), timeout=1)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert pending_cancelled.is_set()

    async def test_graceful_runs_function_tools_before_output(self):
        """Streaming commits the output as it streams, but `graceful` still runs the function tools
        the model emitted alongside it (their side effects happen)."""
        called: list[str] = []

        async def stream_function(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            assert info.output_tools is not None
            yield {0: DeltaToolCall(name='tool_a')}
            yield {1: DeltaToolCall(name='tool_b')}
            yield {2: DeltaToolCall('final_result', '{"value": "done"}')}

        agent = Agent(FunctionModel(stream_function=stream_function), output_type=OutputType, end_strategy='graceful')

        @agent.tool_plain
        def tool_a() -> str:
            called.append('tool_a')
            return 'a'

        @agent.tool_plain
        def tool_b() -> str:
            called.append('tool_b')
            return 'b'

        async with agent.run_stream('test') as result:
            output = await result.get_output()
        assert output.value == 'done'
        assert sorted(called) == ['tool_a', 'tool_b']

    async def test_graceful_interleaved_outputs_and_function_tools(self):
        """Graceful streaming with outputs and function tools interleaved: the first streamed output
        wins, later outputs are skipped, and the function tools still run."""
        called: list[str] = []

        async def stream_function(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            assert info.output_tools is not None
            yield {0: DeltaToolCall(name='tool_a')}
            yield {1: DeltaToolCall('first_output', '{"value": "a"}')}
            yield {2: DeltaToolCall(name='tool_b')}
            yield {3: DeltaToolCall('second_output', '{"value": "b"}')}

        agent = Agent(
            FunctionModel(stream_function=stream_function),
            output_type=[
                ToolOutput(OutputType, name='first_output'),
                ToolOutput(OutputType, name='second_output'),
            ],
            end_strategy='graceful',
        )

        @agent.tool_plain
        def tool_a() -> str:
            called.append('tool_a')
            return 'a'

        @agent.tool_plain
        def tool_b() -> str:
            called.append('tool_b')
            return 'b'

        async with agent.run_stream('test') as result:
            output = await result.get_output()
        assert output.value == 'a'
        assert sorted(called) == ['tool_a', 'tool_b']

    async def test_exhaustive_tool_output_sequential_barrier(self):
        """`ToolOutput(sequential=True)` under streaming: the output is committed as it streams, so
        (unlike the non-streaming path) it isn't held behind the function tool; the function tool
        still runs."""
        events: list[str] = []

        async def stream_function(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            assert info.output_tools is not None
            yield {0: DeltaToolCall(name='tool_a')}
            yield {1: DeltaToolCall('do_output', '{"value": "done"}')}

        def do_output(output: OutputType) -> OutputType:
            events.append('output')
            return output

        agent = Agent(
            FunctionModel(stream_function=stream_function),
            output_type=ToolOutput(do_output, name='do_output', sequential=True),
            end_strategy='exhaustive',
        )

        @agent.tool_plain
        async def tool_a() -> str:
            await asyncio.sleep(0.02)
            events.append('tool_a')
            return 'a'

        async with agent.run_stream('test') as result:
            output = await result.get_output()
        assert output.value == 'done'
        assert 'tool_a' in events

    async def test_early_output_failure_raises_when_streaming(self):
        """The non-streaming `early` fallback (run function tools when every output fails) has no
        streaming equivalent: a streamed output that fails validation raises, since `run_stream()`
        can't retry outputs."""

        async def stream_function(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            assert info.output_tools is not None
            yield {0: DeltaToolCall('regular_tool', '{"x": 1}')}
            yield {1: DeltaToolCall('bad_output', '{"value": "x"}')}

        def bad_output(output: OutputType) -> OutputType:
            if output.value == 'x':
                raise ModelRetry('bad')
            return output  # pragma: no cover

        agent = Agent(
            FunctionModel(stream_function=stream_function),
            output_type=ToolOutput(bad_output, name='bad_output'),
            end_strategy='early',
        )

        @agent.tool_plain
        def regular_tool(x: int) -> int:  # pragma: no cover
            return x

        with pytest.raises(UnexpectedModelBehavior, match='retries are not supported in `run_stream\\(\\)`'):
            async with agent.run_stream('test') as result:
                await result.get_output()

    async def test_early_multiple_outputs_and_function_tools(self):
        """Early streaming with several output tools: the first streamed output wins, later outputs
        are skipped, and function tools are stubbed (not run) once an output succeeds."""
        called: list[str] = []

        async def stream_function(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            assert info.output_tools is not None
            yield {0: DeltaToolCall('first_output', '{"value": "a"}')}
            yield {1: DeltaToolCall('second_output', '{"value": "b"}')}
            yield {2: DeltaToolCall('regular_tool', '{"x": 1}')}

        agent = Agent(
            FunctionModel(stream_function=stream_function),
            output_type=[
                ToolOutput(OutputType, name='first_output'),
                ToolOutput(OutputType, name='second_output'),
            ],
            end_strategy='early',
        )

        @agent.tool_plain
        def regular_tool(x: int) -> int:  # pragma: no cover
            called.append('regular_tool')
            return x

        async with agent.run_stream('test') as result:
            output = await result.get_output()
        assert output.value == 'a'
        assert called == []

    async def test_graceful_function_tool_retry_does_not_suppress_committed_output(self):
        """Retry-wins doesn't apply when streaming: the output is committed as it streams, so a
        function tool's `ModelRetry` in the same response can't revoke it (`graceful`)."""
        rounds = 0

        async def stream_function(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            nonlocal rounds
            assert info.output_tools is not None
            rounds += 1
            yield {0: DeltaToolCall('flaky_tool', '{"x": 1}')}
            yield {1: DeltaToolCall('final_result', '{"value": "committed"}')}

        agent = Agent(FunctionModel(stream_function=stream_function), output_type=OutputType, end_strategy='graceful')

        @agent.tool_plain
        def flaky_tool(x: int) -> int:
            raise ModelRetry('not yet')

        async with agent.run_stream('test') as result:
            output = await result.get_output()
        # The streamed output is committed and not suppressed, so the run ends in a single round.
        assert output.value == 'committed'
        assert rounds == 1

    async def test_exhaustive_function_tool_retry_does_not_suppress_committed_output(self):
        """Retry-wins is also exempt under `exhaustive` streaming: the committed output stands."""
        rounds = 0

        async def stream_function(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            nonlocal rounds
            assert info.output_tools is not None
            rounds += 1
            yield {0: DeltaToolCall('flaky_tool', '{"x": 1}')}
            yield {1: DeltaToolCall('final_result', '{"value": "committed"}')}

        agent = Agent(FunctionModel(stream_function=stream_function), output_type=OutputType, end_strategy='exhaustive')

        @agent.tool_plain
        def flaky_tool(x: int) -> int:
            raise ModelRetry('not yet')

        async with agent.run_stream('test') as result:
            output = await result.get_output()
        assert output.value == 'committed'
        assert rounds == 1

    # NOTE: When changing tests in this class:
    # 1. Follow the existing order
    # 2. Update tests in `tests/test_agent.py::TestMultipleToolCalls` as well
    # The retry-wins tests (a function-tool `ModelRetry` suppressing an output result) have no
    # streaming counterpart: under `run_stream` the streamed output is committed as soon as it's
    # detected, so retry-wins doesn't apply (see `docs/output.md`).


async def test_custom_output_type_default_str() -> None:
    agent = Agent('test')

    async with agent.run_stream('test') as result:
        response = await result.get_output()
        assert response == snapshot('success (no tool calls)')
    assert result.response == snapshot(
        ModelResponse(
            parts=[TextPart(content='success (no tool calls)')],
            usage=RequestUsage(input_tokens=51, output_tokens=4),
            model_name='test',
            timestamp=IsDatetime(),
            provider_name='test',
        )
    )

    async with agent.run_stream('test', output_type=OutputType) as result:
        response = await result.get_output()
        assert response == snapshot(OutputType(value='a'))


async def test_custom_output_type_default_structured() -> None:
    agent = Agent('test', output_type=OutputType)

    async with agent.run_stream('test') as result:
        response = await result.get_output()
        assert response == snapshot(OutputType(value='a'))

    async with agent.run_stream('test', output_type=str) as result:
        response = await result.get_output()
        assert response == snapshot('success (no tool calls)')


async def test_iter_stream_output():
    m = TestModel(custom_output_text='The cat sat on the mat.')

    agent = Agent(m)

    @agent.output_validator
    def output_validator_simple(data: str) -> str:
        # Make a substitution in the validated results
        return re.sub('cat sat', 'bat sat', data)

    run: AgentRun
    stream: AgentStream | None = None
    messages: list[str] = []

    stream_usage: RunUsage | None = None
    async with agent.iter('Hello') as run:
        async for node in run:
            if agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for chunk in stream.stream_output(debounce_by=None):
                        messages.append(chunk)
                stream_usage = deepcopy(stream.usage)
    assert stream is not None
    assert stream.response == snapshot(
        ModelResponse(
            parts=[TextPart(content='The cat sat on the mat.')],
            usage=RequestUsage(input_tokens=51, output_tokens=7),
            model_name='test',
            timestamp=IsDatetime(),
            provider_name='test',
        )
    )
    assert run.next_node == End(data=FinalResult(output='The bat sat on the mat.', tool_name=None, tool_call_id=None))
    assert run.usage == stream_usage == RunUsage(requests=1, input_tokens=51, output_tokens=7)

    assert messages == snapshot(
        [
            '',
            'The ',
            'The cat ',
            'The bat sat ',
            'The bat sat on ',
            'The bat sat on the ',
            'The bat sat on the mat.',
            'The bat sat on the mat.',
        ]
    )


async def test_streamed_run_result_metadata_available() -> None:
    agent = Agent(TestModel(custom_output_text='stream metadata'), metadata={'env': 'stream'})

    async with agent.run_stream('stream metadata prompt') as result:
        assert await result.get_output() == 'stream metadata'

    assert result.metadata == {'env': 'stream'}


async def test_agent_stream_metadata_available() -> None:
    agent = Agent(
        TestModel(custom_output_text='agent stream metadata'),
        metadata=lambda ctx: {'prompt': ctx.prompt},
    )

    captured_stream: AgentStream | None = None
    async with agent.iter('agent stream prompt') as run:
        async for node in run:
            if agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    captured_stream = stream
                    async for _ in stream.stream_text(debounce_by=None):
                        pass

    assert captured_stream is not None
    assert captured_stream.metadata == {'prompt': 'agent stream prompt'}


def test_agent_stream_metadata_falls_back_to_run_context() -> None:
    response_message = ModelResponse(parts=[TextPart('fallback metadata')], model_name='test')
    stream_response = ModelTestStreamedResponse(
        model_request_parameters=models.ModelRequestParameters(),
        _model_name='test',
        _structured_response=response_message,
        _messages=[],
        _provider_name='test',
    )
    run_ctx = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        metadata={'source': 'run-context'},
    )
    output_schema = TextOutputSchema[str](
        text_processor=TextOutputProcessor(),
        allows_deferred_tools=False,
        allows_image=False,
        allows_none=False,
    )
    stream = AgentStream(
        _raw_stream_response=stream_response,
        _output_schema=output_schema,
        _model_request_parameters=models.ModelRequestParameters(),
        _output_validators=[],
        _run_ctx=run_ctx,
        _usage_limits=None,
        _tool_manager=ToolManager(toolset=MagicMock()),
        _root_capability=CombinedCapability([]),
    )

    assert stream.metadata == {'source': 'run-context'}


@pytest.mark.parametrize(
    ('leading_text', 'trailing_text', 'provider_metadata', 'expected'),
    [
        pytest.param('pre-tool text', None, False, '', id='trailing-native-pair-resets-prior-text'),
        pytest.param('final answer', None, True, 'final answer', id='trailing-provider-metadata-preserves-prior-text'),
        pytest.param(
            'pre-tool text', 'final answer', True, 'final answer', id='later-text-resets-before-provider-metadata'
        ),
    ],
)
async def test_agent_stream_text_output_with_native_tool_parts(
    leading_text: str, trailing_text: str | None, provider_metadata: bool, expected: str
) -> None:
    parts: list[ModelResponsePart] = [
        TextPart(leading_text),
        NativeToolCallPart(
            tool_name='web_search',
            args={'queries': ['query']},
            tool_call_id='web-search-call',
            provider_name='test',
        ),
        NativeToolReturnPart(
            tool_name='web_search',
            content=[{'uri': 'https://example.com', 'title': 'Example'}],
            tool_call_id='web-search-call',
            provider_name='test',
        ),
    ]
    if trailing_text is not None:
        parts.append(TextPart(trailing_text))
    response = ModelResponse(parts=parts, model_name='test')
    if provider_metadata:
        response.metadata = _utils.add_provider_metadata_tool_call_id(response.metadata, 'web-search-call')
        [round_tripped] = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json([response]))
        response = cast(ModelResponse, round_tripped)

    assert await _make_text_output_agent_stream(response).validate_response_output(response) == expected


def _make_text_output_agent_stream(response: ModelResponse) -> AgentStream[None, str]:
    stream_response = ModelTestStreamedResponse(
        model_request_parameters=models.ModelRequestParameters(),
        _model_name='test',
        _structured_response=response,
        _messages=[],
        _provider_name='test',
    )
    stream_response.final_result_event = FinalResultEvent(tool_name=None, tool_call_id=None)
    output_schema = TextOutputSchema[str](
        text_processor=TextOutputProcessor(),
        allows_deferred_tools=False,
        allows_image=False,
        allows_none=False,
    )
    return AgentStream(
        _raw_stream_response=stream_response,
        _output_schema=output_schema,
        _model_request_parameters=models.ModelRequestParameters(),
        _output_validators=[],
        _run_ctx=RunContext(deps=None, model=TestModel(), usage=RunUsage()),
        _usage_limits=None,
        _tool_manager=ToolManager(toolset=MagicMock()),
        _root_capability=CombinedCapability([]),
    )


def _native_pair_parts(n: int) -> list[BuiltinToolCallsReturns]:
    tool_return = NativeToolReturnPart(
        tool_name='web_search',
        content=[{'uri': 'https://example.com', 'title': 'Example'}],
        tool_call_id=f'web-search-call-{n}',
        provider_name='function',
    )
    return [
        {
            2 * n: NativeToolCallPart(
                tool_name='web_search',
                args={'queries': ['query']},
                tool_call_id=f'web-search-call-{n}',
                provider_name='function',
            )
        },
        {2 * n + 1: tool_return},
    ]


async def test_agent_stream_text_separator_when_text_resumes_as_deltas() -> None:
    """Providers that stream text under a constant vendor ID (e.g. xAI, Groq, `FunctionModel`) resume
    the pre-tool `TextPart` as deltas after a native tool pair rather than starting a new part, so the
    deferred separator must be flushed on the delta branch too, not only on a new `TextPart` start.
    """

    async def stream_function(
        _messages: list[ModelMessage], _info: AgentInfo
    ) -> AsyncIterator[str | BuiltinToolCallsReturns]:
        yield 'first'
        for part in _native_pair_parts(1):
            yield part
        yield 'second'
        for part in _native_pair_parts(2):
            yield part
        yield 'third'

    agent = Agent(FunctionModel(stream_function=stream_function))

    async with agent.run_stream('hello') as result:
        deltas = [text async for text in result.stream_text(delta=True, debounce_by=None)]

    assert deltas == snapshot(['first', '\n\n', 'second', '\n\n', 'third'])


async def test_agent_does_not_treat_text_before_trailing_native_pair_as_output() -> None:
    async def function(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    TextPart('Let me search.'),
                    NativeToolCallPart(
                        tool_name='web_search',
                        args={'queries': ['query']},
                        tool_call_id='web-search-call',
                        provider_name='function',
                    ),
                    NativeToolReturnPart(
                        tool_name='web_search',
                        content=[{'uri': 'https://example.com', 'title': 'Example'}],
                        tool_call_id='web-search-call',
                        provider_name='function',
                    ),
                ]
            )
        return ModelResponse(parts=[TextPart('final answer')])

    result = await Agent(FunctionModel(function=function)).run('hello')

    assert result.output == 'final answer'
    assert result.usage.requests == 2


def _make_run_result(*, metadata: dict[str, Any] | None) -> AgentRunResult[str]:
    state = GraphAgentState(metadata=metadata)
    response_message = ModelResponse(parts=[TextPart('final')], model_name='test')
    state.message_history.append(response_message)
    return AgentRunResult('final', _state=state)


def test_streamed_run_result_metadata_prefers_run_result_state() -> None:
    run_result = _make_run_result(metadata={'from': 'run-result'})
    streamed = StreamedRunResult(
        all_messages=run_result.all_messages(),
        new_message_index=0,
        run_result=run_result,
    )
    assert streamed.metadata == {'from': 'run-result'}


def test_streamed_run_result_metadata_none_without_sources() -> None:
    run_result = _make_run_result(metadata=None)
    streamed = StreamedRunResult(all_messages=[], new_message_index=0, run_result=run_result)
    assert streamed.metadata is None


def test_streamed_run_result_metadata_none_without_run_or_stream() -> None:
    streamed = StreamedRunResult(all_messages=[], new_message_index=0, stream_response=None, on_complete=None)
    assert streamed.metadata is None


def test_streamed_run_result_sync_exposes_metadata() -> None:
    run_result = _make_run_result(metadata={'sync': 'metadata'})
    streamed = StreamedRunResult(
        all_messages=run_result.all_messages(),
        new_message_index=0,
        run_result=run_result,
    )

    @asynccontextmanager
    async def run_stream_cm() -> AsyncGenerator[StreamedRunResult[None, str]]:
        yield streamed

    with StreamedRunResultSync(run_stream_cm()) as sync_result:
        assert sync_result.metadata == {'sync': 'metadata'}


async def test_iter_stream_response():
    m = TestModel(custom_output_text='The cat sat on the mat.')

    agent = Agent(m)

    @agent.output_validator
    def output_validator_simple(data: str) -> str:
        # Make a substitution in the validated results
        return re.sub('cat sat', 'bat sat', data)

    run: AgentRun
    stream: AgentStream
    messages: list[ModelResponse] = []
    async with agent.iter('Hello') as run:
        assert isinstance(run.run_id, str)
        async for node in run:
            if agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for chunk in stream.stream_response(debounce_by=None):
                        messages.append(chunk)

    incomplete_texts = [
        '',
        '',
        'The ',
        'The cat ',
        'The cat sat ',
        'The cat sat on ',
        'The cat sat on the ',
        'The cat sat on the mat.',
        'The cat sat on the mat.',
    ]
    assert messages == [
        *(
            ModelResponse(
                parts=[TextPart(content=text)],
                usage=RequestUsage(input_tokens=IsInt(), output_tokens=IsInt()),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='test',
                state='incomplete',
            )
            for text in incomplete_texts
        ),
        ModelResponse(
            parts=[TextPart(content='The cat sat on the mat.')],
            usage=RequestUsage(input_tokens=IsInt(), output_tokens=IsInt()),
            model_name='test',
            timestamp=IsNow(tz=timezone.utc),
            provider_name='test',
        ),
    ]

    # Note: as you can see above, the output validator is not applied to the streamed responses, just the final result:
    assert run.result is not None
    assert run.result.output == 'The bat sat on the mat.'


async def test_stream_iter_structured_validator() -> None:
    class NotOutputType(BaseModel):
        not_value: str

    agent = Agent[object, OutputType | NotOutputType]('test', output_type=OutputType | NotOutputType)

    @agent.output_validator
    def output_validator(data: OutputType | NotOutputType) -> OutputType | NotOutputType:
        assert isinstance(data, OutputType)
        return OutputType(value=data.value + ' (validated)')

    outputs: list[OutputType] = []
    async with agent.iter('test') as run:
        async for node in run:
            if agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for output in stream.stream_output(debounce_by=None):
                        outputs.append(output)
    assert outputs == snapshot([OutputType(value='a (validated)'), OutputType(value='a (validated)')])


async def test_unknown_tool_call_events():
    """Test that unknown tool calls emit both FunctionToolCallEvent and FunctionToolResultEvent during streaming."""

    def call_mixed_tools(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        """Mock function that calls both known and unknown tools."""
        return ModelResponse(
            parts=[
                ToolCallPart('unknown_tool', {'arg': 'value'}),
                ToolCallPart('known_tool', {'x': 5}),
            ]
        )

    agent = Agent(FunctionModel(call_mixed_tools))

    @agent.tool_plain
    def known_tool(x: int) -> int:
        return x * 2

    event_parts: list[Any] = []

    try:
        async with agent.iter('test') as agent_run:
            async for node in agent_run:  # pragma: no branch
                if Agent.is_call_tools_node(node):
                    async with node.stream(agent_run.ctx) as event_stream:
                        async for event in event_stream:
                            event_parts.append(event)

    except UnexpectedModelBehavior:
        pass

    assert event_parts == snapshot(
        [
            FunctionToolCallEvent(
                part=ToolCallPart(
                    tool_name='unknown_tool',
                    args={'arg': 'value'},
                    tool_call_id=IsStr(),
                ),
                args_valid=False,
            ),
            FunctionToolCallEvent(
                part=ToolCallPart(
                    tool_name='known_tool',
                    args={'x': 5},
                    tool_call_id=IsStr(),
                ),
                args_valid=True,
            ),
            FunctionToolResultEvent(
                part=RetryPromptPart(
                    content="Unknown tool name: 'unknown_tool'. Available tools: 'known_tool'",
                    tool_name='unknown_tool',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ),
            FunctionToolResultEvent(
                part=ToolReturnPart(
                    tool_name='known_tool',
                    content=10,
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ),
            FunctionToolCallEvent(
                part=ToolCallPart(
                    tool_name='unknown_tool',
                    args={'arg': 'value'},
                    tool_call_id=IsStr(),
                ),
                args_valid=False,
            ),
        ]
    )


async def test_output_tool_success_events():
    """Test that a successful output tool call emits `OutputToolCallEvent` and `OutputToolResultEvent`."""

    def call_final_result(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        return ModelResponse(parts=[ToolCallPart('final_result', {'value': 'hello'})])

    agent = Agent(FunctionModel(call_final_result), output_type=OutputType)

    events: list[Any] = []
    async with agent.iter('test') as agent_run:
        async for node in agent_run:
            if Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as event_stream:
                    async for event in event_stream:
                        events.append(event)

    assert agent_run.result is not None
    assert agent_run.result.output == snapshot(OutputType(value='hello'))

    assert events == snapshot(
        [
            OutputToolCallEvent(
                part=ToolCallPart(
                    tool_name='final_result',
                    args={'value': 'hello'},
                    tool_call_id=IsStr(),
                ),
                args_valid=True,
            ),
            OutputToolResultEvent(
                part=ToolReturnPart(
                    tool_name='final_result',
                    content='Final result processed.',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                )
            ),
        ]
    )


async def test_output_tool_events():
    """Test that output tools emit events during streaming for both validation failure and success."""

    def call_final_result_with_bad_data(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        """Mock function that calls final_result tool with invalid data."""
        assert info.output_tools is not None
        return ModelResponse(
            parts=[
                ToolCallPart('final_result', {'bad_value': 'invalid'}),  # Invalid field name
                ToolCallPart('final_result', {'value': 'valid'}),  # Valid field name
            ]
        )

    agent = Agent(FunctionModel(call_final_result_with_bad_data), output_type=OutputType)

    events: list[Any] = []
    async with agent.iter('test') as agent_run:
        async for node in agent_run:
            if Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as event_stream:
                    async for event in event_stream:
                        events.append(event)

    assert events == snapshot(
        [
            OutputToolCallEvent(
                part=ToolCallPart(
                    tool_name='final_result',
                    args={'bad_value': 'invalid'},
                    tool_call_id=IsStr(),
                ),
                args_valid=False,
            ),
            OutputToolResultEvent(
                part=RetryPromptPart(
                    content=[
                        ErrorDetails(
                            type='missing',
                            loc=('value',),
                            msg='Field required',
                            input={'bad_value': 'invalid'},
                        ),
                    ],
                    tool_name='final_result',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                )
            ),
            OutputToolCallEvent(
                part=ToolCallPart(
                    tool_name='final_result',
                    args={'value': 'valid'},
                    tool_call_id=IsStr(),
                ),
                args_valid=True,
            ),
            OutputToolResultEvent(
                part=ToolReturnPart(
                    tool_name='final_result',
                    content='Final result processed.',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                )
            ),
        ]
    )


def _tool_call_and_return_ids_from_messages(messages: list[ModelMessage]) -> tuple[set[str], set[str]]:
    call_ids: set[str] = set()
    return_ids: set[str] = set()
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                call_ids.add(part.tool_call_id)
            elif isinstance(part, ToolReturnPart):
                return_ids.add(part.tool_call_id)
    return call_ids, return_ids


async def test_output_tool_event_history_has_no_dangling_call():
    """Regression test for #2640: event-reconstructed history should not have a dangling output tool call.

    Every `OutputToolCallEvent` seen on the event stream should have a matching
    `OutputToolResultEvent`, and the tool_call_ids should match those in `all_messages()`.
    """

    def call_final_result(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        return ModelResponse(parts=[ToolCallPart('final_result', {'value': 'hello'})])

    agent = Agent(FunctionModel(call_final_result), output_type=OutputType)

    events: list[Any] = []
    async with agent.iter('test') as agent_run:
        async for node in agent_run:
            if Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as handle_stream:
                    async for event in handle_stream:
                        events.append(event)

    call_ids_from_events = {e.part.tool_call_id for e in events if isinstance(e, OutputToolCallEvent)}
    return_ids_from_events = {e.part.tool_call_id for e in events if isinstance(e, OutputToolResultEvent)}

    # No dangling calls: every call seen on the event stream has a matching result.
    assert call_ids_from_events == return_ids_from_events
    assert None not in call_ids_from_events

    # And the event-stream view matches `all_messages()`.
    assert agent_run.result is not None
    call_ids_from_messages, return_ids_from_messages = _tool_call_and_return_ids_from_messages(
        agent_run.result.all_messages()
    )
    assert call_ids_from_events == call_ids_from_messages
    assert return_ids_from_events == return_ids_from_messages


async def test_output_function_model_retry_in_stream():
    """`ModelRetry` from a `ToolOutput` function during `run_stream()` must surface as
    `UnexpectedModelBehavior` (caused by `ModelRetry`), not propagate as `ToolRetryError`.

    Regression test for an earlier version of `ToolManager.execute_output_tool_call` that
    unconditionally wrapped `ModelRetry` from the output function as `ToolRetryError`,
    which `result.validate_response_output` doesn't catch in the streaming path.
    """

    async def stream_final_result(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        assert info.output_tools is not None
        yield {0: DeltaToolCall(name='final_result', json_args='{"value": "anything"}')}

    def reject(value: str) -> str:
        raise ModelRetry('please try again')

    agent = Agent(
        FunctionModel(stream_function=stream_final_result),
        output_type=ToolOutput(reject, name='final_result'),
        retries={'output': 0},
    )

    with pytest.raises(UnexpectedModelBehavior) as exc_info:
        async with agent.run_stream('test') as result:
            await result.get_output()

    # The cause must be ModelRetry, not ToolRetryError — `validate_response_output`
    # only catches `(ValidationError, ModelRetry)` in the streaming path.
    assert isinstance(exc_info.value.__cause__, ModelRetry)


async def test_stream_structured_output():
    class CityLocation(BaseModel):
        city: str
        country: str | None = None

    m = TestModel(custom_output_text='{"city": "Mexico City", "country": "Mexico"}')

    agent = Agent(m, output_type=PromptedOutput(CityLocation))

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_output(debounce_by=None)] == snapshot(
            [
                CityLocation(city='Mexico '),
                CityLocation(city='Mexico City'),
                CityLocation(city='Mexico City'),
                CityLocation(city='Mexico City', country='Mexico'),
                CityLocation(city='Mexico City', country='Mexico'),
            ]
        )
        assert result.is_complete


async def test_iter_stream_structured_output():
    class CityLocation(BaseModel):
        city: str
        country: str | None = None

    m = TestModel(custom_output_text='{"city": "Mexico City", "country": "Mexico"}')

    agent = Agent(m, output_type=PromptedOutput(CityLocation))

    async with agent.iter('') as run:
        async for node in run:
            if agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    assert [c async for c in stream.stream_output(debounce_by=None)] == snapshot(
                        [
                            CityLocation(city='Mexico '),
                            CityLocation(city='Mexico City'),
                            CityLocation(city='Mexico City'),
                            CityLocation(city='Mexico City', country='Mexico'),
                            CityLocation(city='Mexico City', country='Mexico'),
                        ]
                    )


async def test_iter_stream_output_tool_dont_hit_retry_limit():
    class CityLocation(BaseModel):
        city: str
        country: str | None = None

    async def text_stream(_messages: list[ModelMessage], agent_info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        """Stream partial JSON data that will initially fail validation."""
        assert agent_info.output_tools is not None
        assert len(agent_info.output_tools) == 1
        name = agent_info.output_tools[0].name

        yield {0: DeltaToolCall(name=name)}
        yield {0: DeltaToolCall(json_args='{"c')}
        yield {0: DeltaToolCall(json_args='ity":')}
        yield {0: DeltaToolCall(json_args=' "Mex')}
        yield {0: DeltaToolCall(json_args='ico City",')}
        yield {0: DeltaToolCall(json_args=' "cou')}
        yield {0: DeltaToolCall(json_args='ntry": "Mexico"}')}

    agent = Agent(FunctionModel(stream_function=text_stream), output_type=CityLocation)

    async with agent.iter('Generate city info') as run:
        async for node in run:
            if agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    assert [c async for c in stream.stream_output(debounce_by=None)] == snapshot(
                        [
                            CityLocation(city='Mex'),
                            CityLocation(city='Mexico City'),
                            CityLocation(city='Mexico City'),
                            CityLocation(city='Mexico City', country='Mexico'),
                            CityLocation(city='Mexico City', country='Mexico'),
                        ]
                    )


def test_function_tool_event_tool_call_id_properties():
    """Ensure that the `tool_call_id` property on function tool events mirrors the underlying part's ID."""
    # Prepare a ToolCallPart with a fixed ID
    call_part = ToolCallPart(tool_name='sample_tool', args={'a': 1}, tool_call_id='call_id_123')
    call_event = FunctionToolCallEvent(part=call_part, args_valid=True)

    # The event should expose the same `tool_call_id` as the part
    assert call_event.tool_call_id == call_part.tool_call_id == 'call_id_123'

    # Prepare a ToolReturnPart with a fixed ID
    return_part = ToolReturnPart(tool_name='sample_tool', content='ok', tool_call_id='return_id_456')
    result_event = FunctionToolResultEvent(part=return_part)

    # The event should expose the same `tool_call_id` as the result part
    assert result_event.tool_call_id == return_part.tool_call_id == 'return_id_456'


async def test_tool_raises_call_deferred():
    agent = Agent(TestModel(), output_type=[str, DeferredToolRequests])

    @agent.tool_plain()
    def my_tool(x: int) -> int:
        raise CallDeferred

    async with agent.run_stream('Hello') as result:
        assert not result.is_complete
        assert isinstance(result.run_id, str)
        assert isinstance(result.conversation_id, str)
        assert [c async for c in result.stream_output(debounce_by=None)] == snapshot(
            [DeferredToolRequests(calls=[ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id=IsStr())])]
        )
        assert await result.get_output() == snapshot(
            DeferredToolRequests(calls=[ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id=IsStr())])
        )
        responses = [c async for c in result.stream_response(debounce_by=None)]
        assert responses == snapshot(
            [
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id=IsStr())],
                    usage=RequestUsage(input_tokens=51),
                    model_name='test',
                    timestamp=IsDatetime(),
                    provider_name='test',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                )
            ]
        )
        assert await result.validate_response_output(responses[0]) == snapshot(
            DeferredToolRequests(calls=[ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id=IsStr())])
        )
        assert result.usage == snapshot(RunUsage(requests=1, input_tokens=51, output_tokens=0))
        assert result.timestamp == IsNow(tz=timezone.utc)
        assert result.is_complete


async def test_tool_raises_approval_required():
    async def llm(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) == 1:
            yield {0: DeltaToolCall(name='my_tool', json_args='{"x": 1}', tool_call_id='my_tool')}
        else:
            yield 'Done!'

    agent = Agent(FunctionModel(stream_function=llm), output_type=[str, DeferredToolRequests])

    @agent.tool
    def my_tool(ctx: RunContext, x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return x * 42

    async with agent.run_stream('Hello') as result:
        assert not result.is_complete
        messages = result.all_messages()
        output = await result.get_output()
        assert output == snapshot(
            DeferredToolRequests(approvals=[ToolCallPart(tool_name='my_tool', args='{"x": 1}', tool_call_id=IsStr())])
        )
        assert result.is_complete

    async with agent.run_stream(
        message_history=messages,
        deferred_tool_results=DeferredToolResults(approvals={'my_tool': ToolApproved(override_args={'x': 2})}),
    ) as result:
        assert not result.is_complete
        output = await result.get_output()
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Hello',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{"x": 1}', tool_call_id='my_tool')],
                    usage=RequestUsage(input_tokens=50, output_tokens=3),
                    model_name='function::llm',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='my_tool',
                            content=84,
                            tool_call_id='my_tool',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='Done!')],
                    usage=RequestUsage(input_tokens=50, output_tokens=1),
                    model_name='function::llm',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )
        assert output == snapshot('Done!')
        assert result.is_complete


async def test_deferred_tool_iter():
    agent = Agent(TestModel(), output_type=[str, DeferredToolRequests])

    async def prepare_tool(ctx: RunContext, tool_def: ToolDefinition) -> ToolDefinition:
        return replace(tool_def, kind='external')

    @agent.tool_plain(prepare=prepare_tool)
    def my_tool(x: int) -> int:
        return x + 1  # pragma: no cover

    @agent.tool_plain(requires_approval=True)
    def my_other_tool(x: int) -> int:
        return x + 1  # pragma: no cover

    outputs: list[str | DeferredToolRequests] = []
    events: list[Any] = []

    async with agent.iter('test') as run:
        async for node in run:
            if agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        events.append(event)
                    async for output in stream.stream_output(debounce_by=None):
                        outputs.append(output)
            if agent.is_call_tools_node(node):
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        events.append(event)

    assert outputs == snapshot(
        [
            DeferredToolRequests(
                calls=[ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id=IsStr())],
                approvals=[ToolCallPart(tool_name='my_other_tool', args={'x': 0}, tool_call_id=IsStr())],
            ),
            DeferredToolRequests(
                calls=[ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id='pyd_ai_tool_call_id__my_tool')],
                approvals=[
                    ToolCallPart(
                        tool_name='my_other_tool', args={'x': 0}, tool_call_id='pyd_ai_tool_call_id__my_other_tool'
                    )
                ],
            ),
        ]
    )
    assert events == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id=IsStr()),
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartEndEvent(
                index=0,
                part=ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id='pyd_ai_tool_call_id__my_tool'),
                next_part_kind='tool-call',
            ),
            PartStartEvent(
                index=1,
                part=ToolCallPart(
                    tool_name='my_other_tool', args={'x': 0}, tool_call_id='pyd_ai_tool_call_id__my_other_tool'
                ),
                previous_part_kind='tool-call',
            ),
            PartEndEvent(
                index=1,
                part=ToolCallPart(
                    tool_name='my_other_tool', args={'x': 0}, tool_call_id='pyd_ai_tool_call_id__my_other_tool'
                ),
            ),
            FunctionToolCallEvent(
                part=ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id=IsStr()), args_valid=True
            ),
            FunctionToolCallEvent(
                part=ToolCallPart(tool_name='my_other_tool', args={'x': 0}, tool_call_id=IsStr()), args_valid=True
            ),
        ]
    )


async def test_tool_raises_call_deferred_approval_required_iter():
    agent = Agent(TestModel(), output_type=[str, DeferredToolRequests])

    @agent.tool_plain
    def my_tool(x: int) -> int:
        raise CallDeferred

    @agent.tool_plain
    def my_other_tool(x: int) -> int:
        raise ApprovalRequired

    events: list[Any] = []

    async with agent.iter('test') as run:
        async for node in run:
            if agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        events.append(event)
            if agent.is_call_tools_node(node):
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        events.append(event)

    assert events == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id=IsStr()),
            ),
            PartEndEvent(
                index=0,
                part=ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id='pyd_ai_tool_call_id__my_tool'),
                next_part_kind='tool-call',
            ),
            PartStartEvent(
                index=1,
                part=ToolCallPart(
                    tool_name='my_other_tool', args={'x': 0}, tool_call_id='pyd_ai_tool_call_id__my_other_tool'
                ),
                previous_part_kind='tool-call',
            ),
            PartEndEvent(
                index=1,
                part=ToolCallPart(
                    tool_name='my_other_tool', args={'x': 0}, tool_call_id='pyd_ai_tool_call_id__my_other_tool'
                ),
            ),
            FunctionToolCallEvent(
                part=ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id=IsStr()), args_valid=True
            ),
            FunctionToolCallEvent(
                part=ToolCallPart(tool_name='my_other_tool', args={'x': 0}, tool_call_id=IsStr()), args_valid=True
            ),
        ]
    )

    assert run.result is not None
    assert run.result.output == snapshot(
        DeferredToolRequests(
            calls=[ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id=IsStr())],
            approvals=[ToolCallPart(tool_name='my_other_tool', args={'x': 0}, tool_call_id=IsStr())],
        )
    )


async def test_run_event_stream_handler():
    m = TestModel()

    test_agent = Agent(m)
    assert test_agent.name is None

    @test_agent.tool_plain
    async def ret_a(x: str) -> str:
        return f'{x}-apple'

    events: list[AgentStreamEvent] = []

    async def event_stream_handler(ctx: RunContext, stream: AsyncIterable[AgentStreamEvent]):
        async for event in stream:
            events.append(event)

    result = await test_agent.run('Hello', event_stream_handler=event_stream_handler)
    assert result.output == snapshot('{"ret_a":"a-apple"}')
    assert events == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr()),
            ),
            PartEndEvent(
                index=0,
                part=ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id='pyd_ai_tool_call_id__ret_a'),
            ),
            FunctionToolCallEvent(
                part=ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr()), args_valid=True
            ),
            FunctionToolResultEvent(
                part=ToolReturnPart(
                    tool_name='ret_a',
                    content='a-apple',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                )
            ),
            PartStartEvent(index=0, part=TextPart(content='')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='{"ret_a":')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='"a-apple"}')),
            PartEndEvent(index=0, part=TextPart(content='{"ret_a":"a-apple"}')),
        ]
    )


async def test_event_stream_handler_propagates_tool_error():
    """When a tool raises during streaming with event_stream_handler and the error
    is suppressed by the handler, the _stream_error re-raise path in run() should
    propagate the original error — not an internal AssertionError about _next_node."""

    m = TestModel()
    test_agent = Agent(m)

    @test_agent.tool_plain
    async def failing_tool(x: str) -> str:
        raise RuntimeError('tool execution failed')

    events: list[AgentStreamEvent] = []

    async def handler(ctx: RunContext, stream: AsyncIterable[AgentStreamEvent]):
        # Suppress the error to simulate UIEventStream.transform_stream behavior,
        # which catches exceptions and doesn't re-raise them.
        try:
            async for event in stream:
                events.append(event)
        except RuntimeError:
            pass

    with pytest.raises(RuntimeError, match='tool execution failed'):
        await test_agent.run('Hello', event_stream_handler=handler)

    # Events up to the tool call should still have been emitted
    assert any(isinstance(e, FunctionToolCallEvent) for e in events)


def test_run_sync_event_stream_handler():
    m = TestModel()

    test_agent = Agent(m)
    assert test_agent.name is None

    @test_agent.tool_plain
    async def ret_a(x: str) -> str:
        return f'{x}-apple'

    events: list[AgentStreamEvent] = []

    async def event_stream_handler(ctx: RunContext, stream: AsyncIterable[AgentStreamEvent]):
        async for event in stream:
            events.append(event)

    result = test_agent.run_sync('Hello', event_stream_handler=event_stream_handler)
    assert result.output == snapshot('{"ret_a":"a-apple"}')
    assert events == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr()),
            ),
            PartEndEvent(
                index=0,
                part=ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id='pyd_ai_tool_call_id__ret_a'),
            ),
            FunctionToolCallEvent(
                part=ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr()), args_valid=True
            ),
            FunctionToolResultEvent(
                part=ToolReturnPart(
                    tool_name='ret_a',
                    content='a-apple',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                )
            ),
            PartStartEvent(index=0, part=TextPart(content='')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='{"ret_a":')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='"a-apple"}')),
            PartEndEvent(index=0, part=TextPart(content='{"ret_a":"a-apple"}')),
        ]
    )


async def test_run_stream_event_stream_handler():
    m = TestModel()

    test_agent = Agent(m)
    assert test_agent.name is None

    @test_agent.tool_plain
    async def ret_a(x: str) -> str:
        return f'{x}-apple'

    events: list[AgentStreamEvent] = []

    async def event_stream_handler(ctx: RunContext, stream: AsyncIterable[AgentStreamEvent]):
        async for event in stream:
            events.append(event)

    async with test_agent.run_stream('Hello', event_stream_handler=event_stream_handler) as result:
        assert [c async for c in result.stream_output(debounce_by=None)] == snapshot(
            ['{"ret_a":', '{"ret_a":"a-apple"}', '{"ret_a":"a-apple"}']
        )

    assert events == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr()),
            ),
            PartEndEvent(
                index=0,
                part=ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id='pyd_ai_tool_call_id__ret_a'),
            ),
            FunctionToolCallEvent(
                part=ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr()), args_valid=True
            ),
            FunctionToolResultEvent(
                part=ToolReturnPart(
                    tool_name='ret_a',
                    content='a-apple',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                )
            ),
            PartStartEvent(index=0, part=TextPart(content='')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
        ]
    )


async def test_run_event_stream_handler_does_not_need_to_consume_stream():
    agent = Agent(TestModel(custom_output_text='hello world this is a long answer'))

    async def event_stream_handler(ctx: RunContext, stream: AsyncIterable[AgentStreamEvent]) -> None:
        return  # never reads the stream

    result = await agent.run('Hello', event_stream_handler=event_stream_handler)

    assert result.output == 'hello world this is a long answer'


async def test_run_stream_event_stream_handler_does_not_need_to_consume_stream():
    agent = Agent(TestModel(custom_output_text='hello world this is a long answer'))

    async def event_stream_handler(ctx: RunContext, stream: AsyncIterable[AgentStreamEvent]) -> None:
        return  # never reads the stream

    async with agent.run_stream('Hello', event_stream_handler=event_stream_handler) as result:
        output = await result.get_output()

    assert output == 'hello world this is a long answer'


async def test_run_event_stream_handler_unconsumed_still_executes_tool_calls():
    """A handler that ignores the stream must not stop the agent from acting on the model's reply.

    The reply (including tool calls) is built by iterating the stream, so a handler that returns
    without consuming it used to silently drop the tool call.
    """
    tool_calls: list[int] = []
    agent = Agent(TestModel())

    @agent.tool_plain
    def record(x: int) -> str:
        tool_calls.append(x)
        return 'ok'

    async def event_stream_handler(ctx: RunContext, stream: AsyncIterable[AgentStreamEvent]) -> None:
        return  # never reads the stream

    await agent.run('go', event_stream_handler=event_stream_handler)

    assert tool_calls == [0]


async def test_run_stream_event_stream_handler_unconsumed_still_executes_tool_calls():
    """Same as the `agent.run()` case, but for `agent.run_stream()` (exercises the `CallToolsNode` path)."""
    tool_calls: list[int] = []
    agent = Agent(TestModel())

    @agent.tool_plain
    def record(x: int) -> str:
        tool_calls.append(x)
        return 'ok'

    async def event_stream_handler(ctx: RunContext, stream: AsyncIterable[AgentStreamEvent]) -> None:
        return  # never reads the stream

    async with agent.run_stream('go', event_stream_handler=event_stream_handler) as result:
        await result.get_output()

    assert tool_calls == [0]


async def test_run_event_stream_handler_interrupted_does_not_drain():
    """A handler interrupted before returning (cancellation/`break`) must not trigger the drain.

    The drain only runs when the handler returns normally; re-running it on an interrupted handler
    would consume a stream the caller asked to stop, reintroducing the cancellation hang from #5313.
    The stream is unbounded, so a drain that ran after the interrupt would never terminate.
    """
    pulled = 0

    async def counting_stream(_messages: list[ModelMessage], _: AgentInfo) -> AsyncIterator[str]:
        nonlocal pulled
        while True:  # pragma: no cover - the test asserts this unbounded stream is never pulled
            pulled += 1
            yield 'hello'

    agent = Agent(FunctionModel(stream_function=counting_stream))

    async def event_stream_handler(ctx: RunContext, stream: AsyncIterable[AgentStreamEvent]) -> None:
        raise asyncio.CancelledError  # interrupted before consuming the stream

    with pytest.raises(asyncio.CancelledError):
        await agent.run('Hello', event_stream_handler=event_stream_handler)

    # The continuation composite opens each segment's `request_stream` lazily, only once the
    # consumer starts iterating, so a handler that raises before consuming never pulls the model
    # stream at all. The key guarantee is that the post-handler drain was skipped (otherwise this
    # unbounded stream would have been pulled forever).
    assert pulled == 0


async def test_stream_tool_returning_user_content():
    m = TestModel()

    agent = Agent(m)
    assert agent.name is None

    @agent.tool_plain
    async def get_image() -> ImageUrl:
        return ImageUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg')

    events: list[AgentStreamEvent] = []

    async def event_stream_handler(ctx: RunContext, stream: AsyncIterable[AgentStreamEvent]):
        async for event in stream:
            events.append(event)

    await agent.run('Hello', event_stream_handler=event_stream_handler)

    assert events == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='get_image', args={}, tool_call_id=IsStr()),
            ),
            PartEndEvent(
                index=0,
                part=ToolCallPart(tool_name='get_image', args={}, tool_call_id='pyd_ai_tool_call_id__get_image'),
            ),
            FunctionToolCallEvent(
                part=ToolCallPart(tool_name='get_image', args={}, tool_call_id=IsStr()), args_valid=True
            ),
            FunctionToolResultEvent(
                part=ToolReturnPart(
                    tool_name='get_image',
                    content=ImageUrl(
                        url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'
                    ),
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                )
            ),
            PartStartEvent(index=0, part=TextPart(content='')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(
                index=0,
                delta=TextPartDelta(
                    content_delta='{"get_image":{"url":"https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg","'
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=TextPartDelta(
                    content_delta='force_download":false,"vendor_metadata":null,"kind":"image-url","media_type":"image/jpeg","identifier":"bd38f5"}}'
                ),
            ),
            PartEndEvent(
                index=0,
                part=TextPart(
                    content='{"get_image":{"url":"https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg","force_download":false,"vendor_metadata":null,"kind":"image-url","media_type":"image/jpeg","identifier":"bd38f5"}}'
                ),
            ),
        ]
    )


async def test_run_stream_events():
    m = TestModel()

    test_agent = Agent(m)
    assert test_agent.name is None

    @test_agent.tool_plain
    async def ret_a(x: str) -> str:
        return f'{x}-apple'

    async with test_agent.run_stream_events('Hello') as event_stream:
        events = [event async for event in event_stream]
    assert test_agent.name == 'test_agent'

    assert events == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr()),
            ),
            PartEndEvent(
                index=0,
                part=ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id='pyd_ai_tool_call_id__ret_a'),
            ),
            FunctionToolCallEvent(
                part=ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr()), args_valid=True
            ),
            FunctionToolResultEvent(
                part=ToolReturnPart(
                    tool_name='ret_a',
                    content='a-apple',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                )
            ),
            PartStartEvent(index=0, part=TextPart(content='')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='{"ret_a":')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='"a-apple"}')),
            PartEndEvent(index=0, part=TextPart(content='{"ret_a":"a-apple"}')),
            AgentRunResultEvent(result=AgentRunResult(output='{"ret_a":"a-apple"}')),
        ]
    )


def test_structured_response_sync_validation():
    async def text_stream(_messages: list[ModelMessage], agent_info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        assert agent_info.output_tools is not None
        assert len(agent_info.output_tools) == 1
        name = agent_info.output_tools[0].name
        json_data = json.dumps({'response': [1, 2, 3, 4]})
        yield {0: DeltaToolCall(name=name)}
        yield {0: DeltaToolCall(json_args=json_data[:15])}
        yield {0: DeltaToolCall(json_args=json_data[15:])}

    agent = Agent(FunctionModel(stream_function=text_stream), output_type=list[int])

    chunks: list[list[int]] = []
    result = agent.run_stream_sync('')
    for structured_response in result.stream_response(debounce_by=None):
        response_data = result.validate_response_output(
            structured_response, allow_partial=structured_response.state == 'incomplete'
        )
        chunks.append(response_data)

    assert chunks == snapshot([[1], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])


async def test_get_output_after_stream_output():
    """Verify that we don't get duplicate messages in history when using tool output and `get_output` is called after `stream_output`."""
    m = TestModel()

    agent = Agent(m, output_type=bool)

    async with agent.run_stream('Hello') as result:
        outputs: list[bool] = []
        async for o in result.stream_output():
            outputs.append(o)
        o = await result.get_output()
        outputs.append(o)

    assert outputs == snapshot([False, False, False])
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Hello',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'response': False},
                        tool_call_id='pyd_ai_tool_call_id__final_result',
                    )
                ],
                usage=RequestUsage(input_tokens=51),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='test',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='pyd_ai_tool_call_id__final_result',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


@pytest.mark.parametrize('delta', [True, False])
@pytest.mark.parametrize('debounce_by', [None, 0.1])
async def test_stream_text_early_break_cleanup(delta: bool, debounce_by: float | None):
    """Breaking out of `stream_text()` triggers proper async generator cleanup.

    Regression test for https://github.com/pydantic/pydantic-ai/issues/4204
    The `aclosing` wrapper in `_stream_response_text` ensures `aclose()` propagates
    through the nested generator chain so cleanup happens in the same async context,
    preventing `RuntimeError: async generator raised StopAsyncIteration`.

    Tests both `group_by_temporal` code paths:
    - `debounce_by=None`: simple pass-through iterator
    - `debounce_by=0.1`: asyncio.Task-based buffering with pending task cancellation
    """
    cleanup_called = False

    async def sf(_: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        nonlocal cleanup_called
        try:
            for chunk in ['Hello', ' ', 'world', '!', ' More', ' text']:
                yield chunk
        finally:
            # Confirms aclose() propagated synchronously, not deferred to GC.
            cleanup_called = True

    agent = Agent(FunctionModel(stream_function=sf))

    async with agent.run_stream('test') as result:
        await anext(result.stream_text(delta=delta, debounce_by=debounce_by))

    assert cleanup_called, 'stream function cleanup should have been called by aclosing propagation'


async def test_args_validator_failure_events():
    """Test that failed validation emits args_valid=False, retries with error message, then succeeds."""
    validator_calls = 0

    def my_validator(ctx: RunContext[int], x: int, y: int) -> None:
        nonlocal validator_calls
        validator_calls += 1
        if validator_calls == 1:
            raise ModelRetry('Validation failed: x must be positive')

    agent = Agent(
        TestModel(call_tools=['add_numbers']),
        deps_type=int,
    )

    @agent.tool(args_validator=my_validator, retries=2)
    def add_numbers(ctx: RunContext[int], x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    events: list[Any] = []
    async with agent.run_stream_events('call add_numbers with x=1 and y=2', deps=42) as event_stream:
        async for event in event_stream:
            events.append(event)

    assert events == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='add_numbers', args={'x': 0, 'y': 0}, tool_call_id=IsStr()),
            ),
            PartEndEvent(
                index=0,
                part=ToolCallPart(tool_name='add_numbers', args={'x': 0, 'y': 0}, tool_call_id=IsStr()),
            ),
            FunctionToolCallEvent(
                part=ToolCallPart(tool_name='add_numbers', args={'x': 0, 'y': 0}, tool_call_id=IsStr()),
                args_valid=False,
            ),
            FunctionToolResultEvent(
                part=RetryPromptPart(
                    content='Validation failed: x must be positive',
                    tool_name='add_numbers',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ),
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='add_numbers', args={'x': 0, 'y': 0}, tool_call_id=IsStr()),
            ),
            PartEndEvent(
                index=0,
                part=ToolCallPart(tool_name='add_numbers', args={'x': 0, 'y': 0}, tool_call_id=IsStr()),
            ),
            FunctionToolCallEvent(
                part=ToolCallPart(tool_name='add_numbers', args={'x': 0, 'y': 0}, tool_call_id=IsStr()),
                args_valid=True,
            ),
            FunctionToolResultEvent(
                part=ToolReturnPart(
                    tool_name='add_numbers',
                    content=0,
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ),
            PartStartEvent(index=0, part=TextPart(content='')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='{"add_nu')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='mbers":0}')),
            PartEndEvent(index=0, part=TextPart(content='{"add_numbers":0}')),
            AgentRunResultEvent(result=AgentRunResult(output='{"add_numbers":0}')),
        ]
    )


async def test_args_validator_event_args_valid_field():
    """Test that FunctionToolCallEvent has args_valid field set correctly."""

    def my_validator(ctx: RunContext[int], x: int, y: int) -> None:
        pass  # Always succeeds

    agent = Agent(
        TestModel(call_tools=['add_numbers']),
        deps_type=int,
    )

    @agent.tool(args_validator=my_validator)
    def add_numbers(ctx: RunContext[int], x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    events: list[Any] = []
    async with agent.run_stream_events('call add_numbers with x=1 and y=2', deps=42) as event_stream:
        async for event in event_stream:
            events.append(event)

    assert events == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ToolCallPart(
                    tool_name='add_numbers', args={'x': 0, 'y': 0}, tool_call_id='pyd_ai_tool_call_id__add_numbers'
                ),
            ),
            PartEndEvent(
                index=0,
                part=ToolCallPart(
                    tool_name='add_numbers', args={'x': 0, 'y': 0}, tool_call_id='pyd_ai_tool_call_id__add_numbers'
                ),
            ),
            FunctionToolCallEvent(
                part=ToolCallPart(
                    tool_name='add_numbers', args={'x': 0, 'y': 0}, tool_call_id='pyd_ai_tool_call_id__add_numbers'
                ),
                args_valid=True,
            ),
            FunctionToolResultEvent(
                part=ToolReturnPart(
                    tool_name='add_numbers',
                    content=0,
                    tool_call_id='pyd_ai_tool_call_id__add_numbers',
                    timestamp=IsDatetime(),
                )
            ),
            PartStartEvent(index=0, part=TextPart(content='')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='{"add_nu')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='mbers":0}')),
            PartEndEvent(index=0, part=TextPart(content='{"add_numbers":0}')),
            AgentRunResultEvent(result=AgentRunResult(output='{"add_numbers":0}')),
        ]
    )


async def test_args_validator_event_args_valid_no_custom_validator():
    """Test that args_valid=True when no custom validator but schema validation passes."""
    agent = Agent(
        TestModel(call_tools=['add_numbers']),
        deps_type=int,
    )

    @agent.tool
    def add_numbers(ctx: RunContext[int], x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    events: list[Any] = []
    async with agent.run_stream_events('call add_numbers with x=1 and y=2', deps=42) as event_stream:
        async for event in event_stream:
            events.append(event)

    tool_call_events: list[FunctionToolCallEvent] = [e for e in events if isinstance(e, FunctionToolCallEvent)]
    assert len(tool_call_events) >= 1

    add_number_events = [e for e in tool_call_events if e.part.tool_name == 'add_numbers']
    assert add_number_events, 'Should have events for add_numbers'
    for event in add_number_events:
        assert event.args_valid is True


async def test_schema_validation_failure_args_valid_false():
    """Test that args_valid=False when Pydantic schema validation fails (no custom validator)."""

    def return_invalid_args(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # pragma: no cover
        """Return a tool call with invalid arguments (wrong type)."""
        return ModelResponse(parts=[ToolCallPart(tool_name='add_numbers', args={'x': 'not_an_int', 'y': 2})])

    async def stream_invalid_args(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        """Stream a tool call with invalid arguments."""
        yield {0: DeltaToolCall(name='add_numbers')}
        yield {0: DeltaToolCall(json_args='{"x": "not_an_int", "y": 2}')}

    agent = Agent(FunctionModel(return_invalid_args, stream_function=stream_invalid_args), deps_type=int)

    @agent.tool
    def add_numbers(ctx: RunContext[int], x: int, y: int) -> int:  # pragma: no cover
        """Add two numbers."""
        return x + y

    events: list[Any] = []
    try:
        async with agent.run_stream_events('call add_numbers', deps=42) as event_stream:
            async for event in event_stream:  # pragma: no branch
                events.append(event)
    except UnexpectedModelBehavior:
        pass  # Expected when max retries exceeded

    tool_call_events: list[FunctionToolCallEvent] = [e for e in events if isinstance(e, FunctionToolCallEvent)]
    assert len(tool_call_events) >= 1

    first_event = tool_call_events[0]
    assert first_event.part.tool_name == 'add_numbers'
    assert first_event.args_valid is False


async def test_args_validator_run_stream_event_handler():
    """Test that args_valid is correctly set on FunctionToolCallEvent when using run_stream()."""

    def my_validator(ctx: RunContext[int], x: int, y: int) -> None:
        pass  # Always succeeds

    agent = Agent(
        TestModel(call_tools=['add_numbers']),
        deps_type=int,
    )

    @agent.tool(args_validator=my_validator)
    def add_numbers(ctx: RunContext[int], x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    events: list[AgentStreamEvent] = []

    async def handler(ctx: RunContext[int], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    async with agent.run_stream('call add_numbers', deps=42, event_stream_handler=handler) as result:
        await result.get_output()

    tool_call_events = [e for e in events if isinstance(e, FunctionToolCallEvent)]
    assert tool_call_events
    for event in tool_call_events:
        assert event.args_valid is True


async def test_event_ordering_call_before_result():
    """Test that FunctionToolCallEvent is emitted before FunctionToolResultEvent for each tool call."""

    def my_validator(ctx: RunContext, x: int) -> None:
        pass

    agent = Agent(TestModel(call_tools=['my_tool']))

    @agent.tool(args_validator=my_validator)
    def my_tool(ctx: RunContext, x: int) -> int:
        """A tool."""
        return x * 2

    events: list[Any] = []
    async with agent.run_stream_events('test') as event_stream:
        async for event in event_stream:
            events.append(event)

    call_ids_seen: set[str] = set()
    result_ids_seen: set[str] = set()
    for event in events:
        if isinstance(event, FunctionToolCallEvent):
            call_ids_seen.add(event.tool_call_id)
            assert event.tool_call_id not in result_ids_seen, (
                f'FunctionToolResultEvent for {event.tool_call_id} appeared before FunctionToolCallEvent'
            )
        elif isinstance(event, FunctionToolResultEvent):
            result_id = event.part.tool_call_id
            result_ids_seen.add(result_id)
            assert result_id in call_ids_seen, (
                f'FunctionToolResultEvent for {result_id} appeared without prior FunctionToolCallEvent'
            )

    assert call_ids_seen
    assert result_ids_seen


async def test_args_valid_true_for_presupplied_tool_approved():
    """Test that args_valid=True when re-running with ToolApproved (validation runs upfront with approval context)."""

    def my_validator(ctx: RunContext[int], x: int) -> None:
        pass

    agent = Agent(
        TestModel(),
        deps_type=int,
        output_type=[str, DeferredToolRequests],
    )

    @agent.tool(args_validator=my_validator)
    def my_tool(ctx: RunContext[int], x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired()
        return x * 42

    # First run: tool requires approval
    result = await agent.run('Hello', deps=42)
    assert isinstance(result.output, DeferredToolRequests)
    tool_call_id = result.output.approvals[0].tool_call_id

    # Second run with ToolApproved: collect events
    messages = result.all_messages()
    events: list[Any] = []
    async with agent.run_stream_events(
        message_history=messages,
        deferred_tool_results=DeferredToolResults(approvals={tool_call_id: ToolApproved()}),
        deps=42,
    ) as event_stream:
        async for event in event_stream:
            events.append(event)

    # The FunctionToolCallEvent for the pre-supplied result should have args_valid=True
    tool_call_events = [e for e in events if isinstance(e, FunctionToolCallEvent) and e.part.tool_name == 'my_tool']
    assert tool_call_events
    assert tool_call_events[0].args_valid is True


async def test_args_valid_none_for_tool_denied():
    """Test that args_valid=None for ToolDenied and the denial message appears in the result event."""

    def my_validator(ctx: RunContext[int], x: int) -> None:
        pass

    agent = Agent(
        TestModel(),
        deps_type=int,
        output_type=[str, DeferredToolRequests],
    )

    @agent.tool(args_validator=my_validator)
    def my_tool(ctx: RunContext[int], x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired()
        return x  # pragma: no cover

    # First run: tool requires approval
    result = await agent.run('Hello', deps=42)
    assert isinstance(result.output, DeferredToolRequests)
    tool_call_id = result.output.approvals[0].tool_call_id

    # Second run with ToolDenied
    messages = result.all_messages()
    events: list[Any] = []
    async with agent.run_stream_events(
        message_history=messages,
        deferred_tool_results=DeferredToolResults(approvals={tool_call_id: ToolDenied('User denied this tool call')}),
        deps=42,
    ) as event_stream:
        async for event in event_stream:
            events.append(event)

    # FunctionToolCallEvent should have args_valid=None (pre-supplied result, no upfront validation)
    tool_call_events = [e for e in events if isinstance(e, FunctionToolCallEvent) and e.part.tool_name == 'my_tool']
    assert tool_call_events
    assert tool_call_events[0].args_valid is None

    # FunctionToolResultEvent should contain the denial message
    result_events = [e for e in events if isinstance(e, FunctionToolResultEvent) and e.part.tool_name == 'my_tool']
    assert result_events
    assert result_events[0].part.content == 'User denied this tool call'


async def test_deferred_tool_validation_event_in_stream():
    """Test that deferred (requires_approval) tools emit FunctionToolCallEvent with correct args_valid."""

    def my_validator(ctx: RunContext, x: int) -> None:
        pass

    agent = Agent(
        TestModel(),
        output_type=[str, DeferredToolRequests],
    )

    @agent.tool(args_validator=my_validator)
    def my_tool(ctx: RunContext, x: int) -> int:
        raise ApprovalRequired()

    events: list[Any] = []
    async with agent.run_stream_events('test') as event_stream:
        async for event in event_stream:
            events.append(event)

    tool_call_events = [e for e in events if isinstance(e, FunctionToolCallEvent) and e.part.tool_name == 'my_tool']
    assert tool_call_events
    # TestModel generates valid args (x=0 by default), so validation passes
    assert tool_call_events[0].args_valid is True


# region: Stream cancellation tests


async def test_run_stream_cancel():
    agent = Agent(TestModel())

    async with agent.run_stream('Hello') as result:
        assert not result.cancelled
        # Consume one chunk to start the stream
        async for _ in result.stream_text(delta=True, debounce_by=None):  # pragma: no branch
            break
        await result.cancel()
        assert result.cancelled

    # StreamedResponse.get() sets state='interrupted' when _cancelled is True
    assert result.response.state == 'interrupted'


async def test_run_stream_cancel_all_messages_includes_interrupted_response():
    """After cancelling a stream, all_messages() should include the interrupted ModelResponse."""
    agent = Agent(TestModel())

    async with agent.run_stream('Hello') as result:
        # Consume one chunk to start the stream
        async for _ in result.stream_text(delta=True, debounce_by=None):  # pragma: no branch
            break
        await result.cancel()

    assert result.cancelled
    assert result.response.state == 'interrupted'
    # The interrupted ModelResponse must appear in all_messages()
    msgs = result.all_messages()
    assert msgs == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success ')],
                usage=RequestUsage(input_tokens=51, output_tokens=1),
                model_name='test',
                timestamp=IsDatetime(),
                provider_name='test',
                run_id=IsStr(),
                conversation_id=IsStr(),
                state='interrupted',
            ),
        ]
    )


async def test_run_stream_cancel_guard_suppresses_transport_error():
    """When cancel() is called mid-stream and iteration continues, _stream_cancel_guard
    suppresses the simulated transport error and the stream ends gracefully."""
    agent = Agent(TestModel())

    async with agent.run_stream('Hello') as result:
        chunks: list[str] = []
        async for text in result.stream_text(delta=True, debounce_by=None):
            chunks.append(text)
            if not result.cancelled:  # pragma: no branch
                await result.cancel()
                # Don't break: let the loop call anext() again, which resumes
                # the generator into the _cancelled check and exercises the
                # _stream_cancel_guard suppression branch.

    assert result.cancelled
    assert result.response.state == 'interrupted'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success ')],
                usage=RequestUsage(input_tokens=51, output_tokens=1),
                model_name='test',
                timestamp=IsDatetime(),
                provider_name='test',
                run_id=IsStr(),
                conversation_id=IsStr(),
                state='interrupted',
            ),
        ]
    )


async def test_run_stream_cancel_after_complete():
    agent = Agent(TestModel())

    async with agent.run_stream('Hello') as result:
        assert not result.is_complete
        await result.get_output()
        assert result.is_complete
        assert result.response.state == 'complete'
        # A defensive cancel() after the stream is fully consumed records the
        # flag but must not downgrade response.state to 'interrupted'.
        await result.cancel()
        assert result.cancelled
        assert result.response.state == 'complete'


async def test_testmodel_stream_cancel_reports_interrupted():
    """Cancelling a `TestModel` sub-stream mid-iteration simulates the transport tear-down and reports interrupted.

    Driven directly against `model.request_stream` (not the continuation composite, which tears segments
    down via `close_stream` rather than `cancel`) so the stream's own `cancel()` fires the simulated
    `httpx.StreamClosed`, which the cancel-guard suppresses, leaving `get()` reporting `'interrupted'`.
    """
    model = TestModel(custom_output_text='hello world')
    params = models.ModelRequestParameters()

    async with model.request_stream([ModelRequest(parts=[UserPromptPart('go')])], None, params) as stream:
        iterator = stream.__aiter__()
        await iterator.__anext__()
        await stream.cancel()
        async for _ in iterator:  # the next pull raises the simulated `StreamClosed`, suppressed by the guard
            pass

    assert stream.get().state == 'interrupted'


async def test_stream_cancel_with_natural_drain_reports_interrupted():
    """A `cancel()` on a stream with no live connection still drains naturally but reports interrupted.

    Mirrors a local model whose `close_stream()` has nothing to tear down: iteration reaches a natural
    `StopAsyncIteration`, so the cancel-guard's else-branch runs but `_cancelled` keeps `_finished` unset,
    and `get()` reports `'interrupted'` rather than `'complete'`.
    """

    @dataclass
    class _NaturalDrainStream(models.StreamedResponse):
        async def _get_event_iterator(self) -> AsyncIterator[Any]:
            for _ in range(2):
                for event in self._parts_manager.handle_text_delta(vendor_part_id=0, content='x'):
                    yield event

        async def close_stream(self) -> None:
            pass  # no live connection to tear down

        @property
        def model_name(self) -> str:
            return 'drain'

        @property
        def provider_name(self) -> str | None:
            return 'drain'

        @property
        def provider_url(self) -> str | None:
            return None

        @property
        def timestamp(self) -> _datetime:
            return _datetime.now(tz=timezone.utc)

    stream = _NaturalDrainStream(models.ModelRequestParameters())
    iterator = stream.__aiter__()
    await iterator.__anext__()
    await stream.cancel()
    async for _ in iterator:  # drains to a natural completion while cancelled
        pass

    assert stream.get().state == 'interrupted'


async def test_stream_cancel_outranks_incomplete_state_hint():
    """A cancelled stream reports `'interrupted'` even when it stamped `state='incomplete'` mid-iteration.

    OpenAI Responses stamps `state='incomplete'` on every `in_progress` event. That in-flight hint must
    not outrank an explicit `cancel()` in `get()`, or a cancelled foreground stream would report
    `'incomplete'` instead of `'interrupted'` — a regression of the cancellation-state feature that a VCR
    test can't catch, since it hinges on `get()`'s internal state precedence rather than the request body.
    """

    @dataclass
    class _InProgressStream(models.StreamedResponse):
        async def _get_event_iterator(self) -> AsyncIterator[Any]:
            for _ in range(2):
                self.state = 'incomplete'  # mirror OpenAI Responses stamping on each `in_progress` event
                for event in self._parts_manager.handle_text_delta(vendor_part_id=0, content='x'):
                    yield event

        async def close_stream(self) -> None:
            pass

        @property
        def model_name(self) -> str:
            return 'in-progress'

        @property
        def provider_name(self) -> str | None:
            return 'in-progress'

        @property
        def provider_url(self) -> str | None:
            return None

        @property
        def timestamp(self) -> _datetime:
            return _datetime.now(tz=timezone.utc)

    stream = _InProgressStream(models.ModelRequestParameters())
    iterator = stream.__aiter__()
    await iterator.__anext__()
    await stream.cancel()
    async for _ in iterator:
        pass

    assert stream.get().state == 'interrupted'


async def test_completed_streamed_response_cancel_noop():
    response = ModelResponse(parts=[TextPart(content='done')], model_name='test')
    streamed_response = CompletedStreamedResponse(models.ModelRequestParameters(), response)

    await streamed_response.cancel()
    await streamed_response.cancel()

    assert streamed_response.cancelled
    assert streamed_response.response is response
    assert response.state == 'complete'


async def test_stream_response_state_incomplete_until_finished():
    """`response.state` reads `'incomplete'` mid-stream and flips to `'complete'` once iteration ends."""
    agent = Agent(TestModel(custom_output_text='hello world'))

    async with agent.run_stream('Hello') as result:
        async for _ in result.stream_text(delta=True, debounce_by=None):
            assert result.response.state == 'incomplete'
        await result.get_output()

    assert result.response.state == 'complete'


async def test_stream_response_yields_incomplete_then_complete():
    """`stream_response` yields `state='incomplete'` mid-stream; the trailing yield is `'complete'`."""
    agent = Agent(TestModel(custom_output_text='hello world'))

    async with agent.run_stream('Hello') as result:
        states = [msg.state async for msg in result.stream_response(debounce_by=None)]

    assert states[-1] == 'complete'
    assert all(state == 'incomplete' for state in states[:-1])


async def test_stream_response_state_incomplete_after_early_break():
    """Breaking out of the stream early must not flip `state` to `'complete'`.

    `aclose()` on the underlying async generator raises `GeneratorExit` at the
    suspended `yield`, so `_finished` must stay `False` and the truncated
    response must keep reporting `'incomplete'`.
    """
    agent = Agent(TestModel(custom_output_text='hello world'))

    async with agent.iter('Hello') as run:
        async for node in run:  # pragma: no branch
            if agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for _ in stream:  # pragma: no branch
                        break
                    assert stream.response.state == 'incomplete'
                return


async def test_run_stream_events_break_cleanup():
    agent = Agent(TestModel())

    async with agent.run_stream_events('Hello') as events:
        await anext(events)

    # __aexit__ closes the iterator and drains the background task; no task leak, no error.


def make_cleanup_signal_test_model(producer_started: asyncio.Event) -> type[TestModel]:
    class CleanupSignalTestModel(TestModel):
        @asynccontextmanager
        async def request_stream(
            self,
            messages: list[ModelMessage],
            model_settings: models.ModelSettings | None,
            model_request_parameters: models.ModelRequestParameters,
            run_context: RunContext | None = None,
        ) -> AsyncGenerator[models.StreamedResponse]:
            async with super().request_stream(
                messages,
                model_settings,
                model_request_parameters,
                run_context,
            ) as stream:
                producer_started.set()
                yield stream

    return CleanupSignalTestModel


async def test_run_stream_events_unstarted_iterator_cleanup():
    """Entering and exiting the CM without advancing the iterator must not start the background task."""
    producer_started = asyncio.Event()
    cleanup_signal_test_model = make_cleanup_signal_test_model(producer_started)

    agent = Agent(cleanup_signal_test_model(custom_output_text='hello'))

    # `sleep(0)` yields to the event loop while each context is open, so an eager-start regression would
    # get a chance to schedule its background task and set `producer_started` before we assert it didn't.
    async with agent.run_stream_events(''):
        await asyncio.sleep(0)

    empty_context = agent.run_stream_events('')
    await empty_context.__aexit__(None, None, None)

    context = agent.run_stream_events('')
    await context.__aenter__()
    await asyncio.sleep(0)
    await context.__aexit__(None, None, None)
    await context.__aexit__(None, None, None)

    reentered_context = agent.run_stream_events('')
    await reentered_context.__aenter__()
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match='cannot be entered more than once'):
        await reentered_context.__aenter__()
    await reentered_context.__aexit__(None, None, None)

    assert not producer_started.is_set()


async def test_run_stream_events_first_iteration_starts_background_task():
    producer_started = asyncio.Event()
    cleanup_signal_test_model = make_cleanup_signal_test_model(producer_started)

    agent = Agent(cleanup_signal_test_model(custom_output_text='hello'))

    async with agent.run_stream_events('') as events:
        # Time out the first iteration itself so a lazy-start regression fails fast instead of hanging here.
        await asyncio.wait_for(anext(events), timeout=1.0)
        assert producer_started.is_set()


async def test_run_stream_events_break_on_final_result_retrieves_late_producer_error():
    """Breaking on the documented final-result event must still retrieve background task errors."""
    producer_finished = asyncio.Event()

    async def stream_that_fails_after_final_result(
        _messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[str]:
        yield 'hello'
        try:
            raise RuntimeError('producer boom')
        finally:
            producer_finished.set()

    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    handle_exception = MagicMock()

    loop.set_exception_handler(handle_exception)
    try:
        agent = Agent(FunctionModel(stream_function=stream_that_fails_after_final_result))

        async with agent.run_stream_events('') as events:
            async for event in events:  # pragma: no branch
                if isinstance(event, FinalResultEvent):
                    # This mirrors the documented "stop once final result is known" pattern.
                    # The producer task can still finish with an exception before the CM exits.
                    await asyncio.wait_for(producer_finished.wait(), timeout=1.0)
                    await asyncio.sleep(0)
                    break

        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    handle_exception.assert_not_called()


async def test_run_stream_events_external_task_cancellation():
    """When the outer task is cancelled, the CancelledError handler forwards cancellation to the producer."""
    never = asyncio.Event()

    async def blocking_stream(_messages: list[ModelMessage], agent_info: AgentInfo) -> AsyncIterator[str]:
        yield 'hello'
        await never.wait()  # block forever so the consumer is still awaiting when we cancel

    agent = Agent(FunctionModel(stream_function=blocking_stream))

    async def consume() -> None:
        async with agent.run_stream_events('') as stream:
            async for _ in stream:
                pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)  # let the task start and block on the stream
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_run_stream_events_managed_cancellation_waits_for_cleanup():
    # Test for https://github.com/pydantic/pydantic-ai/issues/5132.
    cleanup_finished = asyncio.Event()
    first_event_seen = asyncio.Event()

    class SlowCleanupTestModel(TestModel):
        @asynccontextmanager
        async def request_stream(
            self,
            messages: list[ModelMessage],
            model_settings: models.ModelSettings | None,
            model_request_parameters: models.ModelRequestParameters,
            run_context: RunContext | None = None,
        ) -> AsyncGenerator[models.StreamedResponse]:
            async with super().request_stream(
                messages,
                model_settings,
                model_request_parameters,
                run_context,
            ) as stream:
                try:
                    yield stream
                finally:
                    await asyncio.sleep(0.2)
                    cleanup_finished.set()

    agent = Agent(SlowCleanupTestModel(custom_output_text='hello'))

    async def consume() -> None:
        async with agent.run_stream_events('Hello') as stream:
            await anext(stream)
            first_event_seen.set()
            await asyncio.sleep(10)

    task = asyncio.create_task(consume())
    await first_event_seen.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert cleanup_finished.is_set()


async def test_stream_wrap_model_request_readiness_wait_cancels_wrapper_task_on_outer_cancellation():
    """Outer cancellation while waiting for streaming model wrapper readiness should clean up the wrapper task.

    Target boundary: `ModelRequestNode.stream()` creates `wrap_task` and `ready_waiter`, then waits for
    `asyncio.wait({ready_waiter, wrap_task}, return_when=asyncio.FIRST_COMPLETED)`. If the outer task is
    cancelled while parked on that wait, the wrapper task must be drained; otherwise the user's
    `wrap_model_request` cleanup never runs.
    """
    cleanup_finished = asyncio.Event()
    started = asyncio.Event()
    never_finishes = asyncio.Future[ModelResponse]()

    class WrapModelRequestCapability(AbstractCapability):
        async def wrap_model_request(
            self,
            ctx: RunContext,
            *,
            request_context: ModelRequestContext,
            handler: WrapModelRequestHandler,
        ) -> ModelResponse:
            try:
                started.set()
                # Suspend before calling handler() so we sit inside the readiness wait at
                # `_agent_graph.py:asyncio.wait({ready_waiter, wrap_task}, ...)`.
                return await never_finishes
            finally:
                # Without the drain on the readiness wait, this finally never runs.
                cleanup_finished.set()

    agent = Agent(TestModel(), capabilities=[WrapModelRequestCapability()])

    async def consume() -> None:
        async with agent.run_stream_events('Hello') as stream:
            async for _ in stream:
                pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cleanup_finished.is_set()


# endregion
