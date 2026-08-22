"""Thread-safety regression tests for the pending-message queue."""

from __future__ import annotations

import threading
from typing import Any, Literal

import pytest

from pydantic_ai import Agent, _agent_graph
from pydantic_ai._enqueue import PendingMessage
from pydantic_ai.capabilities._pending_messages import PendingMessageDrainCapability
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel

pytestmark = pytest.mark.anyio


class _LockCheckingPendingMessages(list[PendingMessage]):
    """Assert that every drain slice-write happens under the run's queue lock."""

    def __init__(self, lock: threading.Lock):
        super().__init__()
        self.lock = lock
        self.slice_writes = 0

    def __setitem__(self, key: Any, value: Any, /) -> None:
        if isinstance(key, slice):
            assert self.lock.locked(), 'pending-message drain replaced the queue without holding its lock'
            self.slice_writes += 1
        super().__setitem__(key, value)  # pyright: ignore[reportUnknownArgumentType]


@pytest.mark.parametrize('enqueue_from', ['run_context', 'agent_run'])
async def test_enqueue_waits_for_pending_message_lock(
    enqueue_from: Literal['run_context', 'agent_run'],
) -> None:
    """Both public enqueue entry points use the same per-run lock as drains."""
    agent = Agent(TestModel())
    async with agent.iter('hello') as agent_run:
        run_context = _agent_graph.build_run_context(agent_run.ctx)
        lock = agent_run._graph_run.deps.pending_messages_lock  # pyright: ignore[reportPrivateUsage]
        assert run_context._pending_messages_lock is lock  # pyright: ignore[reportPrivateUsage]

        enqueue_started = threading.Event()
        enqueue_done = threading.Event()
        enqueue_errors: list[BaseException] = []

        def enqueue() -> None:
            enqueue_started.set()
            try:
                if enqueue_from == 'run_context':
                    run_context.enqueue('late')
                else:
                    agent_run.enqueue('late')
            except BaseException as exc:  # pragma: no cover - asserted below
                enqueue_errors.append(exc)
            finally:
                enqueue_done.set()

        enqueue_thread = threading.Thread(target=enqueue)
        with lock:
            enqueue_thread.start()
            assert enqueue_started.wait(timeout=5)
            completed_while_locked = enqueue_done.wait(timeout=0.1)

        enqueue_thread.join(timeout=5)
        assert not enqueue_thread.is_alive()
        assert not completed_while_locked, 'enqueue did not wait for the pending-message lock'
        assert enqueue_errors == []

        queue = agent_run._graph_run.state.pending_messages  # pyright: ignore[reportPrivateUsage]
        assert len(queue) == 1
        late = queue[0].messages[0]
        assert isinstance(late, ModelRequest)
        assert isinstance(late.parts[0], UserPromptPart)
        assert late.parts[0].content == 'late'
        queue.clear()


async def test_pending_message_drain_holds_lock_through_queue_replacement() -> None:
    """The drain holds the run lock through iteration and slice replacement."""
    agent = Agent(TestModel())
    async with agent.iter('hello') as agent_run:
        lock = agent_run._graph_run.deps.pending_messages_lock  # pyright: ignore[reportPrivateUsage]
        queue = _LockCheckingPendingMessages(lock)
        initial = PendingMessage.from_content('initial')
        assert initial is not None
        queue.append(initial)
        agent_run._graph_run.state.pending_messages = queue  # pyright: ignore[reportPrivateUsage]
        run_context = _agent_graph.build_run_context(agent_run.ctx)

        request_context = ModelRequestContext(
            model=TestModel(),
            messages=[],
            model_settings=None,
            model_request_parameters=ModelRequestParameters(),
        )
        await PendingMessageDrainCapability().before_model_request(run_context, request_context)

        assert queue.slice_writes == 1
        assert queue == []
        assert len(request_context.messages) == 1
        drained = request_context.messages[0]
        assert isinstance(drained, ModelRequest)
        assert isinstance(drained.parts[0], UserPromptPart)
        assert drained.parts[0].content == 'initial'
