"""Tests for `DurableRunCancellation`, the engine-agnostic seam for externally cancelling a
durable agent run.

The capability captures the run's cancellation controller in `before_run` and triggers it from
`cancel()`. Each durable engine wires its own external-cancellation mechanism (a Temporal
`@workflow.signal`, a DBOS/Prefect equivalent) to that one method; the engine-agnostic binding is
exercised here without any durable runtime, since the behavior under test is pure control flow
around injected `asyncio` cancellation that no recorded provider response can trigger. The Temporal
signal wiring itself is covered end-to-end in `test_temporal.py`.
"""

from __future__ import annotations as _annotations

import asyncio

import pytest

from pydantic_ai import Agent, RunCancelled
from pydantic_ai.durable_exec import DurableRunCancellation
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

pytestmark = pytest.mark.anyio


async def test_cancel_from_sibling_task_surfaces_run_cancelled():
    """`cancel()` from another task cancels the bound run and surfaces as `RunCancelled`, standing
    in for an engine's external-cancellation handler reaching a run in flight."""
    started = asyncio.Event()

    async def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError  # pragma: no cover

    cancellation = DurableRunCancellation()
    task = asyncio.create_task(Agent(FunctionModel(model_function)).run('hello', capabilities=[cancellation]))
    await started.wait()
    cancellation.cancel()

    with pytest.raises(RunCancelled) as exc_info:
        await task
    assert [type(message).__name__ for message in exc_info.value.all_messages()] == ['ModelRequest']


async def test_cancel_before_run_binds_controller_still_cancels():
    """A `cancel()` that arrives before the run binds its controller is buffered and applied in
    `before_run`, so the run is cancelled before any model request goes out."""
    called = False

    async def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # pragma: no cover
        nonlocal called
        called = True
        raise AssertionError

    cancellation = DurableRunCancellation()
    cancellation.cancel()

    with pytest.raises(RunCancelled) as exc_info:
        await Agent(FunctionModel(model_function)).run('hello', capabilities=[cancellation])
    assert exc_info.value.all_messages() == []
    assert not called


async def test_cancel_after_run_finishes_is_a_no_op():
    """Once the run has finished, `cancel()` is a no-op and doesn't disturb the event loop."""
    cancellation = DurableRunCancellation()
    result = await Agent(TestModel()).run('hello', capabilities=[cancellation])
    assert result.output

    cancellation.cancel()
    await asyncio.sleep(0)
    assert await asyncio.sleep(0, result='unrelated') == 'unrelated'


def test_capability_is_safe_at_runtime_and_not_spec_constructible():
    """The capability introduces no durable units, so it may be added per-run inside a durable
    container (`_safe_at_runtime`), and it holds a live controller reference, so it opts out of
    spec construction."""
    assert DurableRunCancellation._safe_at_runtime is True  # pyright: ignore[reportPrivateUsage]
    assert DurableRunCancellation.get_serialization_name() is None
