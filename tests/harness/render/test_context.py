from __future__ import annotations

import inspect

import pytest
from render.workflows import TaskContext, Workflows

from pydantic_ai_harness import RenderWorkflows

from .conftest import RecordingTaskContext


async def test_public_task_context_is_app_scoped_and_nestable() -> None:
    app = Workflows()
    first = RenderWorkflows[None](app, deps_type=type(None), name='first')
    second = RenderWorkflows[None](app, deps_type=type(None), name='second')
    states: list[tuple[bool, bool]] = []

    async def inner_impl(ctx: TaskContext) -> None:
        del ctx
        states.append((first.in_durable_context, second.in_durable_context))

    inner = first.task(inner_impl)

    async def outer_impl(ctx: TaskContext) -> None:
        states.append((first.in_durable_context, second.in_durable_context))
        await ctx.run(inner)
        states.append((first.in_durable_context, second.in_durable_context))

    outer = second.task(outer_impl)
    result = outer.func(RecordingTaskContext())
    assert inspect.isawaitable(result)
    await result

    assert states == [(True, True), (True, True), (True, True)]
    assert first.in_durable_context is False
    assert second.in_durable_context is False


async def test_public_task_context_keeps_different_apps_isolated() -> None:
    first = RenderWorkflows[None](Workflows(), deps_type=type(None), name='first')
    second = RenderWorkflows[None](Workflows(), deps_type=type(None), name='second')
    states: list[tuple[bool, bool]] = []

    async def inner_impl(ctx: TaskContext) -> None:
        del ctx
        states.append((first.in_durable_context, second.in_durable_context))

    inner = first.task(inner_impl)

    async def outer_impl(ctx: TaskContext) -> None:
        states.append((first.in_durable_context, second.in_durable_context))
        await ctx.run(inner)
        states.append((first.in_durable_context, second.in_durable_context))

    outer = second.task(outer_impl)
    result = outer.func(RecordingTaskContext())
    assert inspect.isawaitable(result)
    await result

    assert states == [(False, True), (True, True), (False, True)]
    assert first.in_durable_context is False
    assert second.in_durable_context is False


async def test_public_task_context_resets_after_error() -> None:
    runtime = RenderWorkflows[None](Workflows(), deps_type=type(None))

    async def fails_impl(ctx: TaskContext) -> None:
        del ctx
        assert runtime.in_durable_context
        raise RuntimeError('boom')

    fails = runtime.task(fails_impl)
    result = fails.func(RecordingTaskContext())
    assert inspect.isawaitable(result)
    with pytest.raises(RuntimeError, match='boom'):
        await result
    assert runtime.in_durable_context is False
