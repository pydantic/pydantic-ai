"""Graph iteration must release run references and temporary execution state.

In-process graph steps expose reference lifetimes without a model provider or VCR recording.
"""

from __future__ import annotations

import gc
import weakref
from dataclasses import dataclass
from typing import Literal

import anyio
import pytest

from pydantic_graph import GraphBuilder, StepContext
from pydantic_graph.graph_builder import ErrorMarker, GraphRun


@dataclass
class Deps:
    pass


@pytest.mark.parametrize('complete', [False, True])
async def test_graph_releases_dependencies(complete: bool) -> None:
    builder = GraphBuilder(deps_type=Deps, output_type=int)

    @builder.step
    async def answer(ctx: StepContext[None, Deps, None]) -> int:
        return 42

    builder.add(builder.edge_from(builder.start_node).to(answer), builder.edge_from(answer).to(builder.end_node))
    graph = builder.build()
    gc.disable()
    try:
        deps = Deps()
        ref = weakref.ref(deps)
        # Before Python 3.13, name inference snapshots the caller's locals and retains `deps` there.
        async with graph.iter(deps=deps, infer_name=False) as run:
            if complete:
                async for _ in run:
                    pass
        assert run.output == (42 if complete else None)
        assert run.deps is deps
        with pytest.raises(StopAsyncIteration):
            await anext(run)
        del deps, run
        assert ref() is None
    finally:
        gc.enable()


@pytest.mark.filterwarnings('ignore:Unclosed.*:ResourceWarning')
async def test_unentered_graph_run_releases_dependencies() -> None:
    builder = GraphBuilder(deps_type=Deps, output_type=int)
    builder.add(builder.edge_from(builder.start_node).transform(lambda ctx: 1).to(builder.end_node))
    graph = builder.build()
    gc.disable()
    try:
        deps = Deps()
        ref = weakref.ref(deps)
        run = GraphRun(graph, state=None, deps=deps, inputs=None, traceparent=None)
        # No task group was entered; the unentered in-memory streams' warnings are unrelated to ownership.
        del deps, run
        assert ref() is None
    finally:
        gc.enable()


@pytest.mark.parametrize('failure', ['step', 'consumer'])
async def test_failed_graph_keeps_inspection_without_retaining_dropped_dependencies(
    failure: Literal['step', 'consumer'],
) -> None:
    builder = GraphBuilder(deps_type=Deps, output_type=int)

    @builder.step
    async def fail_step(ctx: StepContext[None, Deps, None]) -> int:
        raise RuntimeError('step failed')

    builder.add(builder.edge_from(builder.start_node).to(fail_step), builder.edge_from(fail_step).to(builder.end_node))
    graph = builder.build()
    errors: list[RuntimeError] = []
    runs: list[GraphRun[None, Deps, int]] = []

    async def execute() -> weakref.ReferenceType[Deps]:
        deps = Deps()
        ref = weakref.ref(deps)
        run = None
        try:
            async with graph.iter(deps=deps, infer_name=False) as run:
                runs.append(run)
                if failure == 'consumer':
                    raise RuntimeError('consumer failed')
                async for _ in run:
                    pass
        except RuntimeError as exc:
            errors.append(exc)
        del deps, run
        return ref

    gc.disable()
    try:
        ref = await execute()
        assert ref() is not None
        assert len(errors) == 1
        assert runs[0].deps is ref()
        assert runs[0].output is None
        if failure == 'step':
            marker = runs[0].next_task
            assert isinstance(marker, ErrorMarker)
            assert marker.error is errors[0]
            del marker
        errors.clear()
        assert ref() is not None
        runs.clear()
        assert ref() is None
    finally:
        gc.enable()


async def test_completed_run_releases_cancelled_task_inputs() -> None:
    """Pin `__aexit__` detachment with the completed run still alive.

    GC clears unreachable cycles; inputs reachable through a retained iterator would survive it.
    """

    @dataclass
    class Payload:
        block: bool

    builder = GraphBuilder(output_type=int)
    refs: list[weakref.ReferenceType[Payload]] = []
    started = anyio.Event()
    cancelled = anyio.Event()

    @builder.step
    async def generate(ctx: StepContext[None, None, None]) -> list[Payload]:
        payloads: list[Payload] = [Payload(block=False), Payload(block=True)]
        refs.extend(weakref.ref(payload) for payload in payloads)
        return payloads

    @builder.step
    async def process(ctx: StepContext[None, None, Payload]) -> int:
        if ctx.inputs.block:
            started.set()
            try:
                await anyio.sleep_forever()
            finally:
                cancelled.set()
        with anyio.fail_after(30):
            await started.wait()
        return 1

    builder.add(
        builder.edge_from(builder.start_node).to(generate),
        builder.edge_from(generate).map().to(process),
        builder.edge_from(process).to(builder.end_node),
    )
    async with builder.build().iter(infer_name=False) as run:
        async for _ in run:
            pass

    assert cancelled.is_set()
    gc.collect()
    assert len(refs) == 2
    assert all(ref() is None for ref in refs)
    assert run.output == 1
    with pytest.raises(StopAsyncIteration):
        await anext(run)
