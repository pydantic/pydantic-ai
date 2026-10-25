"""Graph iteration must release run references and temporary execution state.

In-process graph steps expose reference lifetimes without a model provider or VCR recording.
"""

from __future__ import annotations

import gc
import weakref
from dataclasses import dataclass

import anyio
import pytest

from pydantic_graph import GraphBuilder, StepContext


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
