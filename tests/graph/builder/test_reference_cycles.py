"""A finished graph run must not retain run state through its iterator.

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


@dataclass
class Payload:
    name: str


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


async def test_retained_run_releases_cancelled_task_inputs() -> None:
    refs: list[weakref.ref[Payload]] = []
    builder = GraphBuilder(output_type=int)

    @builder.step
    async def generate(ctx: StepContext[None, None, None]) -> list[Payload]:
        payloads = [Payload('fast'), Payload('slow')]
        refs.extend(weakref.ref(payload) for payload in payloads)
        return payloads

    @builder.step
    async def process(ctx: StepContext[None, None, Payload]) -> int:
        if ctx.inputs.name == 'slow':
            # Cancelled when the fast branch ends the graph.
            await anyio.sleep_forever()
        return 1

    builder.add(
        builder.edge_from(builder.start_node).to(generate),
        builder.edge_from(generate).map().to(process),
        builder.edge_from(process).to(builder.end_node),
    )
    with anyio.fail_after(5):
        async with builder.build().iter(infer_name=False) as run:
            async for _ in run:
                pass
    # `run` stays referenced, as when an app keeps it to read the output.
    assert run.output == 1
    gc.collect()
    assert [ref() for ref in refs] == [None, None]
