"""Graph iteration must not retain run state through its iterator callbacks."""

from __future__ import annotations

import gc
import weakref
from dataclasses import dataclass

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
    enabled = gc.isenabled()
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
        if enabled:
            gc.enable()
