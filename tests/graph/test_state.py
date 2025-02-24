# pyright: reportPrivateUsage=false
from __future__ import annotations as _annotations

from dataclasses import dataclass
from datetime import timezone

import pytest
from inline_snapshot import snapshot

from pydantic_graph import BaseNode, End, EndStep, Graph, GraphRunContext, NodeStep

from ..conftest import IsFloat, IsNow

pytestmark = pytest.mark.anyio


async def test_run_graph():
    @dataclass
    class MyState:
        x: int
        y: str

    @dataclass
    class Foo(BaseNode[MyState]):
        async def run(self, ctx: GraphRunContext[MyState]) -> Bar:
            ctx.state.x += 1
            return Bar()

    @dataclass
    class Bar(BaseNode[MyState, None, str]):
        async def run(self, ctx: GraphRunContext[MyState]) -> End[str]:
            ctx.state.y += 'y'
            return End(f'x={ctx.state.x} y={ctx.state.y}')

    graph = Graph(nodes=(Foo, Bar))
    assert graph._get_state_type() is MyState
    assert graph._get_run_end_type() is str
    state = MyState(1, '')
    result = await graph.run(Foo(), state=state)
    assert result.output == snapshot('x=2 y=y')
    assert result.history == snapshot(
        [
            NodeStep(
                state=MyState(x=2, y=''),
                node=Foo(),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
            ),
            NodeStep(
                state=MyState(x=2, y='y'),
                node=Bar(),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
            ),
            EndStep(result=End(data='x=2 y=y'), ts=IsNow(tz=timezone.utc)),
        ]
    )
    assert state == MyState(x=2, y='y')
