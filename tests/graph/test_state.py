from __future__ import annotations as _annotations

from dataclasses import dataclass
from datetime import timezone

import pytest
from inline_snapshot import snapshot

from pydantic_graph import (
    BaseNode,
    End,
    EndSnapshot,
    FullStatePersistence,
    Graph,
    GraphRunContext,
    NodeSnapshot,
)

from ..conftest import IsFloat, IsNow

pytestmark = pytest.mark.anyio


async def test_run_graph(mock_snapshot_id: object):
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
    assert graph.inferred_types == (MyState, str)
    state = MyState(1, '')
    sp = FullStatePersistence()
    result = await graph.run(Foo(), state=state, persistence=sp)
    assert result.output == snapshot('x=2 y=y')
    assert sp.history == snapshot(
        [
            NodeSnapshot(
                state=MyState(x=1, y=''),
                node=Foo(),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
                status='success',
                id='Foo:1',
            ),
            NodeSnapshot(
                state=MyState(x=2, y=''),
                node=Bar(),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
                status='success',
                id='Bar:2',
            ),
            EndSnapshot(state=MyState(x=2, y='y'), result=End(data='x=2 y=y'), ts=IsNow(tz=timezone.utc), id='end:3'),
        ]
    )
    assert state == MyState(x=2, y='y')
