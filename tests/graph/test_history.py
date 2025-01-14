from __future__ import annotations as _annotations

from dataclasses import dataclass
from datetime import timezone

import pytest
from inline_snapshot import snapshot

from pydantic_graph import BaseNode, End, EndStep, Graph, GraphContext, NodeStep

from ..conftest import IsFloat, IsNow

pytestmark = pytest.mark.anyio


@dataclass
class MyState:
    x: int
    y: str


@dataclass
class Foo(BaseNode[MyState]):
    async def run(self, ctx: GraphContext[MyState]) -> Bar:
        ctx.state.x += 1
        return Bar()


@dataclass
class Bar(BaseNode[MyState, str]):
    async def run(self, ctx: GraphContext[MyState]) -> End[str]:
        ctx.state.y += 'y'
        return End(f'x={ctx.state.x} y={ctx.state.y}')


graph = Graph(nodes=(Foo, Bar), state_type=MyState)


async def test_dump_history():
    result, history = await graph.run(MyState(1, ''), Foo())
    assert result == snapshot('x=2 y=y')
    assert history == snapshot(
        [
            NodeStep(state=MyState(x=1, y=''), node=Foo(), start_ts=IsNow(tz=timezone.utc), duration=IsFloat()),
            NodeStep(state=MyState(x=2, y=''), node=Bar(), start_ts=IsNow(tz=timezone.utc), duration=IsFloat()),
            EndStep(state=MyState(x=2, y='y'), result=End(data='x=2 y=y'), ts=IsNow(tz=timezone.utc)),
        ]
    )
    history_json = graph.dump_history(history)
    assert history_json.startswith(b'[{"state":')
    history_loaded = graph.load_history(history_json)
    assert history == history_loaded
