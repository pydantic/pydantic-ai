# pyright: reportPrivateUsage=false
from __future__ import annotations as _annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest
from dirty_equals import IsStr
from inline_snapshot import snapshot

from pydantic_graph import BaseNode, End, EndStep, Graph, GraphRunContext, GraphSetupError, NodeStep

from ..conftest import IsFloat, IsNow

pytestmark = pytest.mark.anyio


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
class Bar(BaseNode[MyState, None, int]):
    async def run(self, ctx: GraphRunContext[MyState]) -> End[int]:
        ctx.state.y += 'y'
        return End(ctx.state.x * 2)


@pytest.mark.parametrize(
    'graph',
    [
        Graph(nodes=(Foo, Bar), state_type=MyState, run_end_type=int),
        Graph(nodes=(Foo, Bar), state_type=MyState),
        Graph(nodes=(Foo, Bar), run_end_type=int),
        Graph(nodes=(Foo, Bar)),
    ],
)
async def test_dump_load_history(graph: Graph[MyState, None, int]):
    result = await graph.run(Foo(), state=MyState(1, ''))
    assert result.output == snapshot(4)
    assert result.state == snapshot(MyState(x=2, y='y'))
    assert result.history == snapshot(
        [
            NodeStep(state=MyState(x=2, y=''), node=Foo(), start_ts=IsNow(tz=timezone.utc), duration=IsFloat()),
            NodeStep(state=MyState(x=2, y='y'), node=Bar(), start_ts=IsNow(tz=timezone.utc), duration=IsFloat()),
            EndStep(result=End(data=4), ts=IsNow(tz=timezone.utc)),
        ]
    )
    history_json = graph.dump_history(result.history)
    assert json.loads(history_json) == snapshot(
        [
            {
                'state': {'x': 2, 'y': ''},
                'node': {'node_id': 'Foo'},
                'start_ts': IsStr(regex=r'20\d\d-\d\d-\d\dT.+'),
                'duration': IsFloat(),
                'kind': 'node',
            },
            {
                'state': {'x': 2, 'y': 'y'},
                'node': {'node_id': 'Bar'},
                'start_ts': IsStr(regex=r'20\d\d-\d\d-\d\dT.+'),
                'duration': IsFloat(),
                'kind': 'node',
            },
            {'result': {'data': 4}, 'ts': IsStr(regex=r'20\d\d-\d\d-\d\dT.+'), 'kind': 'end'},
        ]
    )
    history_loaded = graph.load_history(history_json)
    assert result.history == history_loaded

    custom_history = [
        {
            'state': {'x': 2, 'y': ''},
            'node': {'node_id': 'Foo'},
            'start_ts': '2025-01-01T00:00:00Z',
            'duration': 123,
            'kind': 'node',
        },
        {'result': {'data': '42'}, 'ts': '2025-01-01T00:00:00Z', 'kind': 'end'},
    ]
    history_loaded = graph.load_history(json.dumps(custom_history))
    assert history_loaded == snapshot(
        [
            NodeStep(
                state=MyState(x=2, y=''),
                node=Foo(),
                start_ts=datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc),
                duration=123.0,
            ),
            EndStep(result=End(data=42), ts=datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)),
        ]
    )


def test_one_node():
    @dataclass
    class MyNode(BaseNode[None, None, int]):
        async def run(self, ctx: GraphRunContext) -> End[int]:
            return End(123)

    g = Graph(nodes=[MyNode])

    custom_history = [
        {'result': {'data': '123'}, 'ts': '2025-01-01T00:00:00Z', 'kind': 'end'},
    ]
    history_loaded = g.load_history(json.dumps(custom_history))
    assert history_loaded == snapshot(
        [
            EndStep(result=End(data=123), ts=datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)),
        ]
    )


def test_no_generic_arg():
    @dataclass
    class NoGenericArgsNode(BaseNode):
        async def run(self, ctx: GraphRunContext) -> NoGenericArgsNode:
            return NoGenericArgsNode()

    g = Graph(nodes=[NoGenericArgsNode])
    assert g._get_state_type() is type(None)
    with pytest.raises(GraphSetupError, match='Could not infer run end type from nodes, please set `run_end_type`.'):
        g._get_run_end_type()

    g = Graph(nodes=[NoGenericArgsNode], run_end_type=None)  # pyright: ignore[reportArgumentType]
    assert g._get_run_end_type() is None

    custom_history = [
        {'result': {'data': None}, 'ts': '2025-01-01T00:00:00Z', 'kind': 'end'},
    ]
    history_loaded = g.load_history(json.dumps(custom_history))
    assert history_loaded == snapshot(
        [
            EndStep(result=End(data=None), ts=datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)),
        ]
    )
