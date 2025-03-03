# pyright: reportPrivateUsage=false
from __future__ import annotations as _annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest
from dirty_equals import IsStr
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
    sp = FullStatePersistence()
    result = await graph.run(Foo(), state=MyState(1, ''), persistence=sp)
    assert result.output == snapshot(4)
    assert result.state == snapshot(MyState(x=2, y='y'))
    assert result.persistence == snapshot(
        [
            NodeSnapshot(state=MyState(x=1, y=''), node=Foo(), start_ts=IsNow(tz=timezone.utc), duration=IsFloat()),
            NodeSnapshot(state=MyState(x=2, y=''), node=Bar(), start_ts=IsNow(tz=timezone.utc), duration=IsFloat()),
            EndSnapshot(state=MyState(x=2, y='y'), result=End(4), ts=IsNow(tz=timezone.utc)),
        ]
    )
    history_json = sp.dump_json()
    assert json.loads(history_json) == snapshot(
        [
            {
                'state': {'x': 1, 'y': ''},
                'node': {'node_id': 'Foo'},
                'start_ts': IsStr(regex=r'20\d\d-\d\d-\d\dT.+'),
                'duration': IsFloat(),
                'kind': 'node',
            },
            {
                'state': {'x': 2, 'y': ''},
                'node': {'node_id': 'Bar'},
                'start_ts': IsStr(regex=r'20\d\d-\d\d-\d\dT.+'),
                'duration': IsFloat(),
                'kind': 'node',
            },
            {
                'state': {'x': 2, 'y': 'y'},
                'result': {'data': 4},
                'ts': IsStr(regex=r'20\d\d-\d\d-\d\dT.+'),
                'kind': 'end',
            },
        ]
    )

    sp2 = FullStatePersistence()
    graph.set_persistence_types(sp2)

    sp2.load_json(history_json)
    assert sp.history == sp2.history

    custom_history = [
        {
            'state': {'x': 2, 'y': ''},
            'node': {'node_id': 'Foo'},
            'start_ts': '2025-01-01T00:00:00Z',
            'duration': 123,
            'kind': 'node',
        },
        {
            'state': {'x': 42, 'y': 'new'},
            'result': {'data': '42'},
            'ts': '2025-01-01T00:00:00Z',
            'kind': 'end',
        },
    ]
    sp3 = FullStatePersistence()
    graph.set_persistence_types(sp3)
    sp3.load_json(json.dumps(custom_history))
    assert sp3.history == snapshot(
        [
            NodeSnapshot(
                state=MyState(x=2, y=''),
                node=Foo(),
                start_ts=datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc),
                duration=123.0,
            ),
            EndSnapshot(
                state=MyState(x=42, y='new'), result=End(data=42), ts=datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
            ),
        ]
    )


def test_one_node():
    @dataclass
    class MyNode(BaseNode[None, None, int]):
        node_field: int

        async def run(self, ctx: GraphRunContext) -> End[int]:
            return End(123)

    g = Graph(nodes=[MyNode])

    custom_history = [
        {
            'state': None,
            'node': {'node_id': 'MyNode', 'node_field': 42},
            'start_ts': '2025-01-01T00:00:00Z',
            'duration': 123,
            'kind': 'node',
        },
    ]
    sp = FullStatePersistence()
    g.set_persistence_types(sp)
    sp.load_json(json.dumps(custom_history))
    assert sp.history == snapshot(
        [
            NodeSnapshot(
                state=None,
                node=MyNode(node_field=42),
                start_ts=datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc),
                duration=123.0,
            )
        ]
    )


def test_no_generic_arg():
    @dataclass
    class NoGenericArgsNode(BaseNode):
        async def run(self, ctx: GraphRunContext) -> NoGenericArgsNode:
            return NoGenericArgsNode()

    g = Graph(nodes=[NoGenericArgsNode])
    assert g._inferred_types == (None, None)

    g = Graph(nodes=[NoGenericArgsNode], run_end_type=None)  # pyright: ignore[reportArgumentType]

    assert g._inferred_types == (None, None)

    custom_history = [
        {
            'state': None,
            'node': {'node_id': 'NoGenericArgsNode'},
            'start_ts': '2025-01-01T00:00:00Z',
            'duration': 123,
            'kind': 'node',
        },
    ]

    sp = FullStatePersistence()
    g.set_persistence_types(sp)
    sp.load_json(json.dumps(custom_history))

    assert sp.history == snapshot(
        [
            NodeSnapshot(
                state=None,
                node=NoGenericArgsNode(),
                start_ts=datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc),
                duration=123.0,
            )
        ]
    )
