# pyright: reportPrivateUsage=false
from __future__ import annotations as _annotations

import json
from copy import deepcopy
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
    SimpleStatePersistence,
)
from pydantic_graph.exceptions import GraphNodeStatusError, GraphRuntimeError
from pydantic_graph.persistence import BaseStatePersistence, build_snapshot_list_type_adapter

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
async def test_dump_load_state(graph: Graph[MyState, None, int], mock_snapshot_id: object):
    sp = FullStatePersistence()
    result = await graph.run(Foo(), state=MyState(1, ''), persistence=sp)
    assert result.output == snapshot(4)
    assert result.state == snapshot(MyState(x=2, y='y'))
    assert await sp.load_all() == snapshot(
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
            EndSnapshot(
                state=MyState(x=2, y='y'),
                result=End(data=4),
                ts=IsNow(tz=timezone.utc),
                id='end:3',
            ),
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
                'status': 'success',
                'kind': 'node',
                'id': 'Foo:1',
            },
            {
                'state': {'x': 2, 'y': ''},
                'node': {'node_id': 'Bar'},
                'start_ts': IsStr(regex=r'20\d\d-\d\d-\d\dT.+'),
                'duration': IsFloat(),
                'status': 'success',
                'kind': 'node',
                'id': 'Bar:2',
            },
            {
                'state': {'x': 2, 'y': 'y'},
                'result': {'data': 4},
                'ts': IsStr(regex=r'20\d\d-\d\d-\d\dT.+'),
                'kind': 'end',
                'id': 'end:3',
            },
        ]
    )

    sp2 = FullStatePersistence()
    sp2.set_graph_types(graph)

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
    sp3.set_graph_types(graph)
    sp3.load_json(json.dumps(custom_history))
    assert sp3.history == snapshot(
        [
            NodeSnapshot(
                state=MyState(x=2, y=''),
                node=Foo(),
                start_ts=datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc),
                duration=123.0,
                id='Foo:4',
            ),
            EndSnapshot(
                state=MyState(x=42, y='new'),
                result=End(data=42),
                ts=datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc),
                id='end:5',
            ),
        ]
    )


def test_one_node(mock_snapshot_id: object):
    @dataclass
    class MyNode(BaseNode[None, None, int]):
        node_field: int

        async def run(self, ctx: GraphRunContext) -> End[int]:  # pragma: no cover
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
    sp.set_graph_types(g)
    sp.load_json(json.dumps(custom_history))
    assert sp.history == snapshot(
        [
            NodeSnapshot(
                state=None,
                node=MyNode(node_field=42),
                start_ts=datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc),
                duration=123.0,
                id='MyNode:1',
            )
        ]
    )


def test_no_generic_arg(mock_snapshot_id: object):
    @dataclass
    class NoGenericArgsNode(BaseNode):
        async def run(self, ctx: GraphRunContext) -> NoGenericArgsNode:  # pragma: no cover
            return NoGenericArgsNode()

    g = Graph(nodes=[NoGenericArgsNode])
    assert g.inferred_types == (None, None)

    g = Graph(nodes=[NoGenericArgsNode], run_end_type=None)  # pyright: ignore[reportArgumentType]

    assert g.inferred_types == (None, None)

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
    sp.set_graph_types(g)
    sp.load_json(json.dumps(custom_history))

    assert sp.history == snapshot(
        [
            NodeSnapshot(
                state=None,
                node=NoGenericArgsNode(),
                start_ts=datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc),
                duration=123.0,
                id='NoGenericArgsNode:1',
            )
        ]
    )


async def test_node_error(mock_snapshot_id: object):
    @dataclass
    class Foo(BaseNode):
        async def run(self, ctx: GraphRunContext) -> Spam:
            return Spam()

    @dataclass
    class Spam(BaseNode[None, None, int]):
        async def run(self, ctx: GraphRunContext) -> End[int]:
            raise RuntimeError('test error')

    graph = Graph(nodes=[Foo, Spam])

    sp = FullStatePersistence()
    with pytest.raises(RuntimeError, match='test error'):
        await graph.run(Foo(), persistence=sp)

    assert sp.history == snapshot(
        [
            NodeSnapshot(
                state=None,
                node=Foo(),
                status='success',
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
                id='Foo:1',
            ),
            NodeSnapshot(
                state=None,
                node=Spam(),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
                status='error',
                id='Spam:2',
            ),
        ]
    )


async def test_rerun_node(mock_snapshot_id: object):
    @dataclass
    class Foo(BaseNode[None, None, int]):
        async def run(self, ctx: GraphRunContext) -> End[int]:
            return End(123)

    graph = Graph(nodes=[Foo])

    sp = FullStatePersistence()
    node = Foo()
    async with graph.iter(node, persistence=sp) as run:
        end = await run.next()
        assert end == snapshot(End(data=123))

    msg = "Incorrect snapshot status 'success', must be 'created' or 'pending'."
    with pytest.raises(GraphNodeStatusError, match=msg):
        async with graph.iter(node, persistence=sp) as run:
            await run.next()


@pytest.mark.parametrize('persistence_cls', [SimpleStatePersistence, FullStatePersistence])
async def test_next_from_persistence(persistence_cls: type[BaseStatePersistence[None, int]], mock_snapshot_id: object):
    @dataclass
    class Foo(BaseNode):
        async def run(self, ctx: GraphRunContext) -> Spam:
            return Spam()

    @dataclass
    class Spam(BaseNode[None, None, int]):
        async def run(self, ctx: GraphRunContext) -> End[int]:
            return End(123)

    g1 = Graph(nodes=[Foo, Spam])

    persistence = persistence_cls()
    node = Foo()
    assert g1.name is None
    await g1.initialize(node, persistence)
    assert g1.name == 'g1'

    async with g1.iter_from_persistence(persistence) as run:
        node = await run.next()
        assert node == Spam()
        end = await run.next()
        assert end == End(123)

    g2 = Graph(nodes=[Foo, Spam])
    persistence2 = persistence_cls()
    assert g2.name is None
    with pytest.raises(GraphRuntimeError, match='Unable to restore snapshot from state persistence.'):
        async with g2.iter_from_persistence(persistence2):
            pass
    assert g2.name == 'g2'


@pytest.mark.parametrize('persistence_cls', [SimpleStatePersistence, FullStatePersistence])
async def test_record_lookup_error(persistence_cls: type[BaseStatePersistence]):
    persistence = persistence_cls()
    my_graph = Graph(nodes=(Foo, Bar))
    persistence.set_graph_types(my_graph)
    persistence.set_graph_types(my_graph)

    with pytest.raises(LookupError, match="No snapshot found with id='foobar'"):
        async with persistence.record_run('foobar'):
            pass


def test_snapshot_type_adapter_error():
    with pytest.raises(RuntimeError, match='Unable to build a Pydantic schema for `BaseNode` without setting'):
        build_snapshot_list_type_adapter(int, int)


async def test_full_state_persistence_snapshot_state_stability():
    @dataclass
    class CountDownState:
        counter: int

    @dataclass
    class CountDown(BaseNode[CountDownState, None, int]):
        async def run(self, ctx: GraphRunContext[CountDownState]) -> CountDown | End[int]:
            if ctx.state.counter <= 0:
                return End(ctx.state.counter)
            ctx.state.counter -= 1
            return CountDown()

    persistence = FullStatePersistence()
    state = CountDownState(counter=3)
    count_down_graph = Graph(nodes=[CountDown])

    await count_down_graph.initialize(CountDown(), state=state, persistence=persistence)

    done = False
    while not done:
        history = deepcopy(persistence.history)
        async with count_down_graph.iter_from_persistence(persistence) as run:
            result = await run.next()
            done = isinstance(result, End)

        for i in range(len(history)):
            assert history[i].id == persistence.history[i].id
            assert history[i].state == persistence.history[i].state, 'State should not change'


async def test_simple_state_persistence_snapshot_state_stability():
    @dataclass
    class CountDownState:
        counter: int

    @dataclass
    class CountDown(BaseNode[CountDownState, None, int]):
        async def run(self, ctx: GraphRunContext[CountDownState]) -> CountDown | End[int]:
            ctx.state.counter -= 1
            return CountDown()

    persistence = SimpleStatePersistence()
    state = CountDownState(counter=3)
    count_down_graph = Graph(nodes=[CountDown])

    await count_down_graph.initialize(CountDown(), state=state, persistence=persistence)

    last_snapshot = persistence.last_snapshot
    async with count_down_graph.iter_from_persistence(persistence) as run:
        await run.next()

    assert last_snapshot and last_snapshot.state.counter == 3
    assert persistence.last_snapshot and persistence.last_snapshot.state.counter == 2
