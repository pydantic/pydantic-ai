from __future__ import annotations as _annotations

from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Union

import pytest
from inline_snapshot import snapshot

from pydantic_graph import (
    BaseNode,
    End,
    EndSnapshot,
    Graph,
    GraphRunContext,
    NodeSnapshot,
)
from pydantic_graph.persistence.file import FileStatePersistence

from ..conftest import IsFloat, IsNow

pytestmark = pytest.mark.anyio


@dataclass
class Float2String(BaseNode):
    input_data: float

    async def run(self, ctx: GraphRunContext) -> String2Length:
        return String2Length(str(self.input_data))


@dataclass
class String2Length(BaseNode):
    input_data: str

    async def run(self, ctx: GraphRunContext) -> Double:
        return Double(len(self.input_data))


@dataclass
class Double(BaseNode[None, None, int]):
    input_data: int

    async def run(self, ctx: GraphRunContext) -> Union[String2Length, End[int]]:  # noqa: UP007
        if self.input_data == 7:  # pragma: no cover
            return String2Length('x' * 21)
        else:
            return End(self.input_data * 2)


async def test_run(tmp_path: Path, mock_snapshot_id: object):
    my_graph = Graph(nodes=(Float2String, String2Length, Double))
    p = tmp_path / 'test_graph.json'
    persistence = FileStatePersistence(p)
    result = await my_graph.run(Float2String(3.14), persistence=persistence)
    # len('3.14') * 2 == 8
    assert result.output == 8
    assert my_graph.name == 'my_graph'
    assert await persistence.load_all() == snapshot(
        [
            NodeSnapshot(
                state=None,
                node=Float2String(input_data=3.14),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
                status='success',
                id='Float2String:1',
            ),
            NodeSnapshot(
                state=None,
                node=String2Length(input_data='3.14'),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
                status='success',
                id='String2Length:2',
            ),
            NodeSnapshot(
                state=None,
                node=Double(input_data=4),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
                status='success',
                id='Double:3',
            ),
            EndSnapshot(state=None, result=End(data=8), ts=IsNow(tz=timezone.utc), id='end:4'),
        ]
    )


async def test_next_from_persistence(tmp_path: Path, mock_snapshot_id: object):
    my_graph = Graph(nodes=(Float2String, String2Length, Double))
    p = tmp_path / 'test_graph.json'
    persistence = FileStatePersistence(p)

    async with my_graph.iter(Float2String(3.14), persistence=persistence) as run:
        node = await run.next()
        assert node == snapshot(String2Length(input_data='3.14'))
        assert node.get_snapshot_id() == snapshot('String2Length:2')
        assert my_graph.name == 'my_graph'

    async with my_graph.iter_from_persistence(persistence) as run:
        node = await run.next()
        assert node == snapshot(Double(input_data=4))
        assert node.get_snapshot_id() == snapshot('Double:3')

        node = await run.next()
        assert node == snapshot(End(data=8))
        assert node.get_snapshot_id() == snapshot('end:4')

    assert await persistence.load_all() == snapshot(
        [
            NodeSnapshot(
                state=None,
                node=Float2String(input_data=3.14),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
                status='success',
                id='Float2String:1',
            ),
            NodeSnapshot(
                state=None,
                node=String2Length(input_data='3.14'),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
                status='success',
                id='String2Length:2',
            ),
            NodeSnapshot(
                state=None,
                node=Double(input_data=4),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
                status='success',
                id='Double:3',
            ),
            EndSnapshot(state=None, result=End(data=8), ts=IsNow(tz=timezone.utc), id='end:4'),
        ]
    )


async def test_node_error(tmp_path: Path, mock_snapshot_id: object):
    @dataclass
    class Foo(BaseNode):
        async def run(self, ctx: GraphRunContext) -> Bar:
            return Bar()

    @dataclass
    class Bar(BaseNode[None, None, None]):
        async def run(self, ctx: GraphRunContext) -> End[None]:
            raise RuntimeError('test error')

    g = Graph(nodes=(Foo, Bar))
    p = tmp_path / 'test_graph.json'
    persistence = FileStatePersistence(p)
    with pytest.raises(RuntimeError, match='test error'):
        await g.run(Foo(), persistence=persistence)

    assert await persistence.load_all() == snapshot(
        [
            NodeSnapshot(
                state=None,
                node=Foo(),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
                status='success',
                id='Foo:1',
            ),
            NodeSnapshot(
                state=None,
                node=Bar(),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
                status='error',
                id='Bar:2',
            ),
        ]
    )


async def test_lock_timeout(tmp_path: Path):
    p = tmp_path / 'test_graph.json'
    persistence = FileStatePersistence(p)
    async with persistence._lock():  # type: ignore[reportPrivateUsage]
        pass

    async with persistence._lock():  # type: ignore[reportPrivateUsage]
        with pytest.raises(TimeoutError):
            async with persistence._lock(timeout=0.1):  # type: ignore[reportPrivateUsage]
                pass


async def test_record_lookup_error(tmp_path: Path):
    p = tmp_path / 'test_graph.json'
    persistence = FileStatePersistence(p)
    my_graph = Graph(nodes=(Float2String, String2Length, Double))
    persistence.set_graph_types(my_graph)
    persistence.set_graph_types(my_graph)

    with pytest.raises(LookupError, match="No snapshot found with id='foobar'"):
        async with persistence.record_run('foobar'):
            pass
