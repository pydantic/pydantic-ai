from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from pydantic_graph import (
    BaseNode,
    End,
    EndSnapshot,
    Graph,
    GraphRunContext,
    NodeSnapshot,
)
from pydantic_graph.persistence import _utils as persistence_utils
from pydantic_graph.persistence.db import SQLiteStatePersistence

pytestmark = pytest.mark.anyio


# Define the same toy graph used in the file-based persistence tests
@dataclass
class Float2String(BaseNode[None, None, None]):
    input_data: float

    async def run(self, ctx: GraphRunContext) -> String2Length:  # forward reference
        return String2Length(str(self.input_data))


@dataclass
class String2Length(BaseNode[None, None, None]):
    input_data: str

    async def run(self, ctx: GraphRunContext) -> Double:  # forward reference
        return Double(len(self.input_data))


@dataclass
class Double(BaseNode[None, None, int]):
    input_data: int

    async def run(self, ctx: GraphRunContext) -> String2Length | End[int]:
        # Simulate a possible branch but for this test we always return an End
        if self.input_data == 7:
            return String2Length('x' * 21)
        else:
            return End(self.input_data * 2)


async def test_sqlite_run(tmp_path: Path) -> None:
    """Run a simple graph and ensure the output and snapshots are correct."""
    my_graph = Graph(nodes=(Float2String, String2Length, Double))
    db_file = tmp_path / 'new_test.db'
    run_id = '1234'
    persistence = SQLiteStatePersistence(db_file=db_file, run_id=run_id)
    with persistence_utils.set_nodes_type_context([Float2String, String2Length, Double]):
        persistence.set_types(type(None), int)
    result = await my_graph.run(Float2String(3.14), persistence=persistence)
    assert result.output == 8  # len("3.14") * 2
    assert my_graph.name == 'my_graph'

    snapshots = await persistence.load_all()
    assert len(snapshots) == 4
    # First three snapshots are NodeSnapshot in order Float2String -> String2Length -> Double
    assert isinstance(snapshots[0], NodeSnapshot)
    assert isinstance(snapshots[1], NodeSnapshot)
    assert isinstance(snapshots[2], NodeSnapshot)
    # Last snapshot is an EndSnapshot
    assert isinstance(snapshots[3], EndSnapshot)

    # Verify statuses and that ids/start_ts/duration fields are present
    for snap in snapshots[:3]:
        assert snap.status == 'success'
        assert snap.start_ts is not None
        assert snap.duration is not None
        assert isinstance(snap.id, str) and snap.id
    end_snap = snapshots[3]
    assert isinstance(end_snap.result, End)
    assert end_snap.result.data == 8


async def test_sqlite_next_from_persistence(tmp_path: Path) -> None:
    """Ensure iterating from persistence returns nodes in order with matching snapshot ids."""
    my_graph = Graph(nodes=(Float2String, String2Length, Double))
    db_file = tmp_path / 'new_test.db'
    run_id = '1234'
    persistence = SQLiteStatePersistence(db_file=db_file, run_id=run_id)
    with persistence_utils.set_nodes_type_context([Float2String, String2Length, Double]):
        persistence.set_types(type(None), int)
    # First run: advance one step and get the snapshot id
    async with my_graph.iter(Float2String(3.14), persistence=persistence) as run:
        node = await run.next()
        assert isinstance(node, String2Length)
        first_snapshot_id = node.get_snapshot_id()
        assert isinstance(first_snapshot_id, str) and first_snapshot_id
        assert my_graph.name == 'my_graph'

    # Resume from persistence: should pick up at the Double node, then End
    async with my_graph.iter_from_persistence(persistence) as run:
        node = await run.next()
        assert isinstance(node, Double)
        double_snapshot_id = node.get_snapshot_id()
        assert double_snapshot_id and double_snapshot_id != first_snapshot_id
        node = await run.next()
        assert isinstance(node, End)
        end_snapshot_id = node.get_snapshot_id()
        # Ensure all three IDs are distinct
        assert end_snapshot_id and end_snapshot_id not in {first_snapshot_id, double_snapshot_id}

    # Confirm that all snapshots are persisted
    snapshots = await persistence.load_all()
    assert len(snapshots) == 4
    assert isinstance(snapshots[3], EndSnapshot)


async def test_sqlite_node_error(tmp_path: Path) -> None:
    """Ensure that errors in nodes are recorded with error status."""

    @dataclass
    class Foo(BaseNode[None, None, None]):
        async def run(self, ctx: GraphRunContext) -> Bar:  # forward reference
            return Bar()

    @dataclass
    class Bar(BaseNode[None, None, None]):
        async def run(self, ctx: GraphRunContext) -> End[None]:
            raise RuntimeError('test error')

    my_graph = Graph(nodes=(Foo, Bar))
    db_file = tmp_path / 'new_test.db'
    run_id = '12345'
    persistence = SQLiteStatePersistence(db_file=db_file, run_id=run_id)
    persistence.set_graph_types(my_graph)

    with pytest.raises(RuntimeError, match='test error'):
        await my_graph.run(Foo(), persistence=persistence)

    snapshots = await persistence.load_all()
    assert len(snapshots) == 2
    assert isinstance(snapshots[0], NodeSnapshot)
    assert snapshots[0].status == 'success'
    assert isinstance(snapshots[1], NodeSnapshot)
    assert snapshots[1].status == 'error'


async def test_sqlite_record_lookup_error(tmp_path: Path) -> None:
    """record_run should raise a LookupError if the snapshot_id does not exist."""
    _ = Graph(nodes=(Float2String, String2Length, Double))
    db_file = tmp_path / 'new_test.db'
    run_id = '1234'
    persistence = SQLiteStatePersistence(db_file=db_file, run_id=run_id)
    with persistence_utils.set_nodes_type_context([Float2String, String2Length, Double]):
        persistence.set_types(type(None), int)
    with pytest.raises(LookupError, match="No snapshot found with id='foobar'"):
        async with persistence.record_run('foobar'):
            pass
