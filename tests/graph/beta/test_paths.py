"""Tests for pydantic_graph.beta.paths module."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pydantic_graph.beta import GraphBuilder, StepContext
from pydantic_graph.beta.id_types import ForkID, NodeID
from pydantic_graph.beta.paths import (
    BroadcastMarker,
    DestinationMarker,
    LabelMarker,
    MapMarker,
    Path,
    PathBuilder,
    TransformMarker,
)

pytestmark = pytest.mark.anyio


@dataclass
class TestState:
    value: int = 0


async def test_path_last_fork_with_no_forks():
    """Test Path.last_fork property when there are no forks."""
    path = Path(items=[LabelMarker('test'), DestinationMarker(NodeID('dest'))])
    assert path.last_fork is None


async def test_path_last_fork_with_broadcast():
    """Test Path.last_fork property with a BroadcastMarker."""
    broadcast = BroadcastMarker(paths=[], fork_id=ForkID(NodeID('fork1')))
    path = Path(items=[broadcast, LabelMarker('after fork')])
    assert path.last_fork is broadcast


async def test_path_last_fork_with_map():
    """Test Path.last_fork property with a MapMarker."""
    map = MapMarker(fork_id=ForkID(NodeID('map1')), downstream_join_id=None)
    path = Path(items=[map, LabelMarker('after map')])
    assert path.last_fork is map


async def test_path_builder_last_fork_no_forks():
    """Test PathBuilder.last_fork property when there are no forks."""
    builder = PathBuilder[TestState, None, int](working_items=[LabelMarker('test')])
    assert builder.last_fork is None


async def test_path_builder_last_fork_with_map():
    """Test PathBuilder.last_fork property with a MapMarker."""
    map = MapMarker(fork_id=ForkID(NodeID('map1')), downstream_join_id=None)
    builder = PathBuilder[TestState, None, int](working_items=[map, LabelMarker('test')])
    assert builder.last_fork is map


async def test_path_builder_transform():
    """Test PathBuilder.transform method."""

    async def transform_func(ctx, input_data):
        return input_data * 2

    builder = PathBuilder[TestState, None, int](working_items=[])
    new_builder = builder.transform(transform_func)

    assert len(new_builder.working_items) == 1
    assert isinstance(new_builder.working_items[0], TransformMarker)


async def test_edge_path_builder_transform():
    """Test EdgePathBuilder.transform method creates proper path."""
    g = GraphBuilder(state_type=TestState, output_type=int)

    @g.step
    async def step_a(ctx: StepContext[TestState, None, None]) -> int:
        return 10

    @g.step
    async def step_b(ctx: StepContext[TestState, None, int]) -> int:
        return ctx.inputs * 3

    async def double(ctx: StepContext[TestState, None, int], value: int) -> int:
        return value * 2

    # Build graph with transform in the path
    g.add(
        g.edge_from(g.start_node).to(step_a),
        g.edge_from(step_a).transform(double).to(step_b),
        g.edge_from(step_b).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=TestState())
    assert result == 60  # 10 * 2 * 3


async def test_edge_path_builder_last_fork_id_none():
    """Test EdgePathBuilder.last_fork_id when there are no forks."""
    g = GraphBuilder(state_type=TestState, output_type=int)

    @g.step
    async def step_a(ctx: StepContext[TestState, None, None]) -> int:
        return 10

    edge_builder = g.edge_from(g.start_node)
    # Access internal path_builder to test last_fork_id
    assert edge_builder.last_fork_id is None


async def test_edge_path_builder_last_fork_id_with_map():
    """Test EdgePathBuilder.last_fork_id after a map operation."""
    g = GraphBuilder(state_type=TestState, output_type=int)

    @g.step
    async def list_step(ctx: StepContext[TestState, None, None]) -> list[int]:
        return [1, 2, 3]

    @g.step
    async def process_item(ctx: StepContext[TestState, None, int]) -> int:
        return ctx.inputs * 2

    edge_builder = g.edge_from(list_step).map()
    fork_id = edge_builder.last_fork_id
    assert fork_id is not None
    assert isinstance(fork_id, ForkID)


async def test_path_builder_label():
    """Test PathBuilder.label method."""
    builder = PathBuilder[TestState, None, int](working_items=[])
    new_builder = builder.label('my label')

    assert len(new_builder.working_items) == 1
    assert isinstance(new_builder.working_items[0], LabelMarker)
    assert new_builder.working_items[0].label == 'my label'


async def test_path_next_path():
    """Test Path.next_path removes first item."""
    items = [LabelMarker('first'), LabelMarker('second'), DestinationMarker(NodeID('dest'))]
    path = Path(items=items)

    next_path = path.next_path
    assert len(next_path.items) == 2
    assert next_path.items[0] == items[1]
    assert next_path.items[1] == items[2]
