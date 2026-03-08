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
    PathItem,
    TransformMarker,
)

pytestmark = pytest.mark.anyio


@dataclass
class MyState:
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


async def test_path_builder_transform():
    """Test PathBuilder.transform method."""

    async def transform_func(ctx: StepContext[MyState, None, int]) -> int:
        return ctx.inputs * 2  # pragma: no cover

    builder = PathBuilder[MyState, None, int](working_items=[])
    new_builder = builder.transform(transform_func)

    assert len(new_builder.working_items) == 1
    assert isinstance(new_builder.working_items[0], TransformMarker)


async def test_edge_path_builder_transform():
    """Test EdgePathBuilder.transform method creates proper path."""
    g = GraphBuilder(state_type=MyState, output_type=int)

    @g.step
    async def step_a(ctx: StepContext[MyState, None, None]) -> int:
        return 10

    @g.step
    async def step_b(ctx: StepContext[MyState, None, int]) -> int:
        return ctx.inputs * 3

    def double(ctx: StepContext[MyState, None, int]) -> int:
        return ctx.inputs * 2

    # Build graph with transform in the path
    g.add(
        g.edge_from(g.start_node).to(step_a),
        g.edge_from(step_a).transform(double).to(step_b),
        g.edge_from(step_b).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=MyState())
    assert result == 60  # 10 * 2 * 3


async def test_path_builder_label():
    """Test PathBuilder.label method."""
    builder = PathBuilder[MyState, None, int](working_items=[])
    new_builder = builder.label('my label')

    assert len(new_builder.working_items) == 1
    assert isinstance(new_builder.working_items[0], LabelMarker)
    assert new_builder.working_items[0].label == 'my label'


async def test_path_next_path():
    """Test Path.next_path removes first item."""
    items: list[PathItem] = [LabelMarker('first'), LabelMarker('second'), DestinationMarker(NodeID('dest'))]
    path = Path(items=items)

    next_path = path.next_path
    assert len(next_path.items) == 2
    assert next_path.items[0] == items[1]
    assert next_path.items[1] == items[2]
