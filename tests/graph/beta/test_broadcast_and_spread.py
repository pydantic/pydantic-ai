"""Tests for broadcast (parallel) and map (fan-out) operations."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from pydantic_graph.beta import GraphBuilder, StepContext
from pydantic_graph.beta.join import reduce_list_append

pytestmark = pytest.mark.anyio


@dataclass
class CounterState:
    values: list[int] = field(default_factory=list)


async def test_broadcast_to_multiple_steps():
    """Test broadcasting the same data to multiple parallel steps."""
    g = GraphBuilder(state_type=CounterState, output_type=list[int])

    @g.step
    async def source(ctx: StepContext[CounterState, None, None]) -> int:
        return 10

    @g.step
    async def add_one(ctx: StepContext[CounterState, None, int]) -> int:
        return ctx.inputs + 1

    @g.step
    async def add_two(ctx: StepContext[CounterState, None, int]) -> int:
        return ctx.inputs + 2

    @g.step
    async def add_three(ctx: StepContext[CounterState, None, int]) -> int:
        return ctx.inputs + 3

    collect = g.join(reduce_list_append, initial_factory=list[int])

    g.add(
        g.edge_from(g.start_node).to(source),
        g.edge_from(source).to(add_one, add_two, add_three),
        g.edge_from(add_one, add_two, add_three).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=CounterState())
    # Results can be in any order due to parallel execution
    assert sorted(result) == [11, 12, 13]


async def test_map_over_list():
    """Test mapping a list to process items in parallel."""
    g = GraphBuilder(state_type=CounterState, output_type=list[int])

    @g.step
    async def generate_list(ctx: StepContext[CounterState, None, None]) -> list[int]:
        return [1, 2, 3, 4, 5]

    @g.step
    async def square(ctx: StepContext[CounterState, None, int]) -> int:
        return ctx.inputs * ctx.inputs

    collect = g.join(reduce_list_append, initial_factory=list[int])

    g.add_mapping_edge(generate_list, square)
    g.add(
        g.edge_from(g.start_node).to(generate_list),
        g.edge_from(square).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=CounterState())
    assert sorted(result) == [1, 4, 9, 16, 25]


async def test_map_with_labels():
    """Test map operation with labeled edges."""
    g = GraphBuilder(state_type=CounterState, output_type=list[str])

    @g.step
    async def generate_numbers(ctx: StepContext[CounterState, None, None]) -> list[int]:
        return [10, 20, 30]

    @g.step
    async def stringify(ctx: StepContext[CounterState, None, int]) -> str:
        return f'Value: {ctx.inputs}'

    collect = g.join(reduce_list_append, initial_factory=list[str])

    g.add_mapping_edge(
        generate_numbers,
        stringify,
        pre_map_label='before map',
        post_map_label='after map',
    )
    g.add(
        g.edge_from(g.start_node).to(generate_numbers),
        g.edge_from(stringify).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=CounterState())
    assert sorted(result) == ['Value: 10', 'Value: 20', 'Value: 30']


async def test_map_empty_list():
    """Test mapping an empty list."""
    g = GraphBuilder(state_type=CounterState, output_type=list[int])

    @g.step
    async def generate_empty(ctx: StepContext[CounterState, None, None]) -> list[int]:
        return []

    @g.step
    async def double(ctx: StepContext[CounterState, None, int]) -> int:
        return ctx.inputs * 2  # pragma: no cover

    collect = g.join(reduce_list_append, initial_factory=list[int])

    g.add_mapping_edge(generate_empty, double, downstream_join_id=collect.id)
    g.add(
        g.edge_from(g.start_node).to(generate_empty),
        g.edge_from(double).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=CounterState())
    assert result == []


async def test_nested_broadcasts():
    """Test nested broadcast operations."""
    g = GraphBuilder(state_type=CounterState, output_type=list[int])

    @g.step
    async def start_value(ctx: StepContext[CounterState, None, None]) -> int:
        return 5

    @g.step
    async def path_a1(ctx: StepContext[CounterState, None, int]) -> int:
        return ctx.inputs + 1

    @g.step
    async def path_a2(ctx: StepContext[CounterState, None, int]) -> int:
        return ctx.inputs + 10

    @g.step
    async def path_b1(ctx: StepContext[CounterState, None, int]) -> int:
        return ctx.inputs * 2

    @g.step
    async def path_b2(ctx: StepContext[CounterState, None, int]) -> int:
        return ctx.inputs * 3

    collect = g.join(reduce_list_append, initial_factory=list[int])

    g.add(
        g.edge_from(g.start_node).to(start_value),
        g.edge_from(start_value).to(path_a1, path_b1),
        g.edge_from(path_a1).to(path_a2),
        g.edge_from(path_b1).to(path_b2),
        g.edge_from(path_a2, path_b2).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=CounterState())
    # path_a: 5 + 1 + 10 = 16
    # path_b: 5 * 2 * 3 = 30
    assert sorted(result) == [16, 30]


async def test_map_then_broadcast():
    """Test mapping followed by broadcasting from each map item."""
    g = GraphBuilder(state_type=CounterState, output_type=list[int])

    @g.step
    async def generate_list(ctx: StepContext[CounterState, None, None]) -> list[int]:
        return [10, 20]

    @g.step
    async def add_one(ctx: StepContext[CounterState, None, int]) -> int:
        return ctx.inputs + 1

    @g.step
    async def add_two(ctx: StepContext[CounterState, None, int]) -> int:
        return ctx.inputs + 2

    collect = g.join(reduce_list_append, initial_factory=list[int])

    g.add(
        g.edge_from(g.start_node).to(generate_list),
        g.edge_from(generate_list).map().to(add_one, add_two),
        g.edge_from(add_one, add_two).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=CounterState())
    # From 10: 11, 12
    # From 20: 21, 22
    assert sorted(result) == [11, 12, 21, 22]


async def test_multiple_sequential_maps():
    """Test multiple sequential map operations."""
    g = GraphBuilder(state_type=CounterState, output_type=list[str])

    @g.step
    async def generate_pairs(ctx: StepContext[CounterState, None, None]) -> list[tuple[int, int]]:
        return [(1, 2), (3, 4)]

    @g.step
    async def unpack_pair(ctx: StepContext[CounterState, None, tuple[int, int]]) -> list[int]:
        return [ctx.inputs[0], ctx.inputs[1]]

    @g.step
    async def stringify(ctx: StepContext[CounterState, None, int]) -> str:
        return f'num:{ctx.inputs}'

    collect = g.join(reduce_list_append, initial_factory=list[str])

    g.add(
        g.edge_from(g.start_node).to(generate_pairs),
        g.edge_from(generate_pairs).map().to(unpack_pair),
        g.edge_from(unpack_pair).map().to(stringify),
        g.edge_from(stringify).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=CounterState())
    assert sorted(result) == ['num:1', 'num:2', 'num:3', 'num:4']


async def test_broadcast_with_different_outputs():
    """Test that broadcasts can produce different types of outputs."""
    g = GraphBuilder(state_type=CounterState, output_type=list[int | str])

    @g.step
    async def source(ctx: StepContext[CounterState, None, None]) -> int:
        return 42

    @g.step
    async def return_int(ctx: StepContext[CounterState, None, int]) -> int:
        return ctx.inputs

    @g.step
    async def return_str(ctx: StepContext[CounterState, None, int]) -> str:
        return str(ctx.inputs)

    collect = g.join(reduce_list_append, initial_factory=list[int | str])

    g.add(
        g.edge_from(g.start_node).to(source),
        g.edge_from(source).to(return_int, return_str),
        g.edge_from(return_int, return_str).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=CounterState())
    # Order may vary
    assert set(result) == {42, '42'}
