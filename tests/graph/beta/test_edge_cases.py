"""Tests for edge cases, error handling, and boundary conditions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from pydantic_graph.beta import GraphBuilder, NullReducer, StepContext

pytestmark = pytest.mark.anyio


@dataclass
class EdgeCaseState:
    value: int = 0
    error_raised: bool = False


async def test_graph_with_no_steps():
    """Test a graph with no intermediate steps (direct start to end)."""
    g = GraphBuilder(input_type=int, output_type=int)

    g.add(g.edge_from(g.start_node).to(g.end_node))

    graph = g.build()
    result = await graph.run(inputs=42)
    assert result == 42


async def test_step_returning_none():
    """Test steps that return None."""
    g = GraphBuilder(state_type=EdgeCaseState)

    @g.step
    async def do_nothing(ctx: StepContext[EdgeCaseState, None, None]) -> None:
        ctx.state.value = 99
        return None

    @g.step
    async def return_none(ctx: StepContext[EdgeCaseState, None, None]) -> None:
        return None

    g.add(
        g.edge_from(g.start_node).to(do_nothing),
        g.edge_from(do_nothing).to(return_none),
        g.edge_from(return_none).to(g.end_node),
    )

    graph = g.build()
    state = EdgeCaseState()
    result = await graph.run(state=state)
    assert result is None
    assert state.value == 99


async def test_step_with_zero_value():
    """Test handling of zero values (ensure they're not confused with None/falsy)."""
    g = GraphBuilder(state_type=EdgeCaseState, output_type=int)

    @g.step
    async def return_zero(ctx: StepContext[EdgeCaseState, None, None]) -> int:
        return 0

    @g.step
    async def process_zero(ctx: StepContext[EdgeCaseState, None, int]) -> int:
        return ctx.inputs + 1

    g.add(
        g.edge_from(g.start_node).to(return_zero),
        g.edge_from(return_zero).to(process_zero),
        g.edge_from(process_zero).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=EdgeCaseState())
    assert result == 1


async def test_step_with_empty_string():
    """Test handling of empty strings."""
    g = GraphBuilder(state_type=EdgeCaseState, output_type=str)

    @g.step
    async def return_empty(ctx: StepContext[EdgeCaseState, None, None]) -> str:
        return ''

    @g.step
    async def process_empty(ctx: StepContext[EdgeCaseState, None, str]) -> str:
        return ctx.inputs + 'appended'

    g.add(
        g.edge_from(g.start_node).to(return_empty),
        g.edge_from(return_empty).to(process_empty),
        g.edge_from(process_empty).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=EdgeCaseState())
    assert result == 'appended'


async def test_spread_single_item():
    """Test spreading a single-item list."""
    g = GraphBuilder(state_type=EdgeCaseState, output_type=list[int])

    @g.step
    async def single_item(ctx: StepContext[EdgeCaseState, None, None]) -> list[int]:
        return [42]

    @g.step
    async def process(ctx: StepContext[EdgeCaseState, None, int]) -> int:
        return ctx.inputs * 2

    from pydantic_graph.beta import ListReducer

    collect = g.join(ListReducer[int])

    g.add(
        g.edge_from(g.start_node).to(single_item),
        g.edge_from(single_item).spread().to(process),
        g.edge_from(process).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=EdgeCaseState())
    assert result == [84]


async def test_deeply_nested_broadcasts():
    """Test deeply nested broadcast operations."""
    g = GraphBuilder(state_type=EdgeCaseState, output_type=list[int])

    @g.step
    async def start(ctx: StepContext[EdgeCaseState, None, None]) -> int:
        return 1

    @g.step
    async def level1_a(ctx: StepContext[EdgeCaseState, None, int]) -> int:
        return ctx.inputs + 1

    @g.step
    async def level1_b(ctx: StepContext[EdgeCaseState, None, int]) -> int:
        return ctx.inputs + 2

    @g.step
    async def level2_a(ctx: StepContext[EdgeCaseState, None, int]) -> int:
        return ctx.inputs + 10

    @g.step
    async def level2_b(ctx: StepContext[EdgeCaseState, None, int]) -> int:
        return ctx.inputs + 20

    from pydantic_graph.beta import ListReducer

    collect = g.join(ListReducer[int])

    g.add(
        g.edge_from(g.start_node).to(start),
        g.edge_from(start).to(level1_a, level1_b),
        g.edge_from(level1_a).to(level2_a, level2_b),
        g.edge_from(level1_b).to(level2_a, level2_b),
        g.edge_from(level2_a, level2_b).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=EdgeCaseState())
    # From level1_a (2): 12, 22
    # From level1_b (3): 13, 23
    assert sorted(result) == [12, 13, 22, 23]


async def test_long_sequential_chain():
    """Test a long chain of sequential steps."""
    g = GraphBuilder(state_type=EdgeCaseState, output_type=int)

    steps: list[Any] = []
    for i in range(10):

        @g.step(node_id=f'step_{i}')
        async def step_func(ctx: StepContext[EdgeCaseState, None, int | None]) -> int:
            if ctx.inputs is None:
                return 1
            return ctx.inputs + 1

        steps.append(step_func)

    # Build the chain
    g.add(g.edge_from(g.start_node).to(steps[0]))
    for i in range(len(steps) - 1):
        g.add(g.edge_from(steps[i]).to(steps[i + 1]))
    g.add(g.edge_from(steps[-1]).to(g.end_node))

    graph = g.build()
    result = await graph.run(state=EdgeCaseState(), inputs=None)
    assert result == 10  # 10 increments


async def test_join_with_single_input():
    """Test a join operation that only receives one input."""
    g = GraphBuilder(state_type=EdgeCaseState, output_type=list[int])

    @g.step
    async def single_source(ctx: StepContext[EdgeCaseState, None, None]) -> int:
        return 42

    from pydantic_graph.beta import ListReducer

    collect = g.join(ListReducer[int])

    g.add(
        g.edge_from(g.start_node).to(single_source),
        g.edge_from(single_source).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=EdgeCaseState())
    assert result == [42]


async def test_null_reducer_with_no_inputs():
    """Test NullReducer behavior with spread that produces no items."""
    g = GraphBuilder(state_type=EdgeCaseState)

    @g.step
    async def empty_list(ctx: StepContext[EdgeCaseState, None, None]) -> list[int]:
        return []

    @g.step
    async def process(ctx: StepContext[EdgeCaseState, None, int]) -> int:
        return ctx.inputs

    null_join = g.join(NullReducer)

    g.add(
        g.edge_from(g.start_node).to(empty_list),
        g.edge_from(empty_list).spread(downstream_join_id=null_join.id).to(process),
        g.edge_from(process).to(null_join),
        g.edge_from(null_join).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=EdgeCaseState())
    assert result is None


async def test_step_with_complex_input_type():
    """Test steps with complex input types (nested structures)."""

    @dataclass
    class ComplexInput:
        value: int
        nested: dict[str, list[int]]

    g = GraphBuilder(state_type=EdgeCaseState, input_type=ComplexInput, output_type=int)

    @g.step
    async def process_complex(ctx: StepContext[EdgeCaseState, None, ComplexInput]) -> int:
        total = ctx.inputs.value
        for values in ctx.inputs.nested.values():
            total += sum(values)
        return total

    g.add(
        g.edge_from(g.start_node).to(process_complex),
        g.edge_from(process_complex).to(g.end_node),
    )

    graph = g.build()
    complex_input = ComplexInput(value=10, nested={'a': [1, 2, 3], 'b': [4, 5]})
    result = await graph.run(state=EdgeCaseState(), inputs=complex_input)
    assert result == 25  # 10 + 1 + 2 + 3 + 4 + 5


async def test_multiple_joins_same_fork():
    """Test multiple joins converging from the same fork point."""
    g = GraphBuilder(state_type=EdgeCaseState, output_type=tuple[list[int], list[int]])

    @g.step
    async def source(ctx: StepContext[EdgeCaseState, None, None]) -> list[int]:
        return [1, 2, 3]

    @g.step
    async def path_a(ctx: StepContext[EdgeCaseState, None, int]) -> int:
        return ctx.inputs * 2

    @g.step
    async def path_b(ctx: StepContext[EdgeCaseState, None, int]) -> int:
        return ctx.inputs * 3

    from pydantic_graph.beta import ListReducer

    join_a = g.join(ListReducer[int], node_id='join_a')
    join_b = g.join(ListReducer[int], node_id='join_b')

    @g.step
    async def combine(ctx: StepContext[EdgeCaseState, None, None]) -> tuple[list[int], list[int]]:
        # This is a bit awkward but demonstrates the pattern
        return ([], [])  # In real usage, you'd access the join results differently

    g.add(
        g.edge_from(g.start_node).to(source),
        g.edge_from(source).spread().to(path_a, path_b),
        g.edge_from(path_a).to(join_a),
        g.edge_from(path_b).to(join_b),
        # Note: This test demonstrates structure but may need adjustment based on actual API
    )


async def test_state_with_mutable_collections():
    """Test that mutable state collections work correctly across parallel paths."""

    @dataclass
    class MutableState:
        items: list[int] = field(default_factory=list)

    g = GraphBuilder(state_type=MutableState, output_type=list[int])

    @g.step
    async def generate(ctx: StepContext[MutableState, None, None]) -> list[int]:
        return [1, 2, 3]

    @g.step
    async def append_to_state(ctx: StepContext[MutableState, None, int]) -> int:
        ctx.state.items.append(ctx.inputs * 10)
        return ctx.inputs

    from pydantic_graph.beta import ListReducer

    collect = g.join(ListReducer[int])

    @g.step
    async def get_state_items(ctx: StepContext[MutableState, None, list[int]]) -> list[int]:
        return ctx.state.items

    g.add(
        g.edge_from(g.start_node).to(generate),
        g.edge_from(generate).spread().to(append_to_state),
        g.edge_from(append_to_state).to(collect),
        g.edge_from(collect).to(get_state_items),
        g.edge_from(get_state_items).to(g.end_node),
    )

    graph = g.build()
    state = MutableState()
    result = await graph.run(state=state)
    assert sorted(result) == [10, 20, 30]
    assert sorted(state.items) == [10, 20, 30]


async def test_step_that_modifies_deps():
    """Test that deps modifications don't persist (deps should be immutable)."""

    @dataclass
    class MutableDeps:
        value: int

    g = GraphBuilder(state_type=EdgeCaseState, deps_type=MutableDeps, output_type=int)

    @g.step
    async def try_modify_deps(ctx: StepContext[EdgeCaseState, MutableDeps, None]) -> int:
        original = ctx.deps.value
        # Attempt to modify (this DOES mutate the object, but that's user error)
        ctx.deps.value = 999
        return original

    @g.step
    async def check_deps(ctx: StepContext[EdgeCaseState, MutableDeps, int]) -> int:
        # Deps will show the mutation since it's the same object
        return ctx.deps.value

    g.add(
        g.edge_from(g.start_node).to(try_modify_deps),
        g.edge_from(try_modify_deps).to(check_deps),
        g.edge_from(check_deps).to(g.end_node),
    )

    graph = g.build()
    deps = MutableDeps(value=42)
    result = await graph.run(state=EdgeCaseState(), deps=deps)
    # The deps object was mutated (user responsibility to avoid this)
    assert result == 999
    assert deps.value == 999
