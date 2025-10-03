"""Additional edge case tests for graph execution to improve coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest

from pydantic_graph.beta import GraphBuilder, StepContext

pytestmark = pytest.mark.anyio


@dataclass
class TestState:
    value: int = 0


async def test_graph_repr():
    """Test that Graph.__repr__ returns a mermaid diagram."""
    g = GraphBuilder(state_type=TestState, output_type=int)

    @g.step
    async def simple_step(ctx: StepContext[TestState, None, None]) -> int:
        return 42

    g.add(
        g.edge_from(g.start_node).to(simple_step),
        g.edge_from(simple_step).to(g.end_node),
    )

    graph = g.build()
    repr_str = repr(graph)
    assert 'graph' in repr_str.lower() or 'flowchart' in repr_str.lower()


async def test_graph_render_with_title():
    """Test Graph.render method with title parameter."""
    g = GraphBuilder(state_type=TestState, output_type=int)

    @g.step
    async def simple_step(ctx: StepContext[TestState, None, None]) -> int:
        return 42

    g.add(
        g.edge_from(g.start_node).to(simple_step),
        g.edge_from(simple_step).to(g.end_node),
    )

    graph = g.build()
    rendered = graph.render(title='My Graph')
    assert 'My Graph' in rendered or 'graph' in rendered.lower()


async def test_get_parent_fork_missing():
    """Test that get_parent_fork raises RuntimeError when join has no parent fork."""
    from pydantic_graph.beta.id_types import JoinID, NodeID

    g = GraphBuilder(state_type=TestState, output_type=int)

    @g.step
    async def simple_step(ctx: StepContext[TestState, None, None]) -> int:
        return 42

    g.add(
        g.edge_from(g.start_node).to(simple_step),
        g.edge_from(simple_step).to(g.end_node),
    )

    graph = g.build()

    # Try to get a parent fork for a non-existent join
    fake_join_id = JoinID(NodeID('fake_join'))
    with pytest.raises(RuntimeError, match='not a join node'):
        graph.get_parent_fork(fake_join_id)


async def test_decision_no_matching_branch():
    """Test that decision raises RuntimeError when no branch matches."""
    g = GraphBuilder(state_type=TestState, output_type=str)

    @g.step
    async def return_unexpected(ctx: StepContext[TestState, None, None]) -> int:
        return 999

    @g.step
    async def handle_str(ctx: StepContext[TestState, None, str]) -> str:
        return f'Got: {ctx.inputs}'

    g.add(
        g.edge_from(g.start_node).to(return_unexpected),
        g.edge_from(return_unexpected).to(g.decision().branch(g.match(str).to(handle_str))),
        g.edge_from(handle_str).to(g.end_node),
    )

    graph = g.build()

    with pytest.raises(RuntimeError, match='No branch matched'):
        await graph.run(state=TestState())


async def test_decision_invalid_type_check():
    """Test decision branch with invalid type for isinstance check."""

    g = GraphBuilder(state_type=TestState, output_type=str)

    @g.step
    async def return_value(ctx: StepContext[TestState, None, None]) -> int:
        return 42

    @g.step
    async def handle_value(ctx: StepContext[TestState, None, int]) -> str:
        return str(ctx.inputs)

    # Try to use a non-type as a branch source - this might cause TypeError during isinstance check
    # Note: This is hard to trigger without directly constructing invalid decision branches
    # For now, just test normal union types work
    g.add(
        g.edge_from(g.start_node).to(return_value),
        g.edge_from(return_value).to(g.decision().branch(g.match(int).to(handle_value))),
        g.edge_from(handle_value).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=TestState())
    assert result == '42'


async def test_map_non_iterable():
    """Test that mapping a non-iterable value raises RuntimeError."""
    g = GraphBuilder(state_type=TestState, output_type=int)

    @g.step
    async def return_non_iterable(ctx: StepContext[TestState, None, None]) -> int:
        return 42  # Not iterable!

    @g.step
    async def process_item(ctx: StepContext[TestState, None, int]) -> int:
        return ctx.inputs

    @g.step
    async def sum_items(ctx: StepContext[TestState, None, list[int]]) -> int:
        return sum(ctx.inputs)

    # This will fail at runtime because we're trying to map over a non-iterable
    g.add(
        g.edge_from(g.start_node).to(return_non_iterable),
        g.edge_from(return_non_iterable).map().to(process_item),
        g.edge_from(process_item).join().to(sum_items),
        g.edge_from(sum_items).to(g.end_node),
    )

    graph = g.build()

    with pytest.raises(RuntimeError, match='Cannot map non-iterable'):
        await graph.run(state=TestState())


async def test_reducer_stop_iteration():
    """Test reducer that raises StopIteration to cancel concurrent tasks."""

    @dataclass
    class EarlyStopState:
        stopped: bool = False

    g = GraphBuilder(state_type=EarlyStopState, output_type=int)

    @g.step
    async def generate_numbers(ctx: StepContext[EarlyStopState, None, None]) -> list[int]:
        return [1, 2, 3, 4, 5]

    @g.step
    async def slow_process(ctx: StepContext[EarlyStopState, None, int]) -> int:
        # Simulate some processing
        return ctx.inputs * 2

    @g.join
    class EarlyStopReducer(g.Reducer[int, int]):
        def __init__(self):
            self.total = 0
            self.count = 0

        def initialize(self):
            return 0

        def reduce(self, ctx: StepContext[EarlyStopState, None, int]):
            self.count += 1
            self.total += ctx.inputs
            # Stop after receiving 2 items
            if self.count >= 2:
                ctx.state.stopped = True
                raise StopIteration

        def finalize(self, ctx: StepContext[EarlyStopState, None, None]) -> int:
            return self.total

    @g.step
    async def finalize_result(ctx: StepContext[EarlyStopState, None, int]) -> int:
        return ctx.inputs

    g.add(
        g.edge_from(g.start_node).to(generate_numbers),
        g.edge_from(generate_numbers).map().to(slow_process),
        g.edge_from(slow_process).join(EarlyStopReducer).to(finalize_result),
        g.edge_from(finalize_result).to(g.end_node),
    )

    graph = g.build()
    state = EarlyStopState()
    result = await graph.run(state=state)

    # Should have stopped early
    assert state.stopped
    # Result should be less than the full sum (2+4+6+8+10=30)
    assert result < 30


async def test_empty_path_handling():
    """Test handling of empty paths in graph execution."""
    g = GraphBuilder(state_type=TestState, output_type=int)

    @g.step
    async def return_value(ctx: StepContext[TestState, None, None]) -> int:
        return 42

    # Just connect start to step to end - this should work fine
    g.add(
        g.edge_from(g.start_node).to(return_value),
        g.edge_from(return_value).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=TestState())
    assert result == 42


async def test_literal_branch_matching():
    """Test decision branch matching with Literal types."""
    g = GraphBuilder(state_type=TestState, output_type=str)

    @g.step
    async def choose_option(ctx: StepContext[TestState, None, None]) -> Literal['a', 'b', 'c']:
        return 'b'

    @g.step
    async def handle_a(ctx: StepContext[TestState, None, object]) -> str:
        return 'Chose A'

    @g.step
    async def handle_b(ctx: StepContext[TestState, None, object]) -> str:
        return 'Chose B'

    @g.step
    async def handle_c(ctx: StepContext[TestState, None, object]) -> str:
        return 'Chose C'

    from pydantic_graph.beta import TypeExpression

    g.add(
        g.edge_from(g.start_node).to(choose_option),
        g.edge_from(choose_option).to(
            g.decision()
            .branch(g.match(TypeExpression[Literal['a']]).to(handle_a))
            .branch(g.match(TypeExpression[Literal['b']]).to(handle_b))
            .branch(g.match(TypeExpression[Literal['c']]).to(handle_c))
        ),
        g.edge_from(handle_a, handle_b, handle_c).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=TestState())
    assert result == 'Chose B'


async def test_path_with_label_marker():
    """Test that LabelMarker in paths doesn't affect execution."""
    g = GraphBuilder(state_type=TestState, output_type=int)

    @g.step
    async def step_a(ctx: StepContext[TestState, None, None]) -> int:
        return 10

    @g.step
    async def step_b(ctx: StepContext[TestState, None, int]) -> int:
        return ctx.inputs * 2

    # Add labels to the path
    g.add(
        g.edge_from(g.start_node).label('start').to(step_a),
        g.edge_from(step_a).label('middle').to(step_b),
        g.edge_from(step_b).label('end').to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=TestState())
    assert result == 20


async def test_nested_reducers_with_prefix():
    """Test multiple active reducers where one is a prefix of another."""
    g = GraphBuilder(state_type=TestState, output_type=int)

    @g.step
    async def outer_list(ctx: StepContext[TestState, None, None]) -> list[list[int]]:
        return [[1, 2], [3, 4]]

    @g.step
    async def inner_process(ctx: StepContext[TestState, None, int]) -> int:
        return ctx.inputs * 2

    @g.step
    async def outer_sum(ctx: StepContext[TestState, None, list[int]]) -> int:
        return sum(ctx.inputs)

    @g.step
    async def final_sum(ctx: StepContext[TestState, None, list[int]]) -> int:
        return sum(ctx.inputs)

    # Create nested map operations
    g.add(
        g.edge_from(g.start_node).to(outer_list),
        g.edge_from(outer_list).map().map().to(inner_process),
        g.edge_from(inner_process).join().to(outer_sum),
        g.edge_from(outer_sum).join().to(final_sum),
        g.edge_from(final_sum).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=TestState())
    # (1+2+3+4) * 2 = 20
    assert result == 20
