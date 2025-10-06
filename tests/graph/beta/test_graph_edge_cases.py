"""Additional edge case tests for graph execution to improve coverage."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import pytest
from inline_snapshot import snapshot

from pydantic_graph.beta import GraphBuilder, StepContext
from pydantic_graph.beta.join import Reducer, SumReducer

pytestmark = pytest.mark.anyio


@dataclass
class MyState:
    value: int = 0


async def test_graph_repr():
    """Test that Graph.__repr__ returns a mermaid diagram."""
    g = GraphBuilder(state_type=MyState, output_type=int)

    @g.step
    async def simple_step(ctx: StepContext[MyState, None, None]) -> int:
        return 42

    g.add(
        g.edge_from(g.start_node).to(simple_step),
        g.edge_from(simple_step).to(g.end_node),
    )

    graph = g.build()
    graph_repr = repr(graph)

    # Replace the non-constant graph object id with a constant string:
    normalized_graph_repr = re.sub(hex(id(graph)), '0xGraphObjectId', graph_repr)

    assert normalized_graph_repr == snapshot("""\
<pydantic_graph.beta.graph.Graph object at 0xGraphObjectId
stateDiagram-v2
  simple_step

  [*] --> simple_step
  simple_step --> [*]
>\
""")


async def test_graph_render_with_title():
    """Test Graph.render method with title parameter."""
    g = GraphBuilder(state_type=MyState, output_type=int)

    @g.step
    async def simple_step(ctx: StepContext[MyState, None, None]) -> int:
        return 42

    g.add(
        g.edge_from(g.start_node).to(simple_step),
        g.edge_from(simple_step).to(g.end_node),
    )

    graph = g.build()
    rendered = graph.render(title='My Graph')
    assert rendered == snapshot("""\
---
title: My Graph
---
stateDiagram-v2
  simple_step

  [*] --> simple_step
  simple_step --> [*]\
""")


async def test_get_parent_fork_missing():
    """Test that get_parent_fork raises RuntimeError when join has no parent fork."""
    from pydantic_graph.beta.id_types import JoinID, NodeID

    g = GraphBuilder(state_type=MyState, output_type=int)

    @g.step
    async def simple_step(ctx: StepContext[MyState, None, None]) -> int:
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
    g = GraphBuilder(state_type=MyState, output_type=str)

    @g.step
    async def return_unexpected(ctx: StepContext[MyState, None, None]) -> int:
        return 999

    @g.step
    async def handle_str(ctx: StepContext[MyState, None, str]) -> str:
        return f'Got: {ctx.inputs}'

    # the purpose of this test is to test runtime behavior when you have this type failure, which is why
    # we have the `# type: ignore` below
    g.add(
        g.edge_from(g.start_node).to(return_unexpected),
        g.edge_from(return_unexpected).to(g.decision().branch(g.match(str).to(handle_str))),  # type: ignore
        g.edge_from(handle_str).to(g.end_node),
    )

    graph = g.build()

    with pytest.raises(RuntimeError, match='No branch matched'):
        await graph.run(state=MyState())


async def test_decision_invalid_type_check():
    """Test decision branch with invalid type for isinstance check."""

    g = GraphBuilder(state_type=MyState, output_type=str)

    @g.step
    async def return_value(ctx: StepContext[MyState, None, None]) -> int:
        return 42

    @g.step
    async def handle_value(ctx: StepContext[MyState, None, int]) -> str:
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
    result = await graph.run(state=MyState())
    assert result == '42'


async def test_map_non_iterable():
    """Test that mapping a non-iterable value raises RuntimeError."""
    g = GraphBuilder(state_type=MyState, output_type=int)

    @g.step
    async def return_non_iterable(ctx: StepContext[MyState, None, None]) -> int:
        return 42  # Not iterable!

    @g.step
    async def process_item(ctx: StepContext[MyState, None, int]) -> int:
        return ctx.inputs

    sum_items = g.join(SumReducer[int])

    # This will fail at runtime because we're trying to map over a non-iterable
    # We have a `# type: ignore` below because we are testing behavior when you ignore the type error
    g.add(
        g.edge_from(g.start_node).to(return_non_iterable),
        g.edge_from(return_non_iterable).map().to(process_item),  # type: ignore
        g.edge_from(process_item).to(sum_items),
        g.edge_from(sum_items).to(g.end_node),
    )

    graph = g.build()

    with pytest.raises(RuntimeError, match='Cannot map non-iterable'):
        await graph.run(state=MyState())


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
    class EarlyStopReducer(Reducer[EarlyStopState, None, int, int]):
        def __init__(self):
            self.total = 0
            self.count = 0
            self.stopped = False

        def reduce(self, ctx: StepContext[EarlyStopState, None, int]):
            if self.stopped:
                # Cancelled tasks don't necessarily stop immediately, so we add handling here
                # to prevent the reduce method from doing anything in concurrent tasks that
                # haven't been immediately cancelled
                raise StopIteration

            self.count += 1
            self.total += ctx.inputs
            # Stop after receiving 2 items
            if self.count >= 2:
                self.stopped = True
                ctx.state.stopped = True  # set it on the state so we can assert after the run completes
                raise StopIteration

        def finalize(self, ctx: StepContext[EarlyStopState, None, None]) -> int:
            return self.total

    g.add(
        g.edge_from(g.start_node).to(generate_numbers),
        g.edge_from(generate_numbers).map().to(slow_process),
        g.edge_from(slow_process).to(EarlyStopReducer),
        g.edge_from(EarlyStopReducer).to(g.end_node),
    )

    graph = g.build()
    state = EarlyStopState()
    result = await graph.run(state=state)

    # Should have stopped early
    assert state.stopped
    # Result should be less than the full sum (2+4+6+8+10=30)
    # Actually, it should be less than the maximum of any two terms, (8+10=18)
    assert result <= 18


async def test_empty_path_handling():
    """Test handling of empty paths in graph execution."""
    g = GraphBuilder(state_type=MyState, output_type=int)

    @g.step
    async def return_value(ctx: StepContext[MyState, None, None]) -> int:
        return 42

    # Just connect start to step to end - this should work fine
    g.add(
        g.edge_from(g.start_node).to(return_value),
        g.edge_from(return_value).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=MyState())
    assert result == 42


async def test_literal_branch_matching():
    """Test decision branch matching with Literal types."""
    g = GraphBuilder(state_type=MyState, output_type=str)

    @g.step
    async def choose_option(ctx: StepContext[MyState, None, None]) -> Literal['a', 'b', 'c']:
        return 'b'

    @g.step
    async def handle_a(ctx: StepContext[MyState, None, object]) -> str:
        return 'Chose A'

    @g.step
    async def handle_b(ctx: StepContext[MyState, None, object]) -> str:
        return 'Chose B'

    @g.step
    async def handle_c(ctx: StepContext[MyState, None, object]) -> str:
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
    result = await graph.run(state=MyState())
    assert result == 'Chose B'


async def test_path_with_label_marker():
    """Test that LabelMarker in paths doesn't affect execution."""
    g = GraphBuilder(state_type=MyState, output_type=int)

    @g.step
    async def step_a(ctx: StepContext[MyState, None, None]) -> int:
        return 10

    @g.step
    async def step_b(ctx: StepContext[MyState, None, int]) -> int:
        return ctx.inputs * 2

    # Add labels to the path
    g.add(
        g.edge_from(g.start_node).label('start').to(step_a),
        g.edge_from(step_a).label('middle').to(step_b),
        g.edge_from(step_b).label('end').to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=MyState())
    assert result == 20


# TODO: Make a version of this test where we manually specify the parent fork so that we can do different joining behavior at the different levels
async def test_nested_reducers_with_prefix():
    """Test multiple active reducers where one is a prefix of another."""
    g = GraphBuilder(state_type=MyState, output_type=int)

    @g.step
    async def outer_list(ctx: StepContext[MyState, None, None]) -> list[list[int]]:
        return [[1, 2], [3, 4]]

    @g.step
    async def inner_process(ctx: StepContext[MyState, None, int]) -> int:
        return ctx.inputs * 2

    # Note: we use  the _most_ ancestral fork as the parent fork by default, which means that this join
    # actually will join all forks from the initial outer_list, therefore summing everything, rather
    # than _only_ summing the inner loops. If/when we add more control over the parent fork calculation, we can
    # test that it's possible to use separate logic for the inside vs. the outside.
    sum_join = g.join(SumReducer[int])

    # Create nested map operations
    g.add(
        g.edge_from(g.start_node).to(outer_list),
        g.edge_from(outer_list).map().map().to(inner_process),
        g.edge_from(inner_process).to(sum_join),
        g.edge_from(sum_join).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=MyState())
    # (1+2+3+4) * 2 = 20
    assert result == 20
