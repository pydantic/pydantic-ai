"""Additional edge case tests for graph execution to improve coverage."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import pytest
from inline_snapshot import snapshot

from pydantic_graph.beta import GraphBuilder, StepContext
from pydantic_graph.beta.join import ReduceFirstValue, ReducerContext, reduce_sum

pytestmark = pytest.mark.anyio


@dataclass
class MyState:
    value: int = 0


async def test_graph_repr():
    """Test that Graph.__repr__ returns a mermaid diagram."""
    g = GraphBuilder(state_type=MyState, output_type=int)

    @g.step
    async def simple_step(ctx: StepContext[MyState, None, None]) -> int:
        return 42  # pragma: no cover

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
        return 42  # pragma: no cover

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
        return 42  # pragma: no cover

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
        return f'Got: {ctx.inputs}'  # pragma: no cover

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
        return ctx.inputs  # pragma: no cover

    sum_items = g.join(reduce_sum, initial=0)

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

    def get_early_stopping_reducer():
        count = 0

        def reduce(ctx: ReducerContext[EarlyStopState, object], current: int, inputs: int) -> int:
            nonlocal count
            count += 1
            current += inputs
            if count >= 2:
                ctx.state.stopped = True  # update the state so we can assert on it later
                ctx.cancel_sibling_tasks()
            return current

        return reduce

    stop_early = g.join(get_early_stopping_reducer(), initial=0)

    g.add(
        g.edge_from(g.start_node).to(generate_numbers),
        g.edge_from(generate_numbers).map().to(slow_process),
        g.edge_from(slow_process).to(stop_early),
        g.edge_from(stop_early).to(g.end_node),
    )

    graph = g.build()
    state = EarlyStopState()
    result = await graph.run(state=state)

    # Should have stopped early
    assert state.stopped
    # Result should be less than the full sum (2+4+6+8+10=30)
    # Actually, it should be less than the maximum of any two terms, (8+10=18)
    assert result <= 18


async def test_parallel_reducer_stop_iteration_explicit_fork_ids():
    """Test reducer that raises StopIteration to cancel concurrent tasks."""

    g = GraphBuilder(output_type=int)

    @g.step
    async def generate_numbers(ctx: StepContext[None, None, None]) -> list[int]:
        return [1, 1, 1, 1, 1]

    stop_early_1 = g.join(ReduceFirstValue[int](), initial=0, parent_fork_id='map_1')
    stop_early_2 = g.join(ReduceFirstValue[int](), initial=0, parent_fork_id='map_2')
    collect = g.join(reduce_sum, initial=0)

    g.add(
        g.edge_from(g.start_node).to(generate_numbers),
        g.edge_from(generate_numbers).broadcast(
            lambda b: [
                b.map(fork_id='map_1').transform(lambda ctx: ctx.inputs * 10).to(stop_early_1),
                b.map(fork_id='map_2').to(stop_early_2),
            ]
        ),
        g.edge_from(stop_early_1, stop_early_2).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run()

    # Result should be 11 because it should have stopped early on one input in the first fork and the *10 fork
    assert result == 11


async def test_parallel_reducer_stop_iteration_implicit_fork_ids():
    """Test reducer that raises StopIteration to cancel concurrent tasks."""

    g = GraphBuilder(output_type=int)

    @g.step
    async def generate_numbers(ctx: StepContext[None, None, None]) -> list[int]:
        return [1, 1, 1, 1, 1]

    stop_early_1 = g.join(ReduceFirstValue[int](), initial=0, preferred_parent_fork='closest')
    stop_early_2 = g.join(ReduceFirstValue[int](), initial=0, preferred_parent_fork='closest')
    collect = g.join(reduce_sum, initial=0)

    g.add(
        g.edge_from(g.start_node).to(generate_numbers),
        g.edge_from(generate_numbers).broadcast(
            lambda b: [
                b.map().transform(lambda ctx: ctx.inputs * 10).to(stop_early_1),
                b.map().to(stop_early_2),
            ]
        ),
        g.edge_from(stop_early_1, stop_early_2).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run()

    # Result should be 11 because it should have stopped early on one input in the first fork and the *10 fork
    assert result == 11


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
        return 'Chose A'  # pragma: no cover

    @g.step
    async def handle_b(ctx: StepContext[MyState, None, object]) -> str:
        return 'Chose B'

    @g.step
    async def handle_c(ctx: StepContext[MyState, None, object]) -> str:
        return 'Chose C'  # pragma: no cover

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
    sum_join = g.join(reduce_sum, initial=0)

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
    assert str(graph) == snapshot("""\
stateDiagram-v2
  outer_list
  state map <<fork>>
  state map_2 <<fork>>
  inner_process
  state reduce_sum <<join>>

  [*] --> outer_list
  outer_list --> map
  map --> map_2
  map_2 --> inner_process
  inner_process --> reduce_sum
  reduce_sum --> [*]\
""")
