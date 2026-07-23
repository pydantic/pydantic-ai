"""Tests for join nodes and reducer types."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from pydantic_graph import GraphBuilder, StepContext
from pydantic_graph.join import (
    ReduceFirstValue,
    ReducerContext,
    reduce_dict_update,
    reduce_list_append,
    reduce_null,
)

pytestmark = pytest.mark.anyio


@dataclass
class SimpleState:
    value: int = 0


async def test_null_reducer():
    """Test NullReducer that discards all inputs."""
    g = GraphBuilder(state_type=SimpleState)

    @g.step
    async def source(ctx: StepContext[SimpleState, None, None]) -> list[int]:
        return [1, 2, 3]

    @g.step
    async def process(ctx: StepContext[SimpleState, None, int]) -> int:
        ctx.state.value += ctx.inputs
        return ctx.inputs

    null_join = g.join(reduce_null, initial=None)

    g.add(
        g.edge_from(g.start_node).to(source),
        g.edge_from(source).map().to(process),
        g.edge_from(process).to(null_join),
        g.edge_from(null_join).to(g.end_node),
    )

    graph = g.build()
    state = SimpleState()
    result = await graph.run(state=state)
    assert result is None
    # But side effects should still happen
    assert state.value == 6


async def test_reduce_list_append():
    """Test reduce_list_append that collects all inputs into a list."""
    g = GraphBuilder(state_type=SimpleState, output_type=list[str])

    @g.step
    async def generate_numbers(ctx: StepContext[SimpleState, None, None]) -> list[int]:
        return [1, 2, 3, 4]

    @g.step
    async def to_string(ctx: StepContext[SimpleState, None, int]) -> str:
        return f'item-{ctx.inputs}'

    list_join = g.join(reduce_list_append, initial_factory=list[str])

    g.add(
        g.edge_from(g.start_node).to(generate_numbers),
        g.edge_from(generate_numbers).map().to(to_string),
        g.edge_from(to_string).to(list_join),
        g.edge_from(list_join).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=SimpleState())
    # Order may vary due to parallel execution
    assert sorted(result) == ['item-1', 'item-2', 'item-3', 'item-4']


async def test_reduce_dict_update():
    """Test reduce_dict_update that merges dictionaries."""
    g = GraphBuilder(state_type=SimpleState, output_type=dict[str, int])

    @g.step
    async def generate_keys(ctx: StepContext[SimpleState, None, None]) -> list[str]:
        return ['a', 'b', 'c']

    @g.step
    async def create_dict(ctx: StepContext[SimpleState, None, str]) -> dict[str, int]:
        return {ctx.inputs: len(ctx.inputs)}

    dict_join = g.join(reduce_dict_update, initial_factory=dict[str, int])

    g.add(
        g.edge_from(g.start_node).to(generate_keys),
        g.edge_from(generate_keys).map().to(create_dict),
        g.edge_from(create_dict).to(dict_join),
        g.edge_from(dict_join).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=SimpleState())
    assert result == {'a': 1, 'b': 1, 'c': 1}


async def test_reducer_with_state_access():
    """Test that reducers can access and modify graph state."""

    # Note: doing this means the `count` will be shared across _all_ fork runs.
    def get_state_aware_reducer():
        count = 0

        def reduce_state_aware(ctx: ReducerContext[SimpleState, None], current: int, inputs: int) -> int:
            nonlocal count
            ctx.state.value += inputs
            count += 1
            return count

        return reduce_state_aware

    g = GraphBuilder(state_type=SimpleState, output_type=int)

    @g.step
    async def generate(ctx: StepContext[SimpleState, None, None]) -> list[int]:
        return [1, 2, 3]

    @g.step
    async def process(ctx: StepContext[SimpleState, None, int]) -> int:
        return ctx.inputs * 10

    aware_join = g.join(get_state_aware_reducer(), initial=0)

    g.add(
        g.edge_from(g.start_node).to(generate),
        g.edge_from(generate).map().to(process),
        g.edge_from(process).to(aware_join),
        g.edge_from(aware_join).to(g.end_node),
    )

    graph = g.build()
    state = SimpleState()
    result = await graph.run(state=state)
    assert result == 3  # Three items were reduced
    assert state.value == 60  # 10 + 20 + 30


async def test_join_with_custom_id():
    """Test creating a join with a custom node ID."""
    g = GraphBuilder(state_type=SimpleState, output_type=list[int])

    @g.step
    async def source(ctx: StepContext[SimpleState, None, None]) -> list[int]:
        return [1, 2, 3]  # pragma: no cover

    @g.step
    async def process(ctx: StepContext[SimpleState, None, int]) -> int:
        return ctx.inputs  # pragma: no cover

    custom_join = g.join(reduce_list_append, initial_factory=list[int], node_id='my_custom_join')

    g.add(
        g.edge_from(g.start_node).to(source),
        g.edge_from(source).map().to(process),
        g.edge_from(process).to(custom_join),
        g.edge_from(custom_join).to(g.end_node),
    )

    graph = g.build()
    assert 'my_custom_join' in graph.nodes


async def test_multiple_joins():
    """Test a graph with multiple independent joins."""

    @dataclass
    class MultiState:
        results: dict[str, list[int]] = field(default_factory=dict[str, list[int]])

    g = GraphBuilder(state_type=MultiState, output_type=dict[str, list[int]])

    @g.step
    async def source_a(ctx: StepContext[MultiState, None, None]) -> list[int]:
        return [1, 2]

    @g.step
    async def source_b(ctx: StepContext[MultiState, None, None]) -> list[int]:
        return [10, 20]

    @g.step
    async def process_a(ctx: StepContext[MultiState, None, int]) -> int:
        return ctx.inputs * 2

    @g.step
    async def process_b(ctx: StepContext[MultiState, None, int]) -> int:
        return ctx.inputs * 3

    join_a = g.join(reduce_list_append, initial_factory=list[int], node_id='join_a')
    join_b = g.join(reduce_list_append, initial_factory=list[int], node_id='join_b')

    @g.step
    async def combine(ctx: StepContext[MultiState, None, None]) -> dict[str, list[int]]:
        return ctx.state.results

    @g.step
    async def store_a(ctx: StepContext[MultiState, None, list[int]]) -> None:
        ctx.state.results['a'] = ctx.inputs

    @g.step
    async def store_b(ctx: StepContext[MultiState, None, list[int]]) -> None:
        ctx.state.results['b'] = ctx.inputs

    g.add(
        g.edge_from(g.start_node).to(source_a, source_b),
        g.edge_from(source_a).map().to(process_a),
        g.edge_from(source_b).map().to(process_b),
        g.edge_from(process_a).to(join_a),
        g.edge_from(process_b).to(join_b),
        g.edge_from(join_a).to(store_a),
        g.edge_from(join_b).to(store_b),
        g.edge_from(store_a, store_b).to(combine),
        g.edge_from(combine).to(g.end_node),
    )

    graph = g.build()
    state = MultiState()
    result = await graph.run(state=state)
    assert sorted(result['a']) == [2, 4]
    assert sorted(result['b']) == [30, 60]


async def test_reduce_dict_update_with_overlapping_keys():
    """Test that reduce_dict_update properly handles overlapping keys (later values win)."""
    g = GraphBuilder(state_type=SimpleState, output_type=dict[str, int])

    @g.step
    async def generate(ctx: StepContext[SimpleState, None, None]) -> list[int]:
        return [1, 2, 3]

    @g.step
    async def create_dict(ctx: StepContext[SimpleState, None, int]) -> dict[str, int]:
        # All create the same key
        return {'key': ctx.inputs}

    dict_join = g.join(reduce_dict_update, initial_factory=dict[str, int])

    g.add(
        g.edge_from(g.start_node).to(generate),
        g.edge_from(generate).map().to(create_dict),
        g.edge_from(create_dict).to(dict_join),
        g.edge_from(dict_join).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=SimpleState())
    # One of the values should win (1, 2, or 3)
    assert 'key' in result
    assert result['key'] in [1, 2, 3]


async def test_reducer_with_deps_access():
    """Test that reducer context can access deps"""

    @dataclass
    class DepsWithMultiplier:
        multiplier: int

    def reducer_using_deps(ctx: ReducerContext[SimpleState, DepsWithMultiplier], current: int, inputs: int) -> int:
        # Access deps through the context
        return current + (inputs * ctx.deps.multiplier)

    g = GraphBuilder(state_type=SimpleState, deps_type=DepsWithMultiplier, output_type=int)

    @g.step
    async def generate(ctx: StepContext[SimpleState, DepsWithMultiplier, None]) -> list[int]:
        return [1, 2, 3]

    @g.step
    async def process(ctx: StepContext[SimpleState, DepsWithMultiplier, int]) -> int:
        return ctx.inputs

    deps_join = g.join(reducer_using_deps, initial=0)

    g.add(
        g.edge_from(g.start_node).to(generate),
        g.edge_from(generate).map().to(process),
        g.edge_from(process).to(deps_join),
        g.edge_from(deps_join).to(g.end_node),
    )

    graph = g.build()
    state = SimpleState()
    deps = DepsWithMultiplier(multiplier=10)
    result = await graph.run(state=state, deps=deps)
    # (0 + 1*10) + (2*10) + (3*10) = 60
    assert result == 60


async def test_reduce_list_extend():
    """Test reduce_list_extend that extends a list with iterables"""
    from pydantic_graph.join import reduce_list_extend

    g = GraphBuilder(state_type=SimpleState, output_type=list[int])

    @g.step
    async def generate_iterables(ctx: StepContext[SimpleState, None, None]) -> list[list[int]]:
        return [[1, 2], [3, 4], [5, 6]]

    @g.step
    async def pass_through(ctx: StepContext[SimpleState, None, list[int]]) -> list[int]:
        return ctx.inputs

    extend_join = g.join(reduce_list_extend, initial_factory=list[int])

    g.add(
        g.edge_from(g.start_node).to(generate_iterables),
        g.edge_from(generate_iterables).map().to(pass_through),
        g.edge_from(pass_through).to(extend_join),
        g.edge_from(extend_join).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=SimpleState())
    # All sublists are extended into one flat list
    assert sorted(result) == [1, 2, 3, 4, 5, 6]


async def test_reduce_first_value():
    """Test ReduceFirstValue cancels sibling tasks"""

    @dataclass
    class StateWithResults:
        results: list[str] = field(default_factory=list[str])

    g = GraphBuilder(state_type=StateWithResults, output_type=str)

    @g.step
    async def generate(ctx: StepContext[StateWithResults, None, None]) -> list[int]:
        return [1, 2, 3, 4, 5]

    @g.step
    async def slow_process(ctx: StepContext[StateWithResults, None, int]) -> str:
        # First task finishes quickly
        if ctx.inputs == 1:
            await asyncio.sleep(0.001)
        else:
            # Others take longer (should be cancelled)
            await asyncio.sleep(10)
        ctx.state.results.append(f'completed-{ctx.inputs}')
        return f'result-{ctx.inputs}'

    first_join = g.join(ReduceFirstValue[str](), initial='')

    g.add(
        g.edge_from(g.start_node).to(generate),
        g.edge_from(generate).map().to(slow_process),
        g.edge_from(slow_process).to(first_join),
        g.edge_from(first_join).to(g.end_node),
    )

    graph = g.build()
    state = StateWithResults()
    result = await graph.run(state=state)

    # Only the first value should be returned
    assert result.startswith('result-')
    # Due to cancellation, not all 5 tasks should complete
    # (though timing can be tricky, so we just verify we got a result)
    assert 'completed-1' in state.results


async def test_reduce_first_value_keeps_side_effect_of_cancelled_sibling():
    """`ReduceFirstValue` cancels still-suspended siblings, but cannot un-run a side effect one already ran.

    Cancellation is dispatched synchronously the instant the winner's value is reduced, so a
    sibling's fate hinges on *where it is suspended* at that instant:

    - a sibling that recorded its side effect and *then* suspended keeps the effect — the
      `CancelledError` lands at the later await, after the write has happened;
    - a sibling still suspended *before* its side-effect line is cancelled without ever recording.

    The three branches pin both halves plus the winner, so the guarantee is exact rather than
    incidental. `asyncio.Event` gates force a single deterministic winner and a fixed ordering
    with no reliance on sleeps or timers: the winner only reduces once the effect-then-suspend
    branch has written and parked, so both losing branches are provably suspended (one after its
    side effect, one before) when cancellation reaches them.
    """

    @dataclass
    class CountingState:
        completed: list[str] = field(default_factory=list[str])

    effect_recorded = asyncio.Event()  # set once the effect-then-suspend branch has written and is about to park
    never = asyncio.Event()  # never set: whatever awaits it stays suspended until it is cancelled

    g = GraphBuilder(state_type=CountingState, output_type=str)

    @g.step
    async def generate(ctx: StepContext[CountingState, None, None]) -> list[str]:
        return ['winner', 'effect-then-suspend', 'suspend-then-effect']

    @g.step
    async def process(ctx: StepContext[CountingState, None, str]) -> str:
        if ctx.inputs == 'winner':
            # Win only after the losing sibling has recorded its effect and parked, so cancellation
            # is dispatched while that sibling sits at its post-side-effect await.
            await effect_recorded.wait()
            ctx.state.completed.append('winner')
            return 'result-winner'
        if ctx.inputs == 'effect-then-suspend':
            ctx.state.completed.append('effect-then-suspend')  # side effect BEFORE the await
            effect_recorded.set()
            await never.wait()  # cancelled here, after the write -> the effect survives
            return 'unreachable'  # pragma: no cover
        # 'suspend-then-effect': suspended before its side-effect line -> cancelled without recording.
        await never.wait()
        ctx.state.completed.append('suspend-then-effect')  # pragma: no cover
        return 'unreachable'  # pragma: no cover

    first = g.join(ReduceFirstValue[str](), initial='', node_id='first')

    g.add(
        g.edge_from(g.start_node).to(generate),
        g.edge_from(generate).map().to(process),
        g.edge_from(process).to(first),
        g.edge_from(first).to(g.end_node),
    )

    state = CountingState()
    result = await g.build().run(state=state)

    assert result == 'result-winner'
    # The effect-then-suspend branch recorded before parking and kept its effect despite being
    # cancelled at the later await; the suspend-then-effect branch was cancelled before its
    # side-effect line and never recorded. The winner writes only after the event is set, so it
    # always trails the surviving side effect.
    assert state.completed == ['effect-then-suspend', 'winner']
