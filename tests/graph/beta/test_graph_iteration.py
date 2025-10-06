"""Tests for iterative graph execution and inspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from pydantic_graph.beta import GraphBuilder, StepContext
from pydantic_graph.beta.graph import EndMarker, GraphTask, JoinItem
from pydantic_graph.beta.id_types import NodeID

pytestmark = pytest.mark.anyio


@dataclass
class IterState:
    counter: int = 0


async def test_iter_basic():
    """Test basic iteration over graph execution."""
    g = GraphBuilder(state_type=IterState, output_type=int)

    @g.step
    async def increment(ctx: StepContext[IterState, None, None]) -> int:
        ctx.state.counter += 1
        return ctx.state.counter

    @g.step
    async def double(ctx: StepContext[IterState, None, int]) -> int:
        return ctx.inputs * 2

    g.add(
        g.edge_from(g.start_node).to(increment),
        g.edge_from(increment).to(double),
        g.edge_from(double).to(g.end_node),
    )

    graph = g.build()
    state = IterState()

    events: list[Any] = []
    async with graph.iter(state=state) as run:
        async for event in run:
            events.append(event)

    assert len(events) > 0
    last_event = events[-1]
    assert isinstance(last_event, EndMarker)
    assert last_event.value == 2  # pyright: ignore[reportUnknownMemberType]


async def test_iter_with_next():
    """Test manual iteration using next() method."""
    g = GraphBuilder(state_type=IterState, output_type=int)

    @g.step
    async def step_one(ctx: StepContext[IterState, None, None]) -> int:
        return 10

    @g.step
    async def step_two(ctx: StepContext[IterState, None, int]) -> int:
        return ctx.inputs + 5

    g.add(
        g.edge_from(g.start_node).to(step_one),
        g.edge_from(step_one).to(step_two),
        g.edge_from(step_two).to(g.end_node),
    )

    graph = g.build()
    state = IterState()

    async with graph.iter(state=state) as run:
        # Manually advance through each step
        event1 = await run.next()
        assert isinstance(event1, list)

        event2 = await run.next()
        assert isinstance(event2, list)

        event3 = await run.next()
        assert isinstance(event3, EndMarker)
        assert event3.value == 15


async def test_iter_inspect_tasks():
    """Test inspecting GraphTask objects during iteration."""
    g = GraphBuilder(state_type=IterState, output_type=int)

    @g.step
    async def my_step(ctx: StepContext[IterState, None, None]) -> int:
        return 42

    g.add(
        g.edge_from(g.start_node).to(my_step),
        g.edge_from(my_step).to(g.end_node),
    )

    graph = g.build()
    state = IterState()

    task_nodes: list[NodeID] = []
    async with graph.iter(state=state) as run:
        async for event in run:
            if isinstance(event, list):
                for task in event:
                    assert isinstance(task, GraphTask)
                    task_nodes.append(task.node_id)

    assert 'my_step' in [str(n) for n in task_nodes]


async def test_iter_with_broadcast():
    """Test iteration with parallel broadcast operations."""
    g = GraphBuilder(state_type=IterState, output_type=list[int])

    @g.step
    async def source(ctx: StepContext[IterState, None, None]) -> int:
        return 5

    @g.step
    async def add_one(ctx: StepContext[IterState, None, int]) -> int:
        return ctx.inputs + 1

    @g.step
    async def add_two(ctx: StepContext[IterState, None, int]) -> int:
        return ctx.inputs + 2

    from pydantic_graph.beta import ListAppendReducer

    collect = g.join(ListAppendReducer[int])

    g.add(
        g.edge_from(g.start_node).to(source),
        g.edge_from(source).to(add_one, add_two),
        g.edge_from(add_one, add_two).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    state = IterState()

    join_items_seen = 0
    async with graph.iter(state=state) as run:
        async for event in run:
            if isinstance(event, JoinItem):
                join_items_seen += 1

    # Should see 2 join items (one from each parallel path)
    assert join_items_seen == 2


async def test_iter_output_property():
    """Test accessing the output property during and after iteration."""
    g = GraphBuilder(state_type=IterState, output_type=int)

    @g.step
    async def compute(ctx: StepContext[IterState, None, None]) -> int:
        return 100

    g.add(
        g.edge_from(g.start_node).to(compute),
        g.edge_from(compute).to(g.end_node),
    )

    graph = g.build()
    state = IterState()

    async with graph.iter(state=state) as run:
        # Output should be None before completion
        assert run.output is None

        async for event in run:
            if isinstance(event, EndMarker):
                # Output should be available once we have an EndMarker
                # (though we're still in the loop)
                pass

        # After iteration completes, output should be available
        assert run.output == 100


async def test_iter_next_task_property():
    """Test accessing the next_task property."""
    g = GraphBuilder(state_type=IterState, output_type=int)

    @g.step
    async def my_step(ctx: StepContext[IterState, None, None]) -> int:
        return 42

    g.add(
        g.edge_from(g.start_node).to(my_step),
        g.edge_from(my_step).to(g.end_node),
    )

    graph = g.build()
    state = IterState()

    async with graph.iter(state=state) as run:
        # Before starting, next_task should be the initial task
        initial_task = run.next_task
        assert isinstance(initial_task, list)

        # Advance one step
        await run.next()

        # next_task should update
        next_task = run.next_task
        assert next_task is not None


async def test_iter_with_map():
    """Test iteration with map operations."""
    g = GraphBuilder(state_type=IterState, output_type=list[int])

    @g.step
    async def generate(ctx: StepContext[IterState, None, None]) -> list[int]:
        return [1, 2, 3]

    @g.step
    async def square(ctx: StepContext[IterState, None, int]) -> int:
        return ctx.inputs * ctx.inputs

    from pydantic_graph.beta import ListAppendReducer

    collect = g.join(ListAppendReducer[int])

    g.add(
        g.edge_from(g.start_node).to(generate),
        g.edge_from(generate).map().to(square),
        g.edge_from(square).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    state = IterState()

    task_count = 0
    async with graph.iter(state=state) as run:
        async for event in run:
            if isinstance(event, list):
                task_count += len(event)

    # Should see multiple tasks from the map
    assert task_count >= 3


async def test_iter_early_termination():
    """Test that iteration can be terminated early."""
    g = GraphBuilder(state_type=IterState, output_type=int)

    @g.step
    async def step_one(ctx: StepContext[IterState, None, None]) -> int:
        ctx.state.counter += 1
        return 10

    @g.step
    async def step_two(ctx: StepContext[IterState, None, int]) -> int:
        ctx.state.counter += 1
        return ctx.inputs + 5

    @g.step
    async def step_three(ctx: StepContext[IterState, None, int]) -> int:
        ctx.state.counter += 1
        return ctx.inputs * 2

    g.add(
        g.edge_from(g.start_node).to(step_one),
        g.edge_from(step_one).to(step_two),
        g.edge_from(step_two).to(step_three),
        g.edge_from(step_three).to(g.end_node),
    )

    graph = g.build()
    state = IterState()

    async with graph.iter(state=state) as run:
        event_count = 0
        async for _ in run:
            event_count += 1
            if event_count >= 2:
                break  # Early termination

    # State changes should have happened only for completed steps
    # The exact counter value depends on how many steps completed before break
    assert state.counter < 3  # Not all steps completed


async def test_iter_state_inspection():
    """Test inspecting state changes during iteration."""
    g = GraphBuilder(state_type=IterState, output_type=int)

    @g.step
    async def increment(ctx: StepContext[IterState, None, None]) -> None:
        ctx.state.counter += 1

    @g.step
    async def double_counter(ctx: StepContext[IterState, None, None]) -> int:
        ctx.state.counter *= 2
        return ctx.state.counter

    g.add(
        g.edge_from(g.start_node).to(increment),
        g.edge_from(increment).to(double_counter),
        g.edge_from(double_counter).to(g.end_node),
    )

    graph = g.build()
    state = IterState()

    state_snapshots: list[Any] = []
    async with graph.iter(state=state) as run:
        async for _ in run:
            # Take a snapshot of the state after each event
            state_snapshots.append(state.counter)

    # State should have evolved during execution
    assert state_snapshots[-1] == 2  # (0 + 1) * 2
