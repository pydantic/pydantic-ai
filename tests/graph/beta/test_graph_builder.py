"""Tests for the GraphBuilder API and basic graph construction."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pydantic_graph.beta import GraphBuilder, StepContext

pytestmark = pytest.mark.anyio


@dataclass
class SimpleState:
    counter: int = 0
    result: str | None = None


async def test_basic_graph_builder():
    """Test basic graph builder construction and execution."""
    g = GraphBuilder(state_type=SimpleState, output_type=int)

    @g.step
    async def increment(ctx: StepContext[SimpleState, None, None]) -> int:
        ctx.state.counter += 1
        return ctx.state.counter

    g.add(
        g.edge_from(g.start_node).to(increment),
        g.edge_from(increment).to(g.end_node),
    )

    graph = g.build()
    state = SimpleState()
    result = await graph.run(state=state)
    assert result == 1
    assert state.counter == 1


async def test_sequential_steps():
    """Test multiple sequential steps in a graph."""
    g = GraphBuilder(state_type=SimpleState, output_type=int)

    @g.step
    async def step_one(ctx: StepContext[SimpleState, None, None]) -> None:
        ctx.state.counter += 1

    @g.step
    async def step_two(ctx: StepContext[SimpleState, None, None]) -> None:
        ctx.state.counter *= 2

    @g.step
    async def step_three(ctx: StepContext[SimpleState, None, None]) -> int:
        ctx.state.counter += 10
        return ctx.state.counter

    g.add(
        g.edge_from(g.start_node).to(step_one),
        g.edge_from(step_one).to(step_two),
        g.edge_from(step_two).to(step_three),
        g.edge_from(step_three).to(g.end_node),
    )

    graph = g.build()
    state = SimpleState(counter=5)
    result = await graph.run(state=state)
    # (5 + 1) * 2 + 10 = 22
    assert result == 22


async def test_step_with_inputs():
    """Test steps that receive and transform input data."""
    g = GraphBuilder(state_type=SimpleState, input_type=int, output_type=str)

    @g.step
    async def double_it(ctx: StepContext[SimpleState, None, int]) -> int:
        return ctx.inputs * 2

    @g.step
    async def stringify(ctx: StepContext[SimpleState, None, int]) -> str:
        return f'Result: {ctx.inputs}'

    g.add(
        g.edge_from(g.start_node).to(double_it),
        g.edge_from(double_it).to(stringify),
        g.edge_from(stringify).to(g.end_node),
    )

    graph = g.build()
    state = SimpleState()
    result = await graph.run(state=state, inputs=21)
    assert result == 'Result: 42'


async def test_step_with_custom_id():
    """Test creating steps with custom IDs."""
    g = GraphBuilder(state_type=SimpleState, output_type=int)

    @g.step(node_id='custom_step_id')
    async def my_step(ctx: StepContext[SimpleState, None, None]) -> int:
        return 42

    g.add(
        g.edge_from(g.start_node).to(my_step),
        g.edge_from(my_step).to(g.end_node),
    )

    graph = g.build()
    assert 'custom_step_id' in graph.nodes


async def test_step_with_label():
    """Test creating steps with human-readable labels."""
    g = GraphBuilder(state_type=SimpleState, output_type=int)

    @g.step(label='My Custom Label')
    async def my_step(ctx: StepContext[SimpleState, None, None]) -> int:
        return 42

    assert my_step.label == 'My Custom Label'

    g.add(
        g.edge_from(g.start_node).to(my_step),
        g.edge_from(my_step).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=SimpleState())
    assert result == 42


async def test_add_edge_convenience():
    """Test the add_edge convenience method."""
    g = GraphBuilder(state_type=SimpleState, output_type=int)

    @g.step
    async def step_a(ctx: StepContext[SimpleState, None, None]) -> int:
        return 42

    @g.step
    async def step_b(ctx: StepContext[SimpleState, None, int]) -> int:
        return ctx.inputs + 1

    g.add_edge(g.start_node, step_a)
    g.add_edge(step_a, step_b, label='from a to b')
    g.add_edge(step_b, g.end_node)

    graph = g.build()
    result = await graph.run(state=SimpleState())
    assert result == 43


async def test_graph_with_dependencies():
    """Test graph execution with dependency injection."""

    @dataclass
    class MyDeps:
        multiplier: int

    g = GraphBuilder(state_type=SimpleState, deps_type=MyDeps, output_type=int)

    @g.step
    async def multiply(ctx: StepContext[SimpleState, MyDeps, None]) -> int:
        return ctx.deps.multiplier * 10

    g.add(
        g.edge_from(g.start_node).to(multiply),
        g.edge_from(multiply).to(g.end_node),
    )

    graph = g.build()
    state = SimpleState()
    deps = MyDeps(multiplier=5)
    result = await graph.run(state=state, deps=deps)
    assert result == 50


async def test_empty_graph():
    """Test that a minimal graph can be built and run."""
    g = GraphBuilder(output_type=int)

    g.add(g.edge_from(g.start_node).to(g.end_node))

    graph = g.build()
    result = await graph.run(inputs=42)
    assert result == 42


async def test_graph_name_inference():
    """Test that graph names are properly inferred from variable names."""
    my_graph_builder = GraphBuilder(output_type=int)

    @my_graph_builder.step
    async def return_value(ctx: StepContext[None, None, None]) -> int:
        return 100

    my_graph_builder.add(
        my_graph_builder.edge_from(my_graph_builder.start_node).to(return_value),
        my_graph_builder.edge_from(return_value).to(my_graph_builder.end_node),
    )

    my_custom_graph = my_graph_builder.build()
    result = await my_custom_graph.run()
    assert result == 100
    assert my_custom_graph.name == 'my_custom_graph'


async def test_explicit_graph_name():
    """Test setting an explicit graph name."""
    g = GraphBuilder(name='ExplicitName', output_type=int)

    g.add(g.edge_from(g.start_node).to(g.end_node))

    graph = g.build()
    assert graph.name == 'ExplicitName'


async def test_state_mutation():
    """Test that state mutations persist across steps."""
    g = GraphBuilder(state_type=SimpleState, output_type=str)

    @g.step
    async def set_counter(ctx: StepContext[SimpleState, None, None]) -> None:
        ctx.state.counter = 10

    @g.step
    async def set_result(ctx: StepContext[SimpleState, None, None]) -> None:
        ctx.state.result = f'counter={ctx.state.counter}'

    @g.step
    async def get_result(ctx: StepContext[SimpleState, None, None]) -> str:
        assert ctx.state.result is not None
        return ctx.state.result

    g.add(
        g.edge_from(g.start_node).to(set_counter),
        g.edge_from(set_counter).to(set_result),
        g.edge_from(set_result).to(get_result),
        g.edge_from(get_result).to(g.end_node),
    )

    graph = g.build()
    state = SimpleState()
    result = await graph.run(state=state)
    assert result == 'counter=10'
    assert state.counter == 10
    assert state.result == 'counter=10'
