"""Tests for edge labels and path building."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pydantic_graph.beta import GraphBuilder, StepContext
from pydantic_graph.beta.join import reduce_list_append

pytestmark = pytest.mark.anyio


@dataclass
class LabelState:
    value: int = 0


async def test_edge_with_label():
    """Test adding labels to edges."""
    g = GraphBuilder(state_type=LabelState, output_type=int)

    @g.step
    async def step_a(ctx: StepContext[LabelState, None, None]) -> int:
        return 10

    @g.step
    async def step_b(ctx: StepContext[LabelState, None, int]) -> int:
        return ctx.inputs * 2

    g.add(
        g.edge_from(g.start_node).label('start to A').to(step_a),
        g.edge_from(step_a).label('A to B').to(step_b),
        g.edge_from(step_b).label('B to end').to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=LabelState())
    assert result == 20


async def test_multiple_labels_in_path():
    """Test multiple labels within a single path."""
    g = GraphBuilder(state_type=LabelState, output_type=int)

    @g.step
    async def step_a(ctx: StepContext[LabelState, None, None]) -> int:
        return 5

    @g.step
    async def step_b(ctx: StepContext[LabelState, None, int]) -> int:
        return ctx.inputs + 10

    g.add(
        g.edge_from(g.start_node).label('first label').label('second label').to(step_a),
        g.edge_from(step_a).to(step_b),
        g.edge_from(step_b).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=LabelState())
    assert result == 15


async def test_label_before_map():
    """Test label placement before a map operation."""
    g = GraphBuilder(state_type=LabelState, output_type=list[int])

    @g.step
    async def generate(ctx: StepContext[LabelState, None, None]) -> list[int]:
        return [1, 2, 3]

    @g.step
    async def double(ctx: StepContext[LabelState, None, int]) -> int:
        return ctx.inputs * 2

    collect = g.join(reduce_list_append, initial_factory=list[int])

    g.add(
        g.edge_from(g.start_node).to(generate),
        g.edge_from(generate).label('before map').map().label('after map').to(double),
        g.edge_from(double).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=LabelState())
    assert sorted(result) == [2, 4, 6]


async def test_labeled_broadcast():
    """Test labels on broadcast edges."""
    g = GraphBuilder(state_type=LabelState, output_type=list[int])

    @g.step
    async def source(ctx: StepContext[LabelState, None, None]) -> int:
        return 10

    @g.step
    async def path_a(ctx: StepContext[LabelState, None, int]) -> int:
        return ctx.inputs + 1

    @g.step
    async def path_b(ctx: StepContext[LabelState, None, int]) -> int:
        return ctx.inputs + 2

    collect = g.join(reduce_list_append, initial_factory=list[int])

    g.add(
        g.edge_from(g.start_node).to(source),
        g.edge_from(source).label('broadcasting').to(path_a, path_b),
        g.edge_from(path_a).label('from A').to(collect),
        g.edge_from(path_b).label('from B').to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=LabelState())
    assert sorted(result) == [11, 12]


async def test_label_on_decision_branch():
    """Test labels on decision branches."""
    from typing import Literal

    from pydantic_graph.beta import TypeExpression

    g = GraphBuilder(state_type=LabelState, output_type=str)

    @g.step
    async def choose(ctx: StepContext[LabelState, None, object]) -> Literal['a', 'b']:
        return 'a'

    @g.step
    async def path_a(ctx: StepContext[LabelState, None, object]) -> str:
        return 'A'

    @g.step
    async def path_b(ctx: StepContext[LabelState, None, object]) -> str:
        return 'B'  # pragma: no cover

    g.add(
        g.edge_from(g.start_node).to(choose),
        g.edge_from(choose).to(
            g.decision()
            .branch(g.match(TypeExpression[Literal['a']]).label('choose A').to(path_a))
            .branch(g.match(TypeExpression[Literal['b']]).label('choose B').to(path_b))
        ),
        g.edge_from(path_a, path_b).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=LabelState())
    assert result == 'A'


async def test_label_with_lambda_fork():
    """Test labels with lambda-style fork definitions."""
    g = GraphBuilder(state_type=LabelState, output_type=list[int])

    @g.step
    async def source(ctx: StepContext[LabelState, None, None]) -> int:
        return 5

    @g.step
    async def fork_a(ctx: StepContext[LabelState, None, int]) -> int:
        return ctx.inputs + 1

    @g.step
    async def fork_b(ctx: StepContext[LabelState, None, int]) -> int:
        return ctx.inputs + 2

    collect = g.join(reduce_list_append, initial_factory=list[int])

    g.add(
        g.edge_from(g.start_node).to(source),
        g.edge_from(source).broadcast(
            lambda e: [
                e.label('to fork A').to(fork_a),
                e.label('to fork B').to(fork_b),
            ]
        ),
        g.edge_from(fork_a, fork_b).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=LabelState())
    assert sorted(result) == [6, 7]


async def test_complex_labeled_path():
    """Test a complex path with multiple labels, transforms, and operations."""
    g = GraphBuilder(state_type=LabelState, output_type=list[str])

    @g.step
    async def start(ctx: StepContext[LabelState, None, None]) -> list[int]:
        return [1, 2, 3]

    @g.step
    async def process(ctx: StepContext[LabelState, None, int]) -> int:
        return ctx.inputs * 2

    @g.step
    async def stringify(ctx: StepContext[LabelState, None, int]) -> str:
        return f'value={ctx.inputs}'

    collect = g.join(reduce_list_append, initial_factory=list[str])

    g.add(
        g.edge_from(g.start_node).label('initialize').to(start),
        g.edge_from(start).label('before map').map().label('mapping').to(process),
        g.edge_from(process).label('to stringify').to(stringify),
        g.edge_from(stringify).label('collecting').to(collect),
        g.edge_from(collect).label('done').to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=LabelState())
    assert sorted(result) == ['value=2', 'value=4', 'value=6']
