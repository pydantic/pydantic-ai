"""Tests for Mermaid diagram rendering of beta graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest

from pydantic_graph.beta import GraphBuilder, StepContext, TypeExpression
from pydantic_graph.beta.mermaid import build_mermaid_graph

pytestmark = pytest.mark.anyio


@dataclass
class SimpleState:
    counter: int = 0


async def test_render_with_step_label():
    """Test that step labels appear in mermaid output"""
    g = GraphBuilder(state_type=SimpleState, output_type=int)

    @g.step(label='Process Data')
    async def process(ctx: StepContext[SimpleState, None, None]) -> int:
        return 42  # pragma: no cover

    g.add(
        g.edge_from(g.start_node).to(process),
        g.edge_from(process).to(g.end_node),
    )

    graph = g.build()
    mermaid = graph.render()
    assert 'Process Data' in mermaid


async def test_render_with_edge_labels():
    """Test that edge labels appear in mermaid output"""
    g = GraphBuilder(state_type=SimpleState, output_type=int)

    @g.step
    async def step_a(ctx: StepContext[SimpleState, None, None]) -> int:
        return 10  # pragma: no cover

    @g.step
    async def step_b(ctx: StepContext[SimpleState, None, int]) -> int:
        return ctx.inputs + 1  # pragma: no cover

    g.add(
        g.edge_from(g.start_node).label('start edge').to(step_a),
        g.edge_from(step_a).label('middle edge').to(step_b),
        g.edge_from(step_b).to(g.end_node),
    )

    graph = g.build()
    mermaid_graph = build_mermaid_graph(graph.nodes, graph.edges_by_source)
    mermaid = mermaid_graph.render(edge_labels=True)
    assert 'start edge' in mermaid
    assert 'middle edge' in mermaid


async def test_render_without_edge_labels():
    """Test that edge labels can be suppressed."""
    g = GraphBuilder(state_type=SimpleState, output_type=int)

    @g.step
    async def step_a(ctx: StepContext[SimpleState, None, None]) -> int:
        return 10  # pragma: no cover

    g.add(
        g.edge_from(g.start_node).label('hidden label').to(step_a),
        g.edge_from(step_a).to(g.end_node),
    )

    graph = g.build()
    mermaid_graph = build_mermaid_graph(graph.nodes, graph.edges_by_source)
    mermaid = mermaid_graph.render(edge_labels=False)
    assert 'hidden label' not in mermaid


async def test_render_decision_node():
    """Test rendering a decision node"""
    g = GraphBuilder(state_type=SimpleState, output_type=str)

    @g.step
    async def choose(ctx: StepContext[SimpleState, None, None]) -> Literal['a', 'b']:
        return 'a'  # pragma: no cover

    @g.step
    async def path_a(ctx: StepContext[SimpleState, None, object]) -> str:
        return 'A'  # pragma: no cover

    @g.step
    async def path_b(ctx: StepContext[SimpleState, None, object]) -> str:
        return 'B'  # pragma: no cover

    g.add(
        g.edge_from(g.start_node).to(choose),
        g.edge_from(choose).to(
            g.decision()
            .branch(g.match(TypeExpression[Literal['a']]).to(path_a))
            .branch(g.match(TypeExpression[Literal['b']]).to(path_b))
        ),
        g.edge_from(path_a, path_b).to(g.end_node),
    )

    graph = g.build()
    mermaid = graph.render()
    # Decision nodes should be rendered with <<choice>> marker
    assert '<<choice>>' in mermaid


async def test_render_decision_branches():
    """Test rendering decision branches"""
    g = GraphBuilder(state_type=SimpleState, output_type=int)

    @g.step
    async def get_value(ctx: StepContext[SimpleState, None, None]) -> int:
        return 5  # pragma: no cover

    @g.step
    async def small(ctx: StepContext[SimpleState, None, int]) -> int:
        return ctx.inputs * 2  # pragma: no cover

    @g.step
    async def large(ctx: StepContext[SimpleState, None, int]) -> int:
        return ctx.inputs * 10  # pragma: no cover

    g.add(
        g.edge_from(g.start_node).to(get_value),
        g.edge_from(get_value).to(
            g.decision()
            .branch(g.match(int, matches=lambda x: x < 10).to(small))
            .branch(g.match(int, matches=lambda x: x >= 10).to(large))
        ),
        g.edge_from(small, large).to(g.end_node),
    )

    graph = g.build()
    mermaid = graph.render()
    # Verify decision branches create edges from the decision node
    assert mermaid.count('-->') >= 2  # At least 2 edges from decision


async def test_render_decision_with_note():
    """Test rendering a decision with a note"""
    g = GraphBuilder(state_type=SimpleState, output_type=str)

    @g.step
    async def choose(ctx: StepContext[SimpleState, None, None]) -> Literal['x', 'y']:
        return 'x'  # pragma: no cover

    @g.step
    async def handler(ctx: StepContext[SimpleState, None, object]) -> str:
        return 'result'  # pragma: no cover

    g.add(
        g.edge_from(g.start_node).to(choose),
        g.edge_from(choose).to(
            g.decision(note='Route based on input')
            .branch(g.match(TypeExpression[Literal['x']]).to(handler))
            .branch(g.match(TypeExpression[Literal['y']]).to(handler))
        ),
        g.edge_from(handler).to(g.end_node),
    )

    graph = g.build()
    mermaid = graph.render()
    # Decision notes should appear in the mermaid output
    assert 'Route based on input' in mermaid
    assert 'note right of' in mermaid


async def test_render_with_direction():
    """Test rendering with explicit direction"""
    g = GraphBuilder(state_type=SimpleState, output_type=int)

    @g.step
    async def step(ctx: StepContext[SimpleState, None, None]) -> int:
        return 1  # pragma: no cover

    g.add(
        g.edge_from(g.start_node).to(step),
        g.edge_from(step).transform(lambda ctx: ctx.inputs * 2).to(g.end_node),
    )

    graph = g.build()

    # Test left-to-right direction
    mermaid_lr = graph.render(direction='LR')
    assert 'direction LR' in mermaid_lr

    # Test right-to-left direction
    mermaid_rl = graph.render(direction='RL')
    assert 'direction RL' in mermaid_rl
