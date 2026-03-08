"""Tests for decision nodes and conditional branching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest

from pydantic_graph import BaseNode, End, GraphRunContext
from pydantic_graph.beta import GraphBuilder, StepContext, TypeExpression
from pydantic_graph.beta.join import reduce_list_append, reduce_sum

pytestmark = pytest.mark.anyio


@dataclass
class DecisionState:
    path_taken: str | None = None
    value: int = 0


async def test_simple_decision_literal():
    """Test a simple decision node with literal type matching."""
    g = GraphBuilder(state_type=DecisionState, output_type=str)

    @g.step
    async def choose_path(ctx: StepContext[DecisionState, None, None]) -> Literal['left', 'right']:
        return 'left'

    @g.step
    async def left_path(ctx: StepContext[DecisionState, None, object]) -> str:
        ctx.state.path_taken = 'left'
        return 'Went left'

    @g.step
    async def right_path(ctx: StepContext[DecisionState, None, object]) -> str:  # pragma: no cover
        ctx.state.path_taken = 'right'
        return 'Went right'

    g.add(
        g.edge_from(g.start_node).to(choose_path),
        g.edge_from(choose_path).to(
            g.decision()
            .branch(g.match(TypeExpression[Literal['left']]).to(left_path))
            .branch(g.match(TypeExpression[Literal['right']]).to(right_path))
        ),
        g.edge_from(left_path, right_path).to(g.end_node),
    )

    graph = g.build()
    state = DecisionState()
    result = await graph.run(state=state)
    assert result == 'Went left'
    assert state.path_taken == 'left'


async def test_decision_with_type_matching():
    """Test decision node matching by type."""
    g = GraphBuilder(state_type=DecisionState, output_type=str)

    @g.step
    async def return_int(ctx: StepContext[DecisionState, None, None]) -> int:
        return 42

    @g.step
    async def handle_int(ctx: StepContext[DecisionState, None, int]) -> str:
        return f'Got int: {ctx.inputs}'

    @g.step
    async def handle_str(ctx: StepContext[DecisionState, None, str]) -> str:
        return f'Got str: {ctx.inputs}'  # pragma: no cover

    g.add(
        g.edge_from(g.start_node).to(return_int),
        g.edge_from(return_int).to(
            g.decision()
            .branch(g.match(TypeExpression[int]).to(handle_int))
            .branch(g.match(TypeExpression[str]).to(handle_str))
        ),
        g.edge_from(handle_int, handle_str).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=DecisionState())
    assert result == 'Got int: 42'


async def test_decision_with_custom_matcher():
    """Test decision node with custom matching function."""
    g = GraphBuilder(state_type=DecisionState, output_type=str)

    @g.step
    async def return_number(ctx: StepContext[DecisionState, None, None]) -> int:
        return 7

    @g.step
    async def even_path(ctx: StepContext[DecisionState, None, int]) -> str:
        return f'{ctx.inputs} is even'  # pragma: no cover

    @g.step
    async def odd_path(ctx: StepContext[DecisionState, None, int]) -> str:
        return f'{ctx.inputs} is odd'

    g.add(
        g.edge_from(g.start_node).to(return_number),
        g.edge_from(return_number).to(
            g.decision()
            .branch(g.match(TypeExpression[int], matches=lambda x: x % 2 == 0).to(even_path))
            .branch(g.match(TypeExpression[int], matches=lambda x: x % 2 == 1).to(odd_path))
        ),
        g.edge_from(even_path, odd_path).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=DecisionState())
    assert result == '7 is odd'


async def test_decision_with_state_modification():
    """Test that decision branches can modify state."""
    g = GraphBuilder(state_type=DecisionState, output_type=int)

    @g.step
    async def get_value(ctx: StepContext[DecisionState, None, None]) -> int:
        return 5

    @g.step
    async def small_value(ctx: StepContext[DecisionState, None, int]) -> int:
        ctx.state.path_taken = 'small'
        return ctx.inputs * 2

    @g.step
    async def large_value(ctx: StepContext[DecisionState, None, int]) -> int:  # pragma: no cover
        ctx.state.path_taken = 'large'
        return ctx.inputs * 10

    g.add(
        g.edge_from(g.start_node).to(get_value),
        g.edge_from(get_value).to(
            g.decision()
            .branch(g.match(TypeExpression[int], matches=lambda x: x < 10).to(small_value))
            .branch(g.match(TypeExpression[int], matches=lambda x: x >= 10).to(large_value))
        ),
        g.edge_from(small_value, large_value).to(g.end_node),
    )

    graph = g.build()
    state = DecisionState()
    result = await graph.run(state=state)
    assert result == 10
    assert state.path_taken == 'small'


async def test_decision_all_types_match():
    """Test decision with a branch that matches all types."""
    g = GraphBuilder(state_type=DecisionState, output_type=str)

    @g.step
    async def return_value(ctx: StepContext[DecisionState, None, None]) -> int:
        return 100

    @g.step
    async def catch_all(ctx: StepContext[DecisionState, None, object]) -> str:
        return f'Caught: {ctx.inputs}'

    g.add(
        g.edge_from(g.start_node).to(return_value),
        g.edge_from(return_value).to(g.decision().branch(g.match(TypeExpression[object]).to(catch_all))),
        g.edge_from(catch_all).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=DecisionState())
    assert result == 'Caught: 100'


async def test_decision_first_match_wins():
    """Test that the first matching branch is taken."""
    g = GraphBuilder(state_type=DecisionState, output_type=str)

    @g.step
    async def return_value(ctx: StepContext[DecisionState, None, None]) -> int:
        return 10

    @g.step
    async def branch_a(ctx: StepContext[DecisionState, None, int]) -> str:
        return 'Branch A'

    @g.step
    async def branch_b(ctx: StepContext[DecisionState, None, int]) -> str:
        return 'Branch B'  # pragma: no cover

    g.add(
        g.edge_from(g.start_node).to(return_value),
        g.edge_from(return_value).to(
            g.decision()
            # Both branches match, but A is first
            .branch(g.match(TypeExpression[int], matches=lambda x: x >= 5).to(branch_a))
            .branch(g.match(TypeExpression[int], matches=lambda x: x >= 0).to(branch_b))
        ),
        g.edge_from(branch_a, branch_b).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=DecisionState())
    assert result == 'Branch A'


async def test_nested_decisions():
    """Test nested decision nodes."""
    g = GraphBuilder(state_type=DecisionState, output_type=str)

    @g.step
    async def get_number(ctx: StepContext[DecisionState, None, None]) -> int:
        return 15

    @g.step
    async def is_positive(ctx: StepContext[DecisionState, None, int]) -> int:
        return ctx.inputs

    @g.step
    async def is_negative(ctx: StepContext[DecisionState, None, int]) -> str:
        return 'Negative'  # pragma: no cover

    @g.step
    async def small_positive(ctx: StepContext[DecisionState, None, int]) -> str:
        return 'Small positive'  # pragma: no cover

    @g.step
    async def large_positive(ctx: StepContext[DecisionState, None, int]) -> str:
        return 'Large positive'

    g.add(
        g.edge_from(g.start_node).to(get_number),
        g.edge_from(get_number).to(
            g.decision()
            .branch(g.match(TypeExpression[int], matches=lambda x: x > 0).to(is_positive))
            .branch(g.match(TypeExpression[int], matches=lambda x: x <= 0).to(is_negative))
        ),
        g.edge_from(is_positive).to(
            g.decision()
            .branch(g.match(TypeExpression[int], matches=lambda x: x < 10).to(small_positive))
            .branch(g.match(TypeExpression[int], matches=lambda x: x >= 10).to(large_positive))
        ),
        g.edge_from(is_negative, small_positive, large_positive).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=DecisionState())
    assert result == 'Large positive'


async def test_decision_with_label():
    """Test adding labels to decision branches."""
    g = GraphBuilder(state_type=DecisionState, output_type=str)

    @g.step
    async def choose(ctx: StepContext[DecisionState, None, None]) -> Literal['a', 'b']:
        return 'a'

    @g.step
    async def path_a(ctx: StepContext[DecisionState, None, object]) -> str:
        return 'Path A'

    @g.step
    async def path_b(ctx: StepContext[DecisionState, None, object]) -> str:
        return 'Path B'  # pragma: no cover

    g.add(
        g.edge_from(g.start_node).to(choose),
        g.edge_from(choose).to(
            g.decision()
            .branch(g.match(TypeExpression[Literal['a']]).label('Take path A').to(path_a))
            .branch(g.match(TypeExpression[Literal['b']]).label('Take path B').to(path_b))
        ),
        g.edge_from(path_a, path_b).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=DecisionState())
    assert result == 'Path A'


async def test_decision_with_map():
    """Test decision branch that maps output."""
    g = GraphBuilder(state_type=DecisionState, output_type=int)

    @g.step
    async def get_type(ctx: StepContext[DecisionState, None, object]) -> Literal['list', 'single']:
        return 'list'

    @g.step
    async def make_list(ctx: StepContext[DecisionState, None, object]) -> list[int]:
        return [1, 2, 3]

    @g.step
    async def make_single(ctx: StepContext[DecisionState, None, object]) -> int:
        return 10  # pragma: no cover

    @g.step
    async def process_item(ctx: StepContext[DecisionState, None, int]) -> int:
        ctx.state.value += ctx.inputs
        return ctx.inputs

    @g.step
    async def get_value(ctx: StepContext[DecisionState, None, object]) -> int:
        return ctx.state.value

    g.add(
        g.edge_from(g.start_node).to(get_type),
        g.edge_from(get_type).to(
            g.decision()
            .branch(g.match(TypeExpression[Literal['list']]).to(make_list))
            .branch(g.match(TypeExpression[Literal['single']]).to(make_single))
        ),
        g.edge_from(make_list).map().to(process_item),
        g.edge_from(make_single).to(process_item),
        g.edge_from(process_item).to(get_value),
        g.edge_from(get_value).to(g.end_node),
    )

    graph = g.build()
    state = DecisionState()
    result = await graph.run(state=state)
    assert result == 6
    assert state.value == 6  # 1 + 2 + 3


async def test_decision_branch_transform():
    """Test DecisionBranchBuilder.transform method."""
    g = GraphBuilder(state_type=DecisionState, output_type=str)

    @g.step
    async def get_value(ctx: StepContext[DecisionState, None, None]) -> int:
        return 10

    @g.step
    async def format_result(ctx: StepContext[DecisionState, None, str]) -> str:
        return f'Result: {ctx.inputs}'

    def double_value(ctx: StepContext[DecisionState, None, int]) -> str:
        return str(ctx.inputs * 2)

    g.add(
        g.edge_from(g.start_node).to(get_value),
        g.edge_from(get_value).to(g.decision().branch(g.match(int).transform(double_value).to(format_result))),
        g.edge_from(format_result).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=DecisionState())
    assert result == 'Result: 20'


async def test_decision_branch_map():
    """Test DecisionBranchBuilder.map method."""
    g = GraphBuilder(state_type=DecisionState, output_type=str)

    @g.step
    async def get_value(ctx: StepContext[DecisionState, None, None]) -> int | list[int]:
        return [1, 2, 3, 4, 5, 6]

    @g.step
    async def format_result(ctx: StepContext[DecisionState, None, object]) -> str:
        return f'Result: {ctx.inputs}'

    join_sum = g.join(reduce_sum, initial=0)

    def double_value(ctx: StepContext[DecisionState, None, int]) -> int:
        return ctx.inputs * 2

    g.add(
        g.edge_from(g.start_node).to(get_value),
        g.edge_from(get_value).to(
            g.decision()
            .branch(g.match(int).transform(double_value).to(format_result))
            .branch(
                g.match(list[int], matches=lambda x: isinstance(x, list)).map().transform(double_value).to(join_sum)
            )
        ),
        g.edge_from(join_sum).to(format_result),
        g.edge_from(format_result).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=DecisionState())
    assert result == 'Result: 42'


async def test_decision_branch_label():
    """Test DecisionBranchBuilder.label method."""
    g = GraphBuilder(state_type=DecisionState, output_type=str)

    @g.step
    async def get_value(ctx: StepContext[DecisionState, None, None]) -> Literal['a', 'b']:
        return 'a'

    @g.step
    async def handle_a(ctx: StepContext[DecisionState, None, object]) -> str:
        return 'Got A'

    @g.step
    async def handle_b(ctx: StepContext[DecisionState, None, object]) -> str:
        return 'Got B'  # pragma: no cover

    g.add(
        g.edge_from(g.start_node).to(get_value),
        g.edge_from(get_value).to(
            g.decision()
            .branch(g.match(TypeExpression[Literal['a']]).label('path A').to(handle_a))
            .branch(g.match(TypeExpression[Literal['b']]).label('path B').to(handle_b))
        ),
        g.edge_from(handle_a, handle_b).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=DecisionState())
    assert result == 'Got A'


async def test_decision_branch_fork():
    """Test DecisionBranchBuilder.fork method."""
    g = GraphBuilder(state_type=DecisionState, output_type=list[str])

    @g.step
    async def choose_option(ctx: StepContext[DecisionState, None, None]) -> Literal['fork']:
        return 'fork'

    @g.step
    async def path_1(ctx: StepContext[DecisionState, None, object]) -> str:
        return 'Path 1'

    @g.step
    async def path_2(ctx: StepContext[DecisionState, None, object]) -> str:
        return 'Path 2'

    collect = g.join(reduce_list_append, initial_factory=list[str])

    g.add(
        g.edge_from(g.start_node).to(choose_option),
        g.edge_from(choose_option).to(
            g.decision().branch(
                g.match(TypeExpression[Literal['fork']]).broadcast(
                    lambda b: [
                        b.to(path_1),
                        b.to(path_2),
                    ]
                )
            )
        ),
        g.edge_from(path_1, path_2).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=DecisionState())
    assert sorted(result) == ['Path 1', 'Path 2']


async def test_empty_decision_broadcast():
    """Test DecisionBranchBuilder.fork method."""
    g = GraphBuilder(state_type=DecisionState, output_type=list[str])
    with pytest.raises(ValueError, match=r'returned no branches, but must return at least one'):
        g.match(TypeExpression[Literal['fork']]).broadcast(lambda b: [])


async def test_match_node():
    """Test using match_node() with BaseNode types in decisions.

    match_node() is designed for exhaustive matching of BaseNode return types
    in decision branches. Unlike match().to(), it doesn't require a .to() call
    since the destination is the BaseNode class itself.

    This is only necessary if you have a step that might return a v1-style node _or_ an
    arbitrary output that you want to route to another node using the builder API.
    """
    g = GraphBuilder(state_type=DecisionState, input_type=int, output_type=str)

    @dataclass
    class NodeStep(BaseNode[DecisionState, None, str]):
        value: int

        async def run(self, ctx: GraphRunContext[DecisionState, None]) -> End[str]:
            ctx.state.path_taken = 'path_a'
            return End(f'Path A: {self.value}')

    @g.step
    async def regular_step(ctx: StepContext[DecisionState, None, int]):
        ctx.state.path_taken = 'path_b'
        return f'Path B: {ctx.inputs}'

    @g.step
    async def route_to_node(ctx: StepContext[DecisionState, None, int]) -> NodeStep | int:
        # Route based on input value
        if ctx.inputs < 10:
            return NodeStep(ctx.inputs)
        else:
            return ctx.inputs

    # Use match_node to create decision branches for BaseNode types
    # Note: match_node doesn't require .to() - the node type IS the destination
    g.add(
        g.node(NodeStep),
        g.edge_from(g.start_node).to(route_to_node),
        g.edge_from(route_to_node).to(
            g.decision().branch(g.match_node(NodeStep)).branch(g.match(int).to(regular_step))
        ),
        g.edge_from(regular_step).to(g.end_node),
    )

    graph = g.build()

    # Test path A (value < 10)
    state_a = DecisionState()
    result_a = await graph.run(state=state_a, inputs=5)
    assert result_a == 'Path A: 5'
    assert state_a.path_taken == 'path_a'

    # Test path B (value >= 10)
    state_b = DecisionState()
    result_b = await graph.run(state=state_b, inputs=15)
    assert result_b == 'Path B: 15'
    assert state_b.path_taken == 'path_b'
