"""Static-typing checks for the `pydantic_graph` builder API.

This file is not executed by pytest; it's checked by `pyright` and exists to
catch regressions in the public typing of [`GraphBuilder`][pydantic_graph.GraphBuilder]
and [`Graph`][pydantic_graph.Graph].
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass

from typing_extensions import assert_type

from pydantic_graph import BaseNode, End, Graph, GraphBuilder, GraphRunContext, StepContext


@dataclass
class MyState:
    x: int


@dataclass
class MyDeps:
    y: str


# `GraphBuilder` is generic in (StateT, DepsT, InputT, OutputT), in that order.
g1 = GraphBuilder(state_type=MyState, deps_type=MyDeps, input_type=str, output_type=int)
assert_type(g1, GraphBuilder[MyState, MyDeps, str, int])

g1_graph = g1.build(validate_graph_structure=False)
assert_type(g1_graph, Graph[MyState, MyDeps, str, int])


# Defaults: all four type parameters default to `None`.
g2 = GraphBuilder()
assert_type(g2, GraphBuilder[None, None, None, None])
assert_type(g2.build(validate_graph_structure=False), Graph[None, None, None, None])


# Partial parameterization is allowed; unspecified params fall back to `None`.
g3 = GraphBuilder(input_type=int, output_type=str)
assert_type(g3, GraphBuilder[None, None, int, str])


async def run_graph_returns_output_type() -> None:
    graph = GraphBuilder(state_type=MyState, deps_type=MyDeps, input_type=int, output_type=str).build(
        validate_graph_structure=False
    )
    result = await graph.run(state=MyState(x=1), deps=MyDeps(y='y'), inputs=5)
    assert_type(result, str)


async def run_graph_rejects_wrong_state() -> None:
    graph = GraphBuilder(state_type=MyState, output_type=str).build(validate_graph_structure=False)
    # state=... must match `state_type`; passing a `str` where `MyState` is expected should error.
    await graph.run(state='not a MyState', inputs=None)  # type: ignore[arg-type]


async def run_graph_rejects_wrong_deps() -> None:
    graph = GraphBuilder(deps_type=MyDeps, output_type=str).build(validate_graph_structure=False)
    await graph.run(deps=42, inputs=None)  # type: ignore[arg-type]


async def run_graph_rejects_wrong_inputs() -> None:
    graph = GraphBuilder(input_type=int, output_type=str).build(validate_graph_structure=False)
    await graph.run(inputs='not an int')  # type: ignore[arg-type]


# `BaseNode` subclasses parameterized by state/deps/run-end types still type-check.
@dataclass
class StartingNode(BaseNode[MyState, MyDeps, int]):
    async def run(self, ctx: GraphRunContext[MyState, MyDeps]) -> SecondNode:
        assert ctx.state.x == 1
        assert ctx.deps.y == 'y'
        return SecondNode()


@dataclass
class SecondNode(BaseNode[MyState, MyDeps, int]):
    async def run(self, ctx: GraphRunContext[MyState, MyDeps]) -> End[int]:
        return End(42)


# A `BaseNode` typed for one (State, Deps) pair should be assignable to the
# corresponding `BaseNode[...]` annotation.
def use_starting_node(node: BaseNode[MyState, MyDeps, int]) -> None:
    print(node)


use_starting_node(StartingNode())


# Step functions are typed via `StepContext` and their input/output types
# flow through the graph builder.
async def step_returns_output_type() -> None:
    g = GraphBuilder(state_type=MyState, deps_type=MyDeps, input_type=int, output_type=str)

    @g.step
    async def to_str(ctx: StepContext[MyState, MyDeps, int]) -> str:
        assert_type(ctx.inputs, int)
        assert_type(ctx.state, MyState)
        assert_type(ctx.deps, MyDeps)
        return str(ctx.inputs)

    g.add(g.edge_from(g.start_node).to(to_str), g.edge_from(to_str).to(g.end_node))
    graph = g.build()
    assert_type(graph, Graph[MyState, MyDeps, int, str])
    result = await graph.run(state=MyState(x=1), deps=MyDeps(y='y'), inputs=5)
    assert_type(result, str)
