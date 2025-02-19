from __future__ import annotations as _annotations

from dataclasses import dataclass

from typing_extensions import assert_type

from pydantic_graph import BaseNode, End, Graph, GraphRunContext, HistoryStep


@dataclass
class Float2String(BaseNode):
    input_data: float

    async def run(self, ctx: GraphRunContext) -> String2Length:
        return String2Length(str(self.input_data))


@dataclass
class String2Length(BaseNode):
    input_data: str

    async def run(self, ctx: GraphRunContext) -> Double:
        return Double(len(self.input_data))


@dataclass
class X:
    v: int


@dataclass
class Double(BaseNode[None, X]):
    input_data: int

    async def run(self, ctx: GraphRunContext) -> String2Length | End[X]:
        if self.input_data == 7:
            return String2Length('x' * 21)
        else:
            return End(X(self.input_data * 2))


def use_double(node: BaseNode[None, X]) -> None:
    """Shoe that `Double` is valid as a `BaseNode[None, int, X]`."""
    print(node)


use_double(Double(1))


g1 = Graph[None, X](
    nodes=(
        Float2String,
        String2Length,
        Double,
    )
)
assert_type(g1, Graph[None, X])


g2 = Graph(nodes=(Double,))
assert_type(g2, Graph[None, X])

g3 = Graph(
    nodes=(
        Float2String,
        String2Length,
        Double,
    )
)
# because String2Length came before Double, the output type is Any
assert_type(g3, Graph[None, X])

Graph[None, bytes](nodes=(Float2String, String2Length, Double))  # type: ignore[arg-type]
Graph[None, str](nodes=[Double])  # type: ignore[list-item]


@dataclass
class MyState:
    x: int


@dataclass
class MyDeps:
    y: str


@dataclass
class A(BaseNode[MyDeps]):
    state: MyState

    async def run(self, ctx: GraphRunContext[MyDeps]) -> B:
        assert self.state.x == 1
        assert ctx.deps.y == 'y'
        return B(self.state)


@dataclass
class B(BaseNode[MyDeps, int]):
    state: MyState

    async def run(self, ctx: GraphRunContext[MyDeps]) -> End[int]:
        return End(42)


g4 = Graph[MyDeps, int](nodes=(A, B))
assert_type(g4, Graph[MyDeps, int])

g5 = Graph(nodes=(A, B))
assert_type(g5, Graph[MyDeps, int])


def run_g5() -> None:
    g5.run_sync(A(MyState(x=1)))  # pyright: ignore[reportArgumentType]
    g5.run_sync(A(MyState(x=1)))  # pyright: ignore[reportArgumentType]
    answer, history = g5.run_sync(A(MyState(x=1)), deps=MyDeps(y='y'))
    assert_type(answer, int)
    assert_type(history, list[HistoryStep[int]])
