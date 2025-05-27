from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Any

from typing_extensions import assert_type

from pydantic_graph import BaseNode, End, FullStatePersistence, Graph, GraphRunContext
from pydantic_graph.persistence import BaseStatePersistence


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
class Double(BaseNode[None, None, X]):
    input_data: int

    async def run(self, ctx: GraphRunContext) -> String2Length | End[X]:
        if self.input_data == 7:
            return String2Length('x' * 21)
        else:
            return End(X(self.input_data * 2))


def use_double(node: BaseNode[None, None, X]) -> None:
    """Shoe that `Double` is valid as a `BaseNode[None, int, X]`."""
    print(node)


use_double(Double(1))


g1 = Graph[None, None, X](
    nodes=(
        Float2String,
        String2Length,
        Double,
    )
)
assert_type(g1, Graph[None, None, X])


g2 = Graph(nodes=(Double,))
assert_type(g2, Graph[None, None, X])

g3 = Graph(
    nodes=(
        Float2String,
        String2Length,
        Double,
    )
)
# because String2Length came before Double, the output type is Any
assert_type(g3, Graph[None, None, X])

Graph[None, bytes](nodes=(Float2String, String2Length, Double))  # type: ignore[arg-type]
Graph[None, str](nodes=[Double])  # type: ignore[list-item]


@dataclass
class MyState:
    x: int


@dataclass
class MyDeps:
    y: str


@dataclass
class A(BaseNode[MyState, MyDeps]):
    async def run(self, ctx: GraphRunContext[MyState, MyDeps]) -> B:
        assert ctx.state.x == 1
        assert ctx.deps.y == 'y'
        return B()


@dataclass
class B(BaseNode[MyState, MyDeps, int]):
    async def run(self, ctx: GraphRunContext[MyState, MyDeps]) -> End[int]:
        return End(42)


g4 = Graph[MyState, MyDeps, int](nodes=(A, B))
assert_type(g4, Graph[MyState, MyDeps, int])

g5 = Graph(nodes=(A, B))
assert_type(g5, Graph[MyState, MyDeps, int])


def run_g5() -> None:
    g5.run_sync(A())  # pyright: ignore[reportArgumentType]
    g5.run_sync(A(), state=MyState(x=1))  # pyright: ignore[reportArgumentType]
    g5.run_sync(A(), deps=MyDeps(y='y'))  # pyright: ignore[reportArgumentType]
    result = g5.run_sync(A(), state=MyState(x=1), deps=MyDeps(y='y'))
    assert_type(result.output, int)


def run_g6() -> None:
    result = g5.run_sync(A(), state=MyState(x=1), deps=MyDeps(y='y'))
    assert_type(result.output, int)
    assert_type(result.persistence, BaseStatePersistence[MyState, int])


p = FullStatePersistence()
assert_type(p, FullStatePersistence[Any, Any])


def run_persistence_any() -> None:
    p = FullStatePersistence()
    result = g5.run_sync(A(), persistence=p, state=MyState(x=1), deps=MyDeps(y='y'))
    assert_type(result.output, int)
    assert_type(p, FullStatePersistence[Any, Any])


def run_persistence_right() -> None:
    p = FullStatePersistence[MyState, int]()
    result = g5.run_sync(A(), persistence=p, state=MyState(x=1), deps=MyDeps(y='y'))
    assert_type(result.output, int)
    assert_type(p, FullStatePersistence[MyState, int])


def run_persistence_wrong() -> None:
    p = FullStatePersistence[str, int]()
    g5.run_sync(A(), persistence=p, state=MyState(x=1), deps=MyDeps(y='y'))  # type: ignore[arg-type]
