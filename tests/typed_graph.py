from __future__ import annotations as _annotations

from dataclasses import dataclass

from typing_extensions import assert_type

from pydantic_ai_graph import BaseNode, End, Graph, GraphContext


@dataclass
class Float2String(BaseNode):
    input_data: float

    async def run(self, ctx: GraphContext) -> String2Length:
        return String2Length(str(self.input_data))


@dataclass
class String2Length(BaseNode):
    input_data: str

    async def run(self, ctx: GraphContext) -> Double:
        return Double(len(self.input_data))


@dataclass
class X:
    v: int


@dataclass
class Double(BaseNode[None, X]):
    input_data: int

    async def run(self, ctx: GraphContext) -> String2Length | End[X]:
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

Graph[None, bytes](Float2String, String2Length, Double)  # type: ignore[arg-type]
Graph[None, str](Double)  # type: ignore[arg-type]
