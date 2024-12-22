from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import assert_type

from pydantic_ai_graph import BaseNode, End, Graph, GraphContext


class Float2String(BaseNode[None, float]):
    async def run(self, ctx: GraphContext) -> String2Length:
        return String2Length(str(self.input_data))


class String2Length(BaseNode[None, str]):
    async def run(self, ctx: GraphContext) -> Double:
        return Double(len(self.input_data))


@dataclass
class X:
    v: int


class Double(BaseNode[None, int, X]):
    async def run(self, ctx: GraphContext) -> String2Length | End[X]:
        if self.input_data == 7:
            return String2Length('x' * 21)
        else:
            return End(X(self.input_data * 2))


def use_double(node: BaseNode[None, int, X]) -> None:
    """Shoe that `Double` is valid as a `BaseNode[None, int, X]`."""
    print(node)


use_double(Double(1))


g1 = Graph[None, float, X](
    Float2String,
    String2Length,
    Double,
)
assert_type(g1, Graph[None, float, X])


g2 = Graph(Double)
assert_type(g2, Graph[None, int, X])

g3 = Graph(
    Float2String,
    String2Length,
    Double,
)
# because String2Length came before Double, the output type is Any
assert_type(g3, Graph[None, float])

Graph[None, float, bytes](Float2String, String2Length, Double)  # type: ignore[arg-type]
Graph[None, int, str](Double)  # type: ignore[arg-type]
