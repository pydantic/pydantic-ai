from __future__ import annotations as _annotations

from typing import assert_type

from pydantic_ai_graph import BaseNode, End, Graph, GraphContext


class Float2String(BaseNode[float]):
    async def run(self, ctx: GraphContext) -> String2Length:
        return String2Length(str(self.input_data))


class String2Length(BaseNode[str]):
    async def run(self, ctx: GraphContext) -> Double:
        return Double(len(self.input_data))


class Double(BaseNode[int, int]):
    async def run(self, ctx: GraphContext) -> String2Length | End[int]:
        if self.input_data == 7:
            return String2Length('x' * 21)
        else:
            return End(self.input_data * 2)


def use_double(node: BaseNode[int, int]) -> None:
    """Shoe that `Double` is valid as a `BaseNode[int, int]`."""


use_double(Double(1))


g1 = Graph[float, int](
    Float2String,
    String2Length,
    Double,
)
assert_type(g1, Graph[float, int])


g2 = Graph(Float2String, Double)
assert_type(g2, Graph[float, int])

g3 = Graph(
    Float2String,
    Double,
    String2Length,
)
MYPY = False
if MYPY:
    # with mypy the presence of `String2Length` makes the output type Any
    assert_type(g3, Graph[float])  # pyright: ignore[reportAssertTypeFailure]
else:
    # pyright works correct and uses `Double` to infer the output type
    assert_type(g3, Graph[float, int])

g4 = Graph(
    Float2String,
    String2Length,
    Double,
)
# because String2Length came before Double, the output type is Any
assert_type(g4, Graph[float])
