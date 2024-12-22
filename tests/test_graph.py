from __future__ import annotations as _annotations

import pytest

from pydantic_ai_graph import BaseNode, End, Graph, GraphContext

pytestmark = pytest.mark.anyio


async def test_graph():
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

    g = Graph[float, int](
        Float2String,
        String2Length,
        Double,
    )
    # len('3.14') * 2 == 8
    assert await g.run(3.14) == (8, None)
    # len('3.14159') == 7, 21 * 2 == 42
    assert await g.run(3.14159) == (42, None)
