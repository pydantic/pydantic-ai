from __future__ import annotations as _annotations

from datetime import timezone

import pytest
from inline_snapshot import snapshot

from pydantic_ai_graph import BaseNode, End, Graph, GraphContext, Snapshot

from .conftest import IsFloat, IsNow

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
    result, history = await g.run(3.14)
    # len('3.14') * 2 == 8
    assert result == 8
    assert history == snapshot(
        [
            Snapshot(
                last_node_id='Float2String',
                next_node_id='String2Length',
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(gt=0, lt=1e-5),
            ),
            Snapshot(
                last_node_id='String2Length',
                next_node_id='Double',
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(gt=0, lt=1e-5),
            ),
            Snapshot(
                last_node_id='Double',
                next_node_id=None,
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(gt=0, lt=1e-5),
            ),
        ]
    )
    result, history = await g.run(3.14159)
    # len('3.14159') == 7, 21 * 2 == 42
    assert result == 42
    assert history == snapshot(
        [
            Snapshot(
                last_node_id='Float2String',
                next_node_id='String2Length',
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(gt=0, lt=1e-5),
            ),
            Snapshot(
                last_node_id='String2Length',
                next_node_id='Double',
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(gt=0, lt=1e-5),
            ),
            Snapshot(
                last_node_id='Double',
                next_node_id='String2Length',
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(gt=0, lt=1e-5),
            ),
            Snapshot(
                last_node_id='String2Length',
                next_node_id='Double',
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(gt=0, lt=1e-5),
            ),
            Snapshot(
                last_node_id='Double',
                next_node_id=None,
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(gt=0, lt=1e-5),
            ),
        ]
    )
