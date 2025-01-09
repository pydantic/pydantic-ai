from __future__ import annotations as _annotations

from dataclasses import dataclass
from datetime import timezone
from typing import Union

import pytest
from inline_snapshot import snapshot

from pydantic_ai_graph import BaseNode, End, EndEvent, Graph, GraphContext, NodeEvent

from .conftest import IsFloat, IsNow

pytestmark = pytest.mark.anyio


async def test_graph():
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
    class Double(BaseNode[None, int]):
        input_data: int

        async def run(self, ctx: GraphContext) -> Union[String2Length, End[int]]:  # noqa: UP007
            if self.input_data == 7:
                return String2Length('x' * 21)
            else:
                return End(self.input_data * 2)

    g = Graph[None, int](nodes=(Float2String, String2Length, Double))
    result, history = await g.run(None, Float2String(3.14))
    # len('3.14') * 2 == 8
    assert result == End(8)
    assert history == snapshot(
        [
            NodeEvent(
                state=None,
                node=Float2String(input_data=3.14),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
            ),
            NodeEvent(
                state=None,
                node=String2Length(input_data='3.14'),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
            ),
            NodeEvent(
                state=None,
                node=Double(input_data=4),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
            ),
            EndEvent(
                state=None,
                result=End(data=8),
                ts=IsNow(tz=timezone.utc),
            ),
        ]
    )
    result, history = await g.run(None, Float2String(3.14159))
    # len('3.14159') == 7, 21 * 2 == 42
    assert result == End(42)
    assert history == snapshot(
        [
            NodeEvent(
                state=None,
                node=Float2String(input_data=3.14159),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
            ),
            NodeEvent(
                state=None,
                node=String2Length(input_data='3.14159'),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
            ),
            NodeEvent(
                state=None,
                node=Double(input_data=7),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
            ),
            NodeEvent(
                state=None,
                node=String2Length(input_data='xxxxxxxxxxxxxxxxxxxxx'),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
            ),
            NodeEvent(
                state=None,
                node=Double(input_data=21),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
            ),
            EndEvent(
                state=None,
                result=End(data=42),
                ts=IsNow(tz=timezone.utc),
            ),
        ]
    )
