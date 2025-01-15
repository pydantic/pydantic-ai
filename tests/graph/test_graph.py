from __future__ import annotations as _annotations

from dataclasses import dataclass
from datetime import timezone
from functools import cache
from typing import Union

import pytest
from dirty_equals import IsStr
from inline_snapshot import snapshot
from typing_extensions import Never

from pydantic_graph import (
    BaseNode,
    End,
    EndStep,
    Graph,
    GraphContext,
    GraphRuntimeError,
    GraphSetupError,
    HistoryStep,
    NodeStep,
)

from ..conftest import IsFloat, IsNow

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

    my_graph = Graph[None, int](nodes=(Float2String, String2Length, Double))
    assert my_graph.name is None
    result, history = await my_graph.run(None, Float2String(3.14))
    # len('3.14') * 2 == 8
    assert result == 8
    assert my_graph.name == 'my_graph'
    assert history == snapshot(
        [
            NodeStep(
                state=None,
                node=Float2String(input_data=3.14),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
            ),
            NodeStep(
                state=None,
                node=String2Length(input_data='3.14'),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
            ),
            NodeStep(
                state=None,
                node=Double(input_data=4),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
            ),
            EndStep(
                state=None,
                result=End(data=8),
                ts=IsNow(tz=timezone.utc),
            ),
        ]
    )
    result, history = await my_graph.run(None, Float2String(3.14159))
    # len('3.14159') == 7, 21 * 2 == 42
    assert result == 42
    assert history == snapshot(
        [
            NodeStep(
                state=None,
                node=Float2String(input_data=3.14159),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
            ),
            NodeStep(
                state=None,
                node=String2Length(input_data='3.14159'),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
            ),
            NodeStep(
                state=None,
                node=Double(input_data=7),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
            ),
            NodeStep(
                state=None,
                node=String2Length(input_data='xxxxxxxxxxxxxxxxxxxxx'),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
            ),
            NodeStep(
                state=None,
                node=Double(input_data=21),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
            ),
            EndStep(
                state=None,
                result=End(data=42),
                ts=IsNow(tz=timezone.utc),
            ),
        ]
    )
    assert [e.data_snapshot() for e in history] == snapshot(
        [
            Float2String(input_data=3.14159),
            String2Length(input_data='3.14159'),
            Double(input_data=7),
            String2Length(input_data='xxxxxxxxxxxxxxxxxxxxx'),
            Double(input_data=21),
            End(data=42),
        ]
    )


def test_one_bad_node():
    class Float2String(BaseNode):
        async def run(self, ctx: GraphContext) -> String2Length:
            raise NotImplementedError()

    class String2Length(BaseNode[None, None]):
        async def run(self, ctx: GraphContext) -> End[None]:
            raise NotImplementedError()

    with pytest.raises(GraphSetupError) as exc_info:
        Graph(nodes=(Float2String,))

    assert exc_info.value.message == snapshot(
        '`String2Length` is referenced by `Float2String` but not included in the graph.'
    )


def test_two_bad_nodes():
    class Foo(BaseNode):
        input_data: float

        async def run(self, ctx: GraphContext) -> Union[Bar, Spam]:  # noqa: UP007
            raise NotImplementedError()

    class Bar(BaseNode[None, None]):
        input_data: str

        async def run(self, ctx: GraphContext) -> End[None]:
            raise NotImplementedError()

    class Spam(BaseNode[None, None]):
        async def run(self, ctx: GraphContext) -> End[None]:
            raise NotImplementedError()

    with pytest.raises(GraphSetupError) as exc_info:
        Graph(nodes=(Foo,))

    assert exc_info.value.message == snapshot("""\
Nodes are referenced in the graph but not included in the graph:
 `Bar` is referenced by `Foo`
 `Spam` is referenced by `Foo`\
""")


def test_three_bad_nodes_separate():
    class Foo(BaseNode):
        input_data: float

        async def run(self, ctx: GraphContext) -> Eggs:
            raise NotImplementedError()

    class Bar(BaseNode[None, None]):
        input_data: str

        async def run(self, ctx: GraphContext) -> Eggs:
            raise NotImplementedError()

    class Spam(BaseNode[None, None]):
        async def run(self, ctx: GraphContext) -> Eggs:
            raise NotImplementedError()

    class Eggs(BaseNode[None, None]):
        async def run(self, ctx: GraphContext) -> End[None]:
            raise NotImplementedError()

    with pytest.raises(GraphSetupError) as exc_info:
        Graph(nodes=(Foo, Bar, Spam))

    assert exc_info.value.message == snapshot(
        '`Eggs` is referenced by `Foo`, `Bar`, and `Spam` but not included in the graph.'
    )


def test_duplicate_id():
    class Foo(BaseNode):
        async def run(self, ctx: GraphContext) -> Bar:
            raise NotImplementedError()

    class Bar(BaseNode[None, None]):
        async def run(self, ctx: GraphContext) -> End[None]:
            raise NotImplementedError()

        @classmethod
        @cache
        def get_id(cls) -> str:
            return 'Foo'

    with pytest.raises(GraphSetupError) as exc_info:
        Graph(nodes=(Foo, Bar))

    assert exc_info.value.message == snapshot(IsStr(regex='Node ID `Foo` is not unique — found on <class.+'))


async def test_run_node_not_in_graph():
    @dataclass
    class Foo(BaseNode):
        async def run(self, ctx: GraphContext) -> Bar:
            return Bar()

    @dataclass
    class Bar(BaseNode[None, None]):
        async def run(self, ctx: GraphContext) -> End[None]:
            return Spam()  # type: ignore

    @dataclass
    class Spam(BaseNode[None, None]):
        async def run(self, ctx: GraphContext) -> End[None]:
            raise NotImplementedError()

    g = Graph(nodes=(Foo, Bar))
    with pytest.raises(GraphRuntimeError) as exc_info:
        await g.run(None, Foo())

    assert exc_info.value.message == snapshot('Node `test_run_node_not_in_graph.<locals>.Spam()` is not in the graph.')


async def test_run_return_other():
    @dataclass
    class Foo(BaseNode):
        async def run(self, ctx: GraphContext) -> Bar:
            return Bar()

    @dataclass
    class Bar(BaseNode[None, None]):
        async def run(self, ctx: GraphContext) -> End[None]:
            return 42  # type: ignore

    g = Graph(nodes=(Foo, Bar))
    with pytest.raises(GraphRuntimeError) as exc_info:
        await g.run(None, Foo())

    assert exc_info.value.message == snapshot('Invalid node return type: `int`. Expected `BaseNode` or `End`.')


async def test_next():
    @dataclass
    class Foo(BaseNode):
        async def run(self, ctx: GraphContext) -> Bar:
            return Bar()

    @dataclass
    class Bar(BaseNode):
        async def run(self, ctx: GraphContext) -> Foo:
            return Foo()

    g = Graph(nodes=(Foo, Bar))
    assert g.name is None
    history: list[HistoryStep[None, Never]] = []
    n = await g.next(None, Foo(), history)
    assert n == Bar()
    assert g.name == 'g'
    assert history == snapshot([NodeStep(state=None, node=Foo(), start_ts=IsNow(tz=timezone.utc), duration=IsFloat())])

    assert isinstance(n, Bar)
    n2 = await g.next(None, n, history)
    assert n2 == Foo()

    assert history == snapshot(
        [
            NodeStep(state=None, node=Foo(), start_ts=IsNow(tz=timezone.utc), duration=IsFloat()),
            NodeStep(state=None, node=Bar(), start_ts=IsNow(tz=timezone.utc), duration=IsFloat()),
        ]
    )