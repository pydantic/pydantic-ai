from __future__ import annotations as _annotations

from dataclasses import dataclass
from datetime import timezone
from typing import Annotated

import pytest
from inline_snapshot import snapshot

from pydantic_graph import BaseNode, Edge, End, EndEvent, Graph, GraphContext, NodeEvent

from .conftest import IsFloat, IsNow

pytestmark = pytest.mark.anyio


@dataclass
class Foo(BaseNode):
    async def run(self, ctx: GraphContext) -> Bar:
        return Bar()


@dataclass
class Bar(BaseNode[None, None]):
    async def run(self, ctx: GraphContext) -> End[None]:
        return End(None)


graph1 = Graph(nodes=(Foo, Bar))


@dataclass
class Spam(BaseNode):
    """This is the docstring for Spam."""

    async def run(self, ctx: GraphContext) -> Annotated[Foo, Edge(label='spam to foo')]:
        return Foo()


@dataclass
class Eggs(BaseNode[None, None]):
    async def run(self, ctx: GraphContext) -> Annotated[End[None], Edge(label='eggs to end')]:
        return End(None)


graph2 = Graph(nodes=(Spam, Foo, Bar, Eggs))


async def test_run_graph():
    result, history = await graph1.run(None, Foo())
    assert result is None
    assert history == snapshot(
        [
            NodeEvent(
                state=None,
                node=Foo(),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
            ),
            NodeEvent(
                state=None,
                node=Bar(),
                start_ts=IsNow(tz=timezone.utc),
                duration=IsFloat(),
            ),
            EndEvent(state=None, result=End(data=None), ts=IsNow(tz=timezone.utc)),
        ]
    )


def test_mermaid_code_no_start():
    assert graph1.mermaid_code() == snapshot("""\
stateDiagram-v2
  Foo --> Bar
  Bar --> [*]\
""")


def test_mermaid_code_start():
    assert graph1.mermaid_code(start_node=Foo) == snapshot("""\
stateDiagram-v2
  [*] --> Foo
  Foo --> Bar
  Bar --> [*]\
""")


def test_mermaid_code_start_wrong():
    with pytest.raises(LookupError):
        graph1.mermaid_code(start_node=Spam)


def test_mermaid_highlight():
    code = graph1.mermaid_code(highlighted_nodes=Foo)
    assert code == snapshot("""\
stateDiagram-v2
  Foo --> Bar
  Bar --> [*]

classDef highlighted fill:#fdff32
class Foo highlighted\
""")
    assert code == graph1.mermaid_code(highlighted_nodes='Foo')


def test_mermaid_highlight_multiple():
    code = graph1.mermaid_code(highlighted_nodes=(Foo, Bar))
    assert code == snapshot("""\
stateDiagram-v2
  Foo --> Bar
  Bar --> [*]

classDef highlighted fill:#fdff32
class Foo highlighted
class Bar highlighted\
""")
    assert code == graph1.mermaid_code(highlighted_nodes=('Foo', 'Bar'))


def test_mermaid_highlight_wrong():
    with pytest.raises(LookupError):
        graph1.mermaid_code(highlighted_nodes=Spam)


def test_mermaid_code_with_edge_labels():
    assert graph2.mermaid_code() == snapshot("""\
stateDiagram-v2
  Spam --> Foo: spam to foo
  note right of Spam
    This is the docstring for Spam.
  end note
  Foo --> Bar
  Bar --> [*]
  Eggs --> [*]: eggs to end\
""")


def test_mermaid_code_without_edge_labels():
    assert graph2.mermaid_code(edge_labels=False, notes=False) == snapshot("""\
stateDiagram-v2
  Spam --> Foo
  Foo --> Bar
  Bar --> [*]
  Eggs --> [*]\
""")


@dataclass
class AllNodes(BaseNode):
    async def run(self, ctx: GraphContext) -> BaseNode:
        return Foo()


graph3 = Graph(nodes=(AllNodes, Foo, Bar))


def test_mermaid_code_all_nodes():
    assert graph3.mermaid_code() == snapshot("""\
stateDiagram-v2
  AllNodes --> AllNodes
  AllNodes --> Foo
  AllNodes --> Bar
  Foo --> Bar
  Bar --> [*]\
""")
