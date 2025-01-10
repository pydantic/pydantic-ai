from __future__ import annotations as _annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Annotated, Callable

import httpx
import pytest
from inline_snapshot import snapshot
from typing_extensions import TypeAlias

from pydantic_graph import BaseNode, Edge, End, EndEvent, Graph, GraphContext, GraphSetupError, NodeEvent
from pydantic_graph.nodes import NodeDef

from ..conftest import IsFloat, IsNow

pytestmark = pytest.mark.anyio
HttpxWithHandler: TypeAlias = 'Callable[[Callable[[httpx.Request], httpx.Response] | httpx.Response], httpx.Client]'


@pytest.fixture
def httpx_with_handler() -> Iterator[HttpxWithHandler]:
    client: httpx.Client | None = None

    def create_client(handler: Callable[[httpx.Request], httpx.Response] | httpx.Response) -> httpx.Client:
        nonlocal client
        assert client is None, 'client_with_handler can only be called once'
        if isinstance(handler, httpx.Response):
            transport = httpx.MockTransport(lambda _: handler)
        else:
            transport = httpx.MockTransport(handler)
        client = httpx.Client(mounts={'all://': transport})
        return client

    try:
        yield create_client
    finally:
        if client:  # pragma: no cover
            client.close()


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
        raise NotImplementedError()


@dataclass
class Eggs(BaseNode[None, None]):
    """This is the docstring for Eggs."""

    enable_docstring_notes = False

    async def run(self, ctx: GraphContext) -> Annotated[End[None], Edge(label='eggs to end')]:
        raise NotImplementedError()


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
        raise NotImplementedError()


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


def test_image_jpg(httpx_with_handler: HttpxWithHandler):
    def get_jpg(request: httpx.Request) -> httpx.Response:
        assert dict(request.url.params) == snapshot({})
        assert request.url.path.startswith('/img/')
        return httpx.Response(200, content=b'fake jpg')

    img = graph1.mermaid_image(start_node=Foo(), httpx_client=httpx_with_handler(get_jpg))
    assert img == b'fake jpg'


def test_image_png(httpx_with_handler: HttpxWithHandler):
    def get_png(request: httpx.Request) -> httpx.Response:
        assert dict(request.url.params) == snapshot(
            {
                'type': 'png',
                'bgColor': '123',
                'theme': 'forest',
                'width': '100',
                'height': '200',
                'scale': '3',
            }
        )
        assert request.url.path.startswith('/img/')
        return httpx.Response(200, content=b'fake png')

    img = graph1.mermaid_image(
        start_node=Foo(),
        image_type='png',
        background_color='123',
        theme='forest',
        width=100,
        height=200,
        scale=3,
        httpx_client=httpx_with_handler(get_png),
    )
    assert img == b'fake png'


def test_image_bad(httpx_with_handler: HttpxWithHandler):
    response = httpx.Response(404, content=b'not found')
    with pytest.raises(httpx.HTTPStatusError, match='404 error generating image:\nnot found') as exc_info:
        graph1.mermaid_image(start_node=Foo(), httpx_client=httpx_with_handler(response))
    assert exc_info.value.response.status_code == 404
    assert exc_info.value.response.content == b'not found'


def test_pdf(httpx_with_handler: HttpxWithHandler):
    def get_pdf(request: httpx.Request) -> httpx.Response:
        assert dict(request.url.params) == snapshot({})
        assert request.url.path.startswith('/pdf/')
        return httpx.Response(200, content=b'fake pdf')

    pdf = graph1.mermaid_image(start_node=Foo(), image_type='pdf', httpx_client=httpx_with_handler(get_pdf))
    assert pdf == b'fake pdf'


def test_pdf_config(httpx_with_handler: HttpxWithHandler):
    def get_pdf(request: httpx.Request) -> httpx.Response:
        assert dict(request.url.params) == snapshot({'fit': '', 'landscape': '', 'paper': 'letter'})
        assert request.url.path.startswith('/pdf/')
        return httpx.Response(200, content=b'fake pdf')

    pdf = graph1.mermaid_image(
        start_node=Foo(),
        image_type='pdf',
        pdf_fit=True,
        pdf_landscape=True,
        pdf_paper='letter',
        httpx_client=httpx_with_handler(get_pdf),
    )
    assert pdf == b'fake pdf'


def test_svg(httpx_with_handler: HttpxWithHandler):
    def get_svg(request: httpx.Request) -> httpx.Response:
        assert dict(request.url.params) == snapshot({})
        assert request.url.path.startswith('/svg/')
        return httpx.Response(200, content=b'fake svg')

    svg = graph1.mermaid_image(start_node=Foo(), image_type='svg', httpx_client=httpx_with_handler(get_svg))
    assert svg == b'fake svg'


def test_save_jpg(tmp_path: Path, httpx_with_handler: HttpxWithHandler):
    def get_jpg(request: httpx.Request) -> httpx.Response:
        assert dict(request.url.params) == snapshot({})
        assert request.url.path.startswith('/img/')
        return httpx.Response(200, content=b'fake jpg')

    path = tmp_path / 'graph.jpg'
    graph1.mermaid_save(path, start_node=Foo(), httpx_client=httpx_with_handler(get_jpg))
    assert path.read_bytes() == b'fake jpg'


def test_save_png(tmp_path: Path, httpx_with_handler: HttpxWithHandler):
    def get_png(request: httpx.Request) -> httpx.Response:
        assert dict(request.url.params) == snapshot({'type': 'png'})
        assert request.url.path.startswith('/img/')
        return httpx.Response(200, content=b'fake png')

    path2 = tmp_path / 'graph.png'
    graph1.mermaid_save(str(path2), start_node=Foo(), httpx_client=httpx_with_handler(get_png))
    assert path2.read_bytes() == b'fake png'


def test_save_pdf_known(tmp_path: Path, httpx_with_handler: HttpxWithHandler):
    def get_pdf(request: httpx.Request) -> httpx.Response:
        assert dict(request.url.params) == snapshot({})
        assert request.url.path.startswith('/pdf/')
        return httpx.Response(200, content=b'fake pdf')

    path2 = tmp_path / 'graph'
    graph1.mermaid_save(str(path2), start_node=Foo(), image_type='pdf', httpx_client=httpx_with_handler(get_pdf))
    assert path2.read_bytes() == b'fake pdf'


def test_get_node_def():
    assert Foo.get_node_def({}) == snapshot(
        NodeDef(
            node=Foo,
            node_id='Foo',
            note=None,
            next_node_edges={'Bar': Edge(label=None)},
            end_edge=None,
            returns_base_node=False,
        )
    )


def test_no_return_type():
    @dataclass
    class NoReturnType(BaseNode):
        async def run(self, ctx: GraphContext):  # type: ignore
            raise NotImplementedError()

    with pytest.raises(GraphSetupError, match=r".*\.NoReturnType'> is missing a return type hint on its `run` method"):
        NoReturnType.get_node_def({})


def test_wrong_return_type():
    @dataclass
    class NoReturnType(BaseNode):
        async def run(self, ctx: GraphContext) -> int:  # type: ignore
            raise NotImplementedError()

    with pytest.raises(GraphSetupError, match="Invalid return type: <class 'int'>"):
        NoReturnType.get_node_def({})
