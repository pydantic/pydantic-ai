from __future__ import annotations as _annotations

import base64
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Annotated, Any, Generic

import logfire_api
from annotated_types import Ge, Le
from typing_extensions import Literal, Never, ParamSpec, Protocol, TypeVar, assert_never

from . import _utils
from ._utils import get_parent_namespace
from .nodes import BaseNode, End, GraphContext, NodeDef
from .state import EndEvent, StateT, Step, StepOrEnd

__all__ = ('Graph', 'GraphRun', 'GraphRunner')

_logfire = logfire_api.Logfire(otel_scope='pydantic-ai-graph')

RunSignatureT = ParamSpec('RunSignatureT')
RunEndT = TypeVar('RunEndT', default=None)
NodeRunEndT = TypeVar('NodeRunEndT', covariant=True, default=Never)


class StartNodeProtocol(Protocol[RunSignatureT, StateT, NodeRunEndT]):
    def get_id(self) -> str: ...
    def __call__(self, *args: RunSignatureT.args, **kwargs: RunSignatureT.kwargs) -> BaseNode[StateT, NodeRunEndT]: ...


@dataclass(init=False)
class Graph(Generic[StateT, RunEndT]):
    """Definition of a graph."""

    name: str | None
    nodes: tuple[type[BaseNode[StateT, RunEndT]], ...]
    node_defs: dict[str, NodeDef[StateT, RunEndT]]

    def __init__(
        self,
        name: str | None = None,
        nodes: tuple[type[BaseNode[StateT, RunEndT]], ...] = (),
        state_type: type[StateT] | None = None,
    ):
        self.name = name

        _nodes_by_id: dict[str, type[BaseNode[StateT, RunEndT]]] = {}
        for node in nodes:
            node_id = node.get_id()
            if (existing_node := _nodes_by_id.get(node_id)) and existing_node is not node:
                raise ValueError(f'Node ID "{node_id}" is not unique — found in {existing_node} and {node}')
            else:
                _nodes_by_id[node_id] = node
        self.nodes = tuple(_nodes_by_id.values())

        parent_namespace = get_parent_namespace(inspect.currentframe())
        self.node_defs: dict[str, NodeDef[StateT, RunEndT]] = {}
        for node in self.nodes:
            self.node_defs[node.get_id()] = node.get_node_def(parent_namespace)

        self._validate_edges()

    def _validate_edges(self):
        known_node_ids = set(self.node_defs.keys())
        bad_edges: dict[str, list[str]] = {}

        for node_id, node_def in self.node_defs.items():
            node_bad_edges = node_def.next_node_ids - known_node_ids
            for bad_edge in node_bad_edges:
                bad_edges.setdefault(bad_edge, []).append(f'"{node_id}"')

        if bad_edges:
            bad_edges_list = [f'"{k}" is referenced by {_utils.comma_and(v)}' for k, v in bad_edges.items()]
            if len(bad_edges_list) == 1:
                raise ValueError(f'{bad_edges_list[0]} but not included in the graph.')
            else:
                b = '\n'.join(f' {be}' for be in bad_edges_list)
                raise ValueError(f'Nodes are referenced in the graph but not included in the graph:\n{b}')

    async def run(
        self, state: StateT, node: BaseNode[StateT, RunEndT]
    ) -> tuple[RunEndT, list[StepOrEnd[StateT, RunEndT]]]:
        if not isinstance(node, self.nodes):
            raise ValueError(f'Node "{node}" is not in the graph.')
        run = GraphRun[StateT, RunEndT](state=state)
        # TODO: Infer the graph name properly
        result = await run.run(self.name or 'graph', node)
        history = run.history
        return result, history

    def get_runner(
        self,
        first_node: StartNodeProtocol[RunSignatureT, StateT, RunEndT],
    ) -> GraphRunner[RunSignatureT, StateT, RunEndT]:
        return GraphRunner(
            graph=self,
            first_node=first_node,
        )


@dataclass
class GraphRunner(Generic[RunSignatureT, StateT, RunEndT]):
    """Runner for a graph.

    This is a separate class from Graph so that you can get a type-safe runner from a graph definition
    without needing to manually annotate the paramspec of the start node.
    """

    graph: Graph[StateT, RunEndT]
    first_node: StartNodeProtocol[RunSignatureT, StateT, RunEndT]

    def __post_init__(self):
        if self.first_node not in self.graph.nodes:
            raise ValueError(f'Start node "{self.first_node}" is not in the graph.')

    async def run(
        self, state: StateT, /, *args: RunSignatureT.args, **kwargs: RunSignatureT.kwargs
    ) -> tuple[RunEndT, list[StepOrEnd[StateT, RunEndT]]]:
        run = GraphRun[StateT, RunEndT](state=state)
        # TODO: Infer the graph name properly
        result = await run.run(self.graph.name or 'graph', self.first_node(*args, **kwargs))
        history = run.history
        return result, history

    def mermaid_code(self) -> str:
        return mermaid_code(self.graph, self.first_node)

    def mermaid_image(
        self,
        image_type: Literal['jpeg', 'png', 'webp', 'svg', 'pdf'] | str | None = None,
        pdf_fit: bool = False,
        pdf_landscape: bool = False,
        pdf_paper: Literal['letter', 'legal', 'tabloid', 'ledger', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
        | str
        | None = None,
        bg_color: str | None = None,
        theme: Literal['default', 'neutral', 'dark', 'forest'] | str | None = None,
        width: int | None = None,
        height: int | None = None,
        scale: Annotated[float, Ge(1), Le(3)] | None = None,
    ) -> bytes:
        return mermaid_image(
            self.graph,
            self.first_node,
            image_type=image_type,
            pdf_fit=pdf_fit,
            pdf_landscape=pdf_landscape,
            pdf_paper=pdf_paper,
            bg_color=bg_color,
            theme=theme,
            width=width,
            height=height,
            scale=scale,
        )

    def mermaid_save(
        self,
        path: Path | str,
        image_type: Literal['jpeg', 'png', 'webp', 'svg', 'pdf'] | str | None = None,
        pdf_fit: bool = False,
        pdf_landscape: bool = False,
        pdf_paper: Literal['letter', 'legal', 'tabloid', 'ledger', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
        | str
        | None = None,
        bg_color: str | None = None,
        theme: Literal['default', 'neutral', 'dark', 'forest'] | str | None = None,
        width: int | None = None,
        height: int | None = None,
        scale: Annotated[float, Ge(1), Le(3)] | None = None,
    ) -> None:
        mermaid_save(
            path,
            self.graph,
            self.first_node,
            image_type=image_type,
            pdf_fit=pdf_fit,
            pdf_landscape=pdf_landscape,
            pdf_paper=pdf_paper,
            bg_color=bg_color,
            theme=theme,
            width=width,
            height=height,
            scale=scale,
        )


@dataclass
class GraphRun(Generic[StateT, RunEndT]):
    """Stateful run of a graph."""

    state: StateT
    history: list[StepOrEnd[StateT, RunEndT]] = field(default_factory=list)

    async def run(self, graph_name: str, start: BaseNode[StateT, RunEndT], infer_name: bool = True) -> RunEndT:
        current_node = start

        with _logfire.span(
            '{graph_name} run {start=}',
            graph_name=graph_name,
            start=start,
        ) as run_span:
            while True:
                next_node = await self.step(current_node)
                if isinstance(next_node, End):
                    self.history.append(EndEvent(self.state, next_node))
                    run_span.set_attribute('history', self.history)
                    return next_node.data
                elif isinstance(next_node, BaseNode):
                    current_node = next_node
                else:
                    if TYPE_CHECKING:
                        assert_never(next_node)
                    else:
                        raise TypeError(f'Invalid node type: {type(next_node)}. Expected `BaseNode` or `End`.')

    async def step(self, node: BaseNode[StateT, RunEndT]) -> BaseNode[StateT, RunEndT] | End[RunEndT]:
        history_step = Step(self.state, node)
        self.history.append(history_step)

        ctx = GraphContext(self.state)
        with _logfire.span('run node {node_id}', node_id=node.get_id()):
            start = perf_counter()
            next_node = await node.run(ctx)
            history_step.duration = perf_counter() - start
        return next_node


def mermaid_code(
    graph: Graph[Any, Any], start: StartNodeProtocol[..., Any, Any] | tuple[StartNodeProtocol[..., Any, Any], ...] = ()
) -> str:
    if not isinstance(start, tuple):
        start = (start,)

    for node in start:
        if node not in graph.nodes:
            raise ValueError(f'Start node "{node}" is not in the graph.')

    node_order = {node_id: index for index, node_id in enumerate(graph.node_defs)}

    lines = ['graph TD']
    for node in graph.nodes:
        node_id = node.get_id()
        node_def = graph.node_defs[node_id]
        if node in start:
            lines.append(f'  START --> {node_id}')
        if node_def.returns_base_node:
            for next_node_id in graph.nodes:
                lines.append(f'  {node_id} --> {next_node_id}')
        else:
            for _, next_node_id in sorted((node_order[node_id], node_id) for node_id in node_def.next_node_ids):
                lines.append(f'  {node_id} --> {next_node_id}')
        if node_def.returns_end:
            lines.append(f'  {node_id} --> END')

    return '\n'.join(lines)


def mermaid_image(
    graph: Graph[Any, Any],
    start: StartNodeProtocol[..., Any, Any] | tuple[StartNodeProtocol[..., Any, Any], ...] = (),
    image_type: Literal['jpeg', 'png', 'webp', 'svg', 'pdf'] | str | None = None,
    pdf_fit: bool = False,
    pdf_landscape: bool = False,
    pdf_paper: Literal['letter', 'legal', 'tabloid', 'ledger', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    | str
    | None = None,
    bg_color: str | None = None,
    theme: Literal['default', 'neutral', 'dark', 'forest'] | str | None = None,
    width: int | None = None,
    height: int | None = None,
    scale: Annotated[float, Ge(1), Le(3)] | None = None,
) -> bytes:
    """Generate an image of a Mermaid diagram using mermaid.ink.

    Args:
        graph: The graph to generate the image for.
        start: The start node(s) of the graph.
        image_type: The image type to generate. If unspecified, the default behavior is `'jpeg'`.
        pdf_fit: When using image_type='pdf', whether to fit the diagram to the PDF page.
        pdf_landscape: When using image_type='pdf', whether to use landscape orientation for the PDF.
            This has no effect if using `pdf_fit`.
        pdf_paper: When using image_type='pdf', the paper size of the PDF.
        bg_color: The background color of the diagram. If None, the default transparent background is used.
            The color value is interpreted as a hexadecimal color code by default (and should not have a leading '#'),
            but you can also use named colors by prefixing the value with '!'.
            For example, valid choices include `bg_color='!white'` or `bg_color='FF0000'`.
        theme: The theme of the diagram. Defaults to 'default'.
        width: The width of the diagram.
        height: The height of the diagram.
        scale: The scale of the diagram. The scale must be a number between 1 and 3, and you can only set
            a scale if one or both of width and height are set.
    """
    import httpx

    code_base64 = base64.b64encode(mermaid_code(graph, start).encode()).decode()

    params: dict[str, str] = {}
    if image_type == 'pdf':
        url = f'https://mermaid.ink/pdf/{code_base64}'
        if pdf_fit:
            params['fit'] = ''
        if pdf_landscape:
            params['landscape'] = ''
        if pdf_paper:
            params['paper'] = pdf_paper
    else:
        url = f'https://mermaid.ink/img/{code_base64}'

        if image_type:
            params['type'] = image_type

    if bg_color:
        params['bgColor'] = bg_color
    if theme:
        params['theme'] = theme
    if width:
        params['width'] = str(width)
    if height:
        params['height'] = str(height)
    if scale:
        params['scale'] = str(scale)

    response = httpx.get(url, params=params)
    response.raise_for_status()
    return response.content


def mermaid_save(
    path: Path | str,
    graph: Graph[Any, Any],
    start: StartNodeProtocol[..., Any, Any] | tuple[StartNodeProtocol[..., Any, Any], ...] = (),
    image_type: Literal['jpeg', 'png', 'webp', 'svg', 'pdf'] | str | None = None,
    pdf_fit: bool = False,
    pdf_landscape: bool = False,
    pdf_paper: Literal['letter', 'legal', 'tabloid', 'ledger', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    | str
    | None = None,
    bg_color: str | None = None,
    theme: Literal['default', 'neutral', 'dark', 'forest'] | str | None = None,
    width: int | None = None,
    height: int | None = None,
    scale: Annotated[float, Ge(1), Le(3)] | None = None,
) -> None:
    # TODO: do something with the path file extension, e.g. error if it's incompatible, or use it to specify a param
    image_data = mermaid_image(
        graph,
        start,
        image_type=image_type,
        pdf_fit=pdf_fit,
        pdf_landscape=pdf_landscape,
        pdf_paper=pdf_paper,
        bg_color=bg_color,
        theme=theme,
        width=width,
        height=height,
        scale=scale,
    )
    Path(path).write_bytes(image_data)
