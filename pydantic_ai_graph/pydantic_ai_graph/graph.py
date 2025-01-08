from __future__ import annotations as _annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Generic

import logfire_api
from typing_extensions import Never, ParamSpec, TypeVar, Unpack, assert_never

from . import _utils, mermaid
from ._utils import get_parent_namespace
from .nodes import BaseNode, End, GraphContext, NodeDef
from .state import EndEvent, StateT, Step, StepOrEnd

__all__ = 'Graph', 'GraphRun'

_logfire = logfire_api.Logfire(otel_scope='pydantic-ai-graph')

RunSignatureT = ParamSpec('RunSignatureT')
RunEndT = TypeVar('RunEndT', default=None)
NodeRunEndT = TypeVar('NodeRunEndT', covariant=True, default=Never)


@dataclass(init=False)
class Graph(Generic[StateT, RunEndT]):
    """Definition of a graph."""

    name: str | None
    nodes: tuple[type[BaseNode[StateT, RunEndT]], ...]
    node_defs: dict[str, NodeDef[StateT, RunEndT]]

    def __init__(
        self,
        *,
        nodes: Sequence[type[BaseNode[StateT, RunEndT]]],
        state_type: type[StateT] | None = None,
        name: str | None = None,
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

    def mermaid_code(
        self,
        start_nodes: Sequence[mermaid.NodeIdent] | mermaid.NodeIdent,
        *,
        highlighted_nodes: Sequence[mermaid.NodeIdent] | mermaid.NodeIdent | None = None,
        highlight_css: str = mermaid.DEFAULT_HIGHLIGHT_CSS,
    ) -> str:
        return mermaid.generate_code(
            self, start_nodes, highlighted_nodes=highlighted_nodes, highlight_css=highlight_css
        )

    def mermaid_image(
        self, start_nodes: Sequence[mermaid.NodeIdent] | mermaid.NodeIdent, **kwargs: Unpack[mermaid.MermaidConfig]
    ) -> bytes:
        return mermaid.request_image(self, start_nodes, **kwargs)

    def mermaid_save(
        self,
        start_nodes: Sequence[mermaid.NodeIdent] | mermaid.NodeIdent,
        path: Path | str,
        **kwargs: Unpack[mermaid.MermaidConfig],
    ) -> None:
        mermaid.save_image(path, self, start_nodes, **kwargs)


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
