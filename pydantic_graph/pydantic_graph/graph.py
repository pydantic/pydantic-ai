from __future__ import annotations as _annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import FrameType
from typing import TYPE_CHECKING, Any, Generic

import logfire_api
from typing_extensions import Literal, Never, ParamSpec, TypeVar, Unpack, assert_never

from . import _utils, exceptions, mermaid
from ._utils import get_parent_namespace
from .nodes import BaseNode, End, GraphContext, NodeDef
from .state import EndEvent, HistoryStep, NodeEvent, StateT

__all__ = ('Graph',)

_logfire = logfire_api.Logfire(otel_scope='pydantic-graph')

RunSignatureT = ParamSpec('RunSignatureT')
RunEndT = TypeVar('RunEndT', default=None)
NodeRunEndT = TypeVar('NodeRunEndT', covariant=True, default=Never)


@dataclass(init=False)
class Graph(Generic[StateT, RunEndT]):
    """Definition of a graph."""

    name: str | None
    node_defs: dict[str, NodeDef[StateT, RunEndT]]

    def __init__(
        self,
        *,
        nodes: Sequence[type[BaseNode[StateT, RunEndT]]],
        state_type: type[StateT] | None = None,
        name: str | None = None,
    ):
        self.name = name

        parent_namespace = get_parent_namespace(inspect.currentframe())
        self.node_defs: dict[str, NodeDef[StateT, RunEndT]] = {}
        for node in nodes:
            self._register_node(node, parent_namespace)

        self._validate_edges()

    def _register_node(self, node: type[BaseNode[StateT, RunEndT]], parent_namespace: dict[str, Any] | None) -> None:
        node_id = node.get_id()
        if existing_node := self.node_defs.get(node_id):
            raise exceptions.GraphSetupError(
                f'Node ID `{node_id}` is not unique — found on {existing_node.node} and {node}'
            )
        else:
            self.node_defs[node_id] = node.get_node_def(parent_namespace)

    def _validate_edges(self):
        known_node_ids = self.node_defs.keys()
        bad_edges: dict[str, list[str]] = {}

        for node_id, node_def in self.node_defs.items():
            for edge in node_def.next_node_edges.keys():
                if edge not in known_node_ids:
                    bad_edges.setdefault(edge, []).append(f'`{node_id}`')

        if bad_edges:
            bad_edges_list = [f'`{k}` is referenced by {_utils.comma_and(v)}' for k, v in bad_edges.items()]
            if len(bad_edges_list) == 1:
                raise exceptions.GraphSetupError(f'{bad_edges_list[0]} but not included in the graph.')
            else:
                b = '\n'.join(f' {be}' for be in bad_edges_list)
                raise exceptions.GraphSetupError(
                    f'Nodes are referenced in the graph but not included in the graph:\n{b}'
                )

    async def next(
        self,
        state: StateT,
        node: BaseNode[StateT, RunEndT],
        history: list[HistoryStep[StateT, RunEndT]],
        *,
        infer_name: bool = True,
    ) -> BaseNode[StateT, Any] | End[RunEndT]:
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())
        node_id = node.get_id()
        if node_id not in self.node_defs:
            raise exceptions.GraphRuntimeError(f'Node `{node}` is not in the graph.')

        history_step: NodeEvent[StateT, RunEndT] | None = NodeEvent(state, node)
        history.append(history_step)

        ctx = GraphContext(state)
        with _logfire.span('run node {node_id}', node_id=node_id, node=node):
            start = perf_counter()
            next_node = await node.run(ctx)
            history_step.duration = perf_counter() - start
        return next_node

    async def run(
        self,
        state: StateT,
        start_node: BaseNode[StateT, RunEndT],
        *,
        infer_name: bool = True,
    ) -> tuple[RunEndT, list[HistoryStep[StateT, RunEndT]]]:
        history: list[HistoryStep[StateT, RunEndT]] = []
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

        with _logfire.span(
            '{graph_name} run {start=}',
            graph_name=self.name or 'graph',
            start=start_node,
        ) as run_span:
            while True:
                next_node = await self.next(state, start_node, history, infer_name=False)
                if isinstance(next_node, End):
                    history.append(EndEvent(state, next_node))
                    run_span.set_attribute('history', history)
                    return next_node.data, history
                elif isinstance(next_node, BaseNode):
                    start_node = next_node
                else:
                    if TYPE_CHECKING:
                        assert_never(next_node)
                    else:
                        raise exceptions.GraphRuntimeError(
                            f'Invalid node return type: `{type(next_node).__name__}`. Expected `BaseNode` or `End`.'
                        )

    def mermaid_code(
        self,
        *,
        start_node: Sequence[mermaid.NodeIdent] | mermaid.NodeIdent | None = None,
        highlighted_nodes: Sequence[mermaid.NodeIdent] | mermaid.NodeIdent | None = None,
        highlight_css: str = mermaid.DEFAULT_HIGHLIGHT_CSS,
        edge_labels: bool = True,
        title: str | None | Literal[False] = None,
        notes: bool = True,
        infer_name: bool = True,
    ) -> str:
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())
        if title is None and self.name:
            title = self.name
        return mermaid.generate_code(
            self,
            start_node=start_node,
            highlighted_nodes=highlighted_nodes,
            highlight_css=highlight_css,
            title=title or None,
            edge_labels=edge_labels,
            notes=notes,
        )

    def mermaid_image(self, infer_name: bool = True, **kwargs: Unpack[mermaid.MermaidConfig]) -> bytes:
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())
        if 'title' not in kwargs and self.name:
            kwargs['title'] = self.name
        return mermaid.request_image(self, **kwargs)

    def mermaid_save(
        self, path: Path | str, /, *, infer_name: bool = True, **kwargs: Unpack[mermaid.MermaidConfig]
    ) -> None:
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())
        if 'title' not in kwargs and self.name:
            kwargs['title'] = self.name
        mermaid.save_image(path, self, **kwargs)

    def _infer_name(self, function_frame: FrameType | None) -> None:
        """Infer the agent name from the call frame.

        Usage should be `self._infer_name(inspect.currentframe())`.

        Copied from `Agent`.
        """
        assert self.name is None, 'Name already set'
        if function_frame is not None and (parent_frame := function_frame.f_back):  # pragma: no branch
            for name, item in parent_frame.f_locals.items():
                if item is self:
                    self.name = name
                    return
            if parent_frame.f_locals != parent_frame.f_globals:
                # if we couldn't find the agent in locals and globals are a different dict, try globals
                for name, item in parent_frame.f_globals.items():
                    if item is self:
                        self.name = name
                        return
