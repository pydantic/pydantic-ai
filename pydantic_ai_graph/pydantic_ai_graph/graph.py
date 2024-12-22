from __future__ import annotations as _annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic

from typing_extensions import TypeVar, assert_never

from . import _utils
from .nodes import (
    BaseNode,
    DepsT,
    End,
    GraphContext,
    GraphOutputT,
    NodeDef,
)
from .state import StateT

__all__ = ('Graph',)
GraphInputT = TypeVar('GraphInputT', default=Any)


# noinspection PyTypeHints
@dataclass(init=False)
class Graph(Generic[GraphInputT, GraphOutputT, DepsT, StateT]):
    """Definition of a graph."""

    first_node: NodeDef[Any, Any, DepsT, StateT]
    nodes: dict[str, NodeDef[Any, Any, DepsT, StateT]]
    state_type: type[StateT] | None

    # noinspection PyUnusedLocal
    def __init__(
        self,
        first_node: type[BaseNode[GraphInputT, Any, DepsT, StateT]],
        *other_nodes: type[BaseNode[Any, GraphOutputT, DepsT, StateT]],
        deps_type: type[DepsT] | None = None,
        state_type: type[StateT] | None = None,
    ):
        self.first_node = first_node.get_node_def()
        self.nodes = nodes = {self.first_node.node_id: self.first_node}
        for node in other_nodes:
            node_def = node.get_node_def()
            nodes[node_def.node_id] = node_def

        self._check()
        self.state_type = state_type

    async def run(
        self, input_data: GraphInputT, deps: DepsT = None, state: StateT = None
    ) -> tuple[GraphOutputT, StateT]:
        current_node_def = self.first_node
        current_node = current_node_def.node(input_data)
        ctx = GraphContext(deps, state)
        while True:
            # noinspection PyUnresolvedReferences
            next_node = await current_node.run(ctx)
            if isinstance(next_node, End):
                if current_node_def.can_end:
                    return next_node.data, ctx.state
                else:
                    raise ValueError(f'Node {current_node_def.node_id} cannot end the graph')
            elif isinstance(next_node, BaseNode):
                next_node_id = next_node.get_id()
                try:
                    next_node_def = self.nodes[next_node_id]
                except KeyError as e:
                    raise ValueError(
                        f'Node {current_node_def.node_id} cannot go to {next_node_id} which is not in the Graph'
                    ) from e

                if not current_node_def.dest_any and next_node_id not in current_node_def.next_node_ids:
                    raise ValueError(
                        f'Node {current_node_def.node_id} cannot go to {next_node_id} which is not in its '
                        f'list of allowed next nodes'
                    )

                current_node_def = next_node_def
                current_node = next_node
            else:
                if TYPE_CHECKING:
                    assert_never(next_node)
                else:
                    raise TypeError(f'Invalid node type: {type(next_node)} expected BaseNode or End')

    def mermaid_code(self) -> str:
        lines = ['graph TD']
        # order of destination nodes should match their order in `self.nodes`
        node_order = {nid: index for index, nid in enumerate(self.nodes.keys())}
        for node_id, node in self.nodes.items():
            if node.dest_any:
                for next_node_id in self.nodes:
                    lines.append(f'  {node_id} --> {next_node_id}')
            for _, next_node_id in sorted((node_order[nid], nid) for nid in node.next_node_ids):
                lines.append(f'  {node_id} --> {next_node_id}')
            if node.can_end:
                lines.append(f'  {node_id} --> END')
        return '\n'.join(lines)

    def mermaid_image(self, mermaid_ink_params: dict[str, str | int] | None = None) -> bytes:
        import httpx

        code_base64 = base64.b64encode(self.mermaid_code().encode()).decode()

        response = httpx.get(f'https://mermaid.ink/img/{code_base64}', params=mermaid_ink_params)
        response.raise_for_status()
        return response.content

    def mermaid_save(self, path: Path, mermaid_ink_params: dict[str, str | int] | None = None) -> None:
        image_data = self.mermaid_image(mermaid_ink_params)
        path.write_bytes(image_data)

    def _check(self):
        bad_edges: dict[str, list[str]] = {}
        for node in self.nodes.values():
            node_bad_edges = node.next_node_ids - self.nodes.keys()
            for bad_edge in node_bad_edges:
                bad_edges.setdefault(bad_edge, []).append(f'"{node.node_id}"')

        if bad_edges:
            bad_edges_list = [f'"{k}" is referenced by {_utils.comma_and(v)}' for k, v in bad_edges.items()]
            if len(bad_edges_list) == 1:
                raise ValueError(f'{bad_edges_list[0]} but not included in the graph.')
            else:
                b = '\n'.join(f' {be}' for be in bad_edges_list)
                raise ValueError(f'Nodes are referenced in the graph but not included in the graph:\n{b}')
