from __future__ import annotations as _annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import FrameType
from typing import TYPE_CHECKING, Any, Callable, Generic

import logfire_api
from typing_extensions import Literal, Never, ParamSpec, TypeVar, Unpack, assert_never

from . import _utils, exceptions, mermaid
from ._utils import get_parent_namespace
from .nodes import BaseNode, End, GraphContext, NodeDef
from .state import EndStep, HistoryStep, NodeStep, StateT, deep_copy_state

__all__ = ('Graph',)

_logfire = logfire_api.Logfire(otel_scope='pydantic-graph')

RunSignatureT = ParamSpec('RunSignatureT')
RunEndT = TypeVar('RunEndT', default=None)
NodeRunEndT = TypeVar('NodeRunEndT', covariant=True, default=Never)


@dataclass(init=False)
class Graph(Generic[StateT, RunEndT]):
    """Definition of a graph.

    In `pydantic-graph`, a graph is a collection of nodes that can be run in sequence. The nodes define
    their outgoing edges — e.g. which nodes may be run next, and thereby the structure of the graph.

    Here's a very simple example of a graph which increments a number by 1, but makes sure the number is never
    42 at the end. Note in this example we don't run the graph, but instead just generate a mermaid diagram
    using [`mermaid_code`][pydantic_graph.graph.Graph.mermaid_code]:

    ```py {title="never_42.py"}
    from __future__ import annotations

    from dataclasses import dataclass
    from pydantic_graph import BaseNode, End, Graph, GraphContext


    @dataclass
    class MyState:
        number: int


    @dataclass
    class Increment(BaseNode[MyState]):
        async def run(self, ctx: GraphContext) -> Check42:
            ctx.state.number += 1
            return Check42()


    @dataclass
    class Check42(BaseNode[MyState]):
        async def run(self, ctx: GraphContext) -> Increment | End:
            if ctx.state.number == 42:
                return Increment()
            else:
                return End(None)


    never_42_graph = Graph(nodes=(Increment, Check42))
    print(never_42_graph.mermaid_code(start_node=Increment))
    ```
    _(This example is complete, it can be run "as is")_

    The rendered mermaid diagram will look like this:

    ```mermaid
    ---
    title: never_42_graph
    ---
    stateDiagram-v2
      [*] --> Increment
      Increment --> Check42
      Check42 --> Increment
      Check42 --> [*]
    ```

    See [`run`][pydantic_graph.graph.Graph.run] For an example of running graph.
    """

    name: str | None
    node_defs: dict[str, NodeDef[StateT, RunEndT]]
    snapshot_state: Callable[[StateT], StateT]

    def __init__(
        self,
        *,
        nodes: Sequence[type[BaseNode[StateT, RunEndT]]],
        name: str | None = None,
        snapshot_state: Callable[[StateT], StateT] = deep_copy_state,
    ):
        """Create a graph from a sequence of nodes.

        Args:
            nodes: The nodes which make up the graph, nodes need to be unique and all be generic in the same
                state type.
            name: Optional name for the graph, if not provided the name will be inferred from the calling frame
                on the first call to a graph method.
            snapshot_state: A function to snapshot the state of the graph, this is used in
                [`NodeStep`][pydantic_graph.state.NodeStep] and [`EndStep`][pydantic_graph.state.EndStep] to record
                the state before each step.
        """
        self.name = name
        self.snapshot_state = snapshot_state

        parent_namespace = get_parent_namespace(inspect.currentframe())
        self.node_defs: dict[str, NodeDef[StateT, RunEndT]] = {}
        for node in nodes:
            self._register_node(node, parent_namespace)

        self._validate_edges()

    async def next(
        self,
        state: StateT,
        node: BaseNode[StateT, RunEndT],
        history: list[HistoryStep[StateT, RunEndT]],
        *,
        infer_name: bool = True,
    ) -> BaseNode[StateT, Any] | End[RunEndT]:
        """Run a node in the graph and return the next node to run.

        Args:
            state: The current state of the graph.
            node: The node to run.
            history: The history of the graph run so far. NOTE: this will be mutated to add the new step.
            infer_name: Whether to infer the graph name from the calling frame.

        Returns:
            The next node to run or [`End`][pydantic_graph.nodes.End] if the graph has finished.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())
        node_id = node.get_id()
        if node_id not in self.node_defs:
            raise exceptions.GraphRuntimeError(f'Node `{node}` is not in the graph.')

        history_step: NodeStep[StateT, RunEndT] = NodeStep(state, node)
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
        """Run the graph from a starting node until it ends.

        Args:
            state: The initial state of the graph.
            start_node: the first node to run, since the graph definition doesn't define the entry point in the graph,
                you need to provide the starting node.
            infer_name: Whether to infer the graph name from the calling frame.

        Returns: The result type from ending the run and the history of the run.

        Here's an example of running the graph from [above][pydantic_graph.graph.Graph]:

        ```py {title="run_never_42.py"}
        from never_42 import MyState, Increment, never_42_graph

        async def main():
            state = MyState(1)
            _, history = await never_42_graph.run(state, Increment())
            print(state)
            print(history)

            state = MyState(41)
            _, history = await never_42_graph.run(state, Increment())
            print(state)
            print(history)
        ```
        """
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
                    history.append(EndStep(state, next_node, snapshot_state=self.snapshot_state))
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
        title: str | None | Literal[False] = None,
        edge_labels: bool = True,
        notes: bool = True,
        highlighted_nodes: Sequence[mermaid.NodeIdent] | mermaid.NodeIdent | None = None,
        highlight_css: str = mermaid.DEFAULT_HIGHLIGHT_CSS,
        infer_name: bool = True,
    ) -> str:
        """Generate a diagram representing the graph as [mermaid](https://mermaid.js.org/) chart.

        This method calls [`pydantic_graph.mermaid.generate_code`][pydantic_graph.mermaid.generate_code].

        Args:
            start_node: The node or nodes to start the graph from.
            title: The title of the diagram, use `False` to not include a title.
            edge_labels: Whether to include edge labels.
            notes: Whether to include notes on each node.
            highlighted_nodes: Optional node or nodes to highlight.
            highlight_css: The CSS to use for highlighting nodes.
            infer_name: Whether to infer the graph name from the calling frame.

        Returns:
            The mermaid code for the graph, which can then be rendered as a diagram.
        """
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
        """Generate a diagram representing the graph as an image.

        The format and diagram can be customized using `kwargs`,
        see [`pydantic_graph.mermaid.MermaidConfig`][pydantic_graph.mermaid.MermaidConfig].

        !!! note "Uses external service"
            This method makes a request to [mermaid.ink](https://mermaid.ink) to render the image, `mermaid.ink`
            is a free service not affiliated with Pydantic.

        Args:
            infer_name: Whether to infer the graph name from the calling frame.
            **kwargs: Additional arguments to pass to `mermaid.request_image`.

        Returns:
            The image bytes.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())
        if 'title' not in kwargs and self.name:
            kwargs['title'] = self.name
        return mermaid.request_image(self, **kwargs)

    def mermaid_save(
        self, path: Path | str, /, *, infer_name: bool = True, **kwargs: Unpack[mermaid.MermaidConfig]
    ) -> None:
        """Generate a diagram representing the graph and save it as an image.

        The format and diagram can be customized using `kwargs`,
        see [`pydantic_graph.mermaid.MermaidConfig`][pydantic_graph.mermaid.MermaidConfig].

        !!! note "Uses external service"
            This method makes a request to [mermaid.ink](https://mermaid.ink) to render the image, `mermaid.ink`
            is a free service not affiliated with Pydantic.

        Args:
            path: The path to save the image to.
            infer_name: Whether to infer the graph name from the calling frame.
            **kwargs: Additional arguments to pass to `mermaid.save_image`.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())
        if 'title' not in kwargs and self.name:
            kwargs['title'] = self.name
        mermaid.save_image(path, self, **kwargs)

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
