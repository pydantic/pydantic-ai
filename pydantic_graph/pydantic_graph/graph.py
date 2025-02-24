from __future__ import annotations as _annotations

import asyncio
import inspect
import types
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import logfire_api
import typing_extensions

from . import _utils, exceptions, mermaid
from .nodes import BaseNode, DepsT, End, GraphRunContext, NodeDef, RunEndT
from .state import StatePersistence, StateT, set_nodes_type_context
from .state.memory import SimpleStatePersistence

# while waiting for https://github.com/pydantic/logfire/issues/745
try:
    import logfire._internal.stack_info
except ImportError:
    pass
else:
    from pathlib import Path

    logfire._internal.stack_info.NON_USER_CODE_PREFIXES += (str(Path(__file__).parent.absolute()),)


__all__ = ('Graph',)

_logfire = logfire_api.Logfire(otel_scope='pydantic-graph')

T = TypeVar('T')
"""An invariant typevar."""


@dataclass(init=False)
class Graph(Generic[StateT, DepsT, RunEndT]):
    """Definition of a graph.

    In `pydantic-graph`, a graph is a collection of nodes that can be run in sequence. The nodes define
    their outgoing edges — e.g. which nodes may be run next, and thereby the structure of the graph.

    Here's a very simple example of a graph which increments a number by 1, but makes sure the number is never
    42 at the end.

    ```py {title="never_42.py" noqa="I001" py="3.10"}
    from __future__ import annotations

    from dataclasses import dataclass

    from pydantic_graph import BaseNode, End, Graph, GraphRunContext

    @dataclass
    class MyState:
        number: int

    @dataclass
    class Increment(BaseNode[MyState]):
        async def run(self, ctx: GraphRunContext) -> Check42:
            ctx.state.number += 1
            return Check42()

    @dataclass
    class Check42(BaseNode[MyState, None, int]):
        async def run(self, ctx: GraphRunContext) -> Increment | End[int]:
            if ctx.state.number == 42:
                return Increment()
            else:
                return End(ctx.state.number)

    never_42_graph = Graph(nodes=(Increment, Check42))
    ```
    _(This example is complete, it can be run "as is")_

    See [`run`][pydantic_graph.graph.Graph.run] For an example of running graph, and
    [`mermaid_code`][pydantic_graph.graph.Graph.mermaid_code] for an example of generating a mermaid diagram
    from the graph.
    """

    name: str | None
    node_defs: dict[str, NodeDef[StateT, DepsT, RunEndT]]
    _state_type: type[StateT] | _utils.Unset = field(repr=False)
    _run_end_type: type[RunEndT] | _utils.Unset = field(repr=False)
    _auto_instrument: bool = field(repr=False)

    def __init__(
        self,
        *,
        nodes: Sequence[type[BaseNode[StateT, DepsT, RunEndT]]],
        name: str | None = None,
        state_type: type[StateT] | _utils.Unset = _utils.UNSET,
        run_end_type: type[RunEndT] | _utils.Unset = _utils.UNSET,
        auto_instrument: bool = True,
    ):
        """Create a graph from a sequence of nodes.

        Args:
            nodes: The nodes which make up the graph, nodes need to be unique and all be generic in the same
                state type.
            name: Optional name for the graph, if not provided the name will be inferred from the calling frame
                on the first call to a graph method.
            state_type: The type of the state for the graph, this can generally be inferred from `nodes`.
            run_end_type: The type of the result of running the graph, this can generally be inferred from `nodes`.
            auto_instrument: Whether to create a span for the graph run and the execution of each node's run method.
        """
        self.name = name
        self._state_type = state_type
        self._run_end_type = run_end_type
        self._auto_instrument = auto_instrument

        parent_namespace = _utils.get_parent_namespace(inspect.currentframe())
        self.node_defs: dict[str, NodeDef[StateT, DepsT, RunEndT]] = {}
        for node in nodes:
            self._register_node(node, parent_namespace)

        self._validate_edges()

    async def run(
        self: Graph[StateT, DepsT, T],
        start_node: BaseNode[StateT, DepsT, T],
        *,
        state: StateT = None,
        deps: DepsT = None,
        persistence: StatePersistence[StateT, T] | None = None,
        infer_name: bool = True,
    ) -> T:
        """Run the graph from a starting node until it ends.

        Args:
            start_node: the first node to run, since the graph definition doesn't define the entry point in the graph,
                you need to provide the starting node.
            state: The initial state of the graph.
            deps: The dependencies of the graph.
            persistence: State persistence interface, defaults to
                [`SimpleStatePersistence`][pydantic_graph.state.memory.SimpleStatePersistence] if `None`.
            infer_name: Whether to infer the graph name from the calling frame.

        Returns:
            The result type from ending the run and the history of the run.

        Here's an example of running the graph from [above][pydantic_graph.graph.Graph]:

        ```py {title="run_never_42.py" noqa="I001" py="3.10"}
        from never_42 import Increment, MyState, never_42_graph

        async def main():
            state = MyState(1)
            await never_42_graph.run(Increment(), state=state)
            print(state)
            #> MyState(number=2)

            state = MyState(41)
            await never_42_graph.run(Increment(), state=state)
            print(state)
            #> MyState(number=43)
        ```
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

        if persistence is None:
            persistence = SimpleStatePersistence()

        self.set_persistence_types(persistence)

        with ExitStack() as stack:
            if self._auto_instrument:
                stack.enter_context(
                    _logfire.span('{graph_name} run {start=}', graph_name=self.name or 'graph', start=start_node)
                )

            next_node = start_node
            while True:
                next_node = await self._next(next_node, persistence, state, deps)
                if isinstance(next_node, End):
                    await persistence.snapshot_end(state, next_node)
                    return next_node.data
                elif not isinstance(next_node, BaseNode):
                    if TYPE_CHECKING:
                        typing_extensions.assert_never(next_node)
                    else:
                        raise exceptions.GraphRuntimeError(
                            f'Invalid node return type: `{type(next_node).__name__}`. Expected `BaseNode` or `End`.'
                        )

    def run_sync(
        self: Graph[StateT, DepsT, T],
        start_node: BaseNode[StateT, DepsT, T],
        *,
        state: StateT = None,
        deps: DepsT = None,
        persistence: StatePersistence[StateT, T] | None = None,
        infer_name: bool = True,
    ) -> T:
        """Run the graph synchronously.

        This is a convenience method that wraps [`self.run`][pydantic_graph.Graph.run] with `loop.run_until_complete(...)`.
        You therefore can't use this method inside async code or if there's an active event loop.

        Args:
            start_node: the first node to run, since the graph definition doesn't define the entry point in the graph,
                you need to provide the starting node.
            state: The initial state of the graph.
            deps: The dependencies of the graph.
            persistence: State persistence interface, defaults to
                [`SimpleStatePersistence`][pydantic_graph.state.memory.SimpleStatePersistence] if `None`.
            infer_name: Whether to infer the graph name from the calling frame.

        Returns:
            The result type from ending the run and the history of the run.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())
        return asyncio.get_event_loop().run_until_complete(
            self.run(start_node, state=state, deps=deps, persistence=persistence, infer_name=False)
        )

    async def next(
        self: Graph[StateT, DepsT, T],
        node: BaseNode[StateT, DepsT, T],
        *,
        persistence: StatePersistence[StateT, T] | None = None,
        state: StateT = None,
        deps: DepsT = None,
        infer_name: bool = True,
    ) -> BaseNode[StateT, DepsT, Any] | End[T]:
        """Run a node in the graph and return the next node to run.

        Args:
            node: The node to run.
            persistence: State persistence interface, defaults to
                [`SimpleStatePersistence`][pydantic_graph.state.memory.SimpleStatePersistence] if `None`.
            state: The current state of the graph.
            deps: The dependencies of the graph.
            infer_name: Whether to infer the graph name from the calling frame.

        Returns:
            The next node to run or [`End`][pydantic_graph.nodes.End] if the graph has finished.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

        if persistence is None:
            persistence = SimpleStatePersistence()
        self.set_persistence_types(persistence)

        return await self._next(node, persistence, state, deps)

    async def _next(
        self: Graph[StateT, DepsT, T],
        node: BaseNode[StateT, DepsT, T],
        persistence: StatePersistence[StateT, T],
        state: StateT,
        deps: DepsT,
    ) -> BaseNode[StateT, DepsT, Any] | End[T]:
        node_id = node.get_id()
        if node_id not in self.node_defs:
            raise exceptions.GraphRuntimeError(f'Node `{node}` is not in the graph.')

        await persistence.snapshot_node(state, node)

        with ExitStack() as stack:
            if self._auto_instrument:
                stack.enter_context(_logfire.span('run node {node_id}', node_id=node_id, node=node))
            ctx = GraphRunContext(state, deps)
            async with persistence.record_run():
                next_or_end = await node.run(ctx)
        return next_or_end

    async def next_from_persistence(
        self: Graph[StateT, DepsT, T],
        persistence: StatePersistence[StateT, T],
        *,
        deps: DepsT = None,
        infer_name: bool = True,
    ) -> BaseNode[StateT, DepsT, Any] | End[T]:
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

        snapshot = await persistence.restore_node_snapshot()
        return await self.next(
            snapshot.node,
            persistence=persistence,
            state=snapshot.state,
            deps=deps,
            infer_name=False,
        )

    def set_persistence_types(self, persistence: StatePersistence[StateT, RunEndT]) -> None:
        with set_nodes_type_context([node_def.node for node_def in self.node_defs.values()]):
            persistence.set_types(lambda: self._inferred_types)

    def mermaid_code(
        self,
        *,
        start_node: Sequence[mermaid.NodeIdent] | mermaid.NodeIdent | None = None,
        title: str | None | typing_extensions.Literal[False] = None,
        edge_labels: bool = True,
        notes: bool = True,
        highlighted_nodes: Sequence[mermaid.NodeIdent] | mermaid.NodeIdent | None = None,
        highlight_css: str = mermaid.DEFAULT_HIGHLIGHT_CSS,
        infer_name: bool = True,
        direction: mermaid.StateDiagramDirection | None = None,
    ) -> str:
        """Generate a diagram representing the graph as [mermaid](https://mermaid.js.org/) diagram.

        This method calls [`pydantic_graph.mermaid.generate_code`][pydantic_graph.mermaid.generate_code].

        Args:
            start_node: The node or nodes which can start the graph.
            title: The title of the diagram, use `False` to not include a title.
            edge_labels: Whether to include edge labels.
            notes: Whether to include notes on each node.
            highlighted_nodes: Optional node or nodes to highlight.
            highlight_css: The CSS to use for highlighting nodes.
            infer_name: Whether to infer the graph name from the calling frame.
            direction: The direction of flow.

        Returns:
            The mermaid code for the graph, which can then be rendered as a diagram.

        Here's an example of generating a diagram for the graph from [above][pydantic_graph.graph.Graph]:

        ```py {title="never_42.py" py="3.10"}
        from never_42 import Increment, never_42_graph

        print(never_42_graph.mermaid_code(start_node=Increment))
        '''
        ---
        title: never_42_graph
        ---
        stateDiagram-v2
          [*] --> Increment
          Increment --> Check42
          Check42 --> Increment
          Check42 --> [*]
        '''
        ```

        The rendered diagram will look like this:

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
            direction=direction,
        )

    def mermaid_image(
        self, infer_name: bool = True, **kwargs: typing_extensions.Unpack[mermaid.MermaidConfig]
    ) -> bytes:
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
        self, path: Path | str, /, *, infer_name: bool = True, **kwargs: typing_extensions.Unpack[mermaid.MermaidConfig]
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

    @cached_property
    def _inferred_types(self) -> tuple[type[StateT], type[RunEndT]]:
        if _utils.is_set(self._state_type) and _utils.is_set(self._run_end_type):
            return self._state_type, self._run_end_type

        state_type = self._state_type
        run_end_type = self._run_end_type

        for node_def in self.node_defs.values():
            for base in typing_extensions.get_original_bases(node_def.node):
                if typing_extensions.get_origin(base) is BaseNode:
                    args = typing_extensions.get_args(base)
                    if not _utils.is_set(state_type) and args:
                        state_type = args[0]

                    if not _utils.is_set(run_end_type) and len(args) == 3:
                        t = args[2]
                        if not _utils.is_never(t):
                            run_end_type = t
                    if _utils.is_set(state_type) and _utils.is_set(run_end_type):
                        return state_type, run_end_type
                    # break the inner (bases) loop
                    break

        if not _utils.is_set(state_type):
            # state defaults to None, so use that if we can't infer it
            state_type = None
        if not _utils.is_set(run_end_type):
            # this happens if a graph has no return nodes, use None so any downstream errors are clear
            run_end_type = None
        return state_type, run_end_type  # pyright: ignore[reportReturnType]

    def _register_node(
        self: Graph[StateT, DepsT, T],
        node: type[BaseNode[StateT, DepsT, T]],
        parent_namespace: dict[str, Any] | None,
    ) -> None:
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

    def _infer_name(self, function_frame: types.FrameType | None) -> None:
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
