from __future__ import annotations as _annotations

import inspect
import types
from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractContextManager, ExitStack, asynccontextmanager
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Generic, cast

import logfire_api
import typing_extensions
from logfire_api import LogfireSpan
from typing_extensions import deprecated
from typing_inspection import typing_objects

from . import _utils, exceptions, mermaid
from .nodes import BaseNode, DepsT, End, GraphRunContext, NodeDef, RunEndT, StateT
from .persistence import BaseStatePersistence
from .persistence.in_mem import SimpleStatePersistence

# while waiting for https://github.com/pydantic/logfire/issues/745
try:
    import logfire._internal.stack_info
except ImportError:
    pass
else:
    from pathlib import Path

    logfire._internal.stack_info.NON_USER_CODE_PREFIXES += (str(Path(__file__).parent.absolute()),)  # pyright: ignore[reportPrivateImportUsage]


__all__ = 'Graph', 'GraphRun', 'GraphRunResult'

_logfire = logfire_api.Logfire(otel_scope='pydantic-graph')


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
    auto_instrument: bool = field(repr=False)

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
        self.auto_instrument = auto_instrument

        parent_namespace = _utils.get_parent_namespace(inspect.currentframe())
        self.node_defs = {}
        for node in nodes:
            self._register_node(node, parent_namespace)

        self._validate_edges()

    async def run(
        self,
        start_node: BaseNode[StateT, DepsT, RunEndT],
        *,
        state: StateT = None,
        deps: DepsT = None,
        persistence: BaseStatePersistence[StateT, RunEndT] | None = None,
        infer_name: bool = True,
        span: LogfireSpan | None = None,
    ) -> GraphRunResult[StateT, RunEndT]:
        """Run the graph from a starting node until it ends.

        Args:
            start_node: the first node to run, since the graph definition doesn't define the entry point in the graph,
                you need to provide the starting node.
            state: The initial state of the graph.
            deps: The dependencies of the graph.
            persistence: State persistence interface, defaults to
                [`SimpleStatePersistence`][pydantic_graph.SimpleStatePersistence] if `None`.
            infer_name: Whether to infer the graph name from the calling frame.
            span: The span to use for the graph run. If not provided, a span will be created depending on the value of
                the `auto_instrument` field.

        Returns:
            A `GraphRunResult` containing information about the run, including its final result.

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

        async with self.iter(
            start_node, state=state, deps=deps, persistence=persistence, span=span, infer_name=False
        ) as graph_run:
            async for _node in graph_run:
                pass

        final_result = graph_run.result
        assert final_result is not None, 'GraphRun should have a final result'
        return final_result

    def run_sync(
        self,
        start_node: BaseNode[StateT, DepsT, RunEndT],
        *,
        state: StateT = None,
        deps: DepsT = None,
        persistence: BaseStatePersistence[StateT, RunEndT] | None = None,
        infer_name: bool = True,
    ) -> GraphRunResult[StateT, RunEndT]:
        """Synchronously run the graph.

        This is a convenience method that wraps [`self.run`][pydantic_graph.Graph.run] with `loop.run_until_complete(...)`.
        You therefore can't use this method inside async code or if there's an active event loop.

        Args:
            start_node: the first node to run, since the graph definition doesn't define the entry point in the graph,
                you need to provide the starting node.
            state: The initial state of the graph.
            deps: The dependencies of the graph.
            persistence: State persistence interface, defaults to
                [`SimpleStatePersistence`][pydantic_graph.SimpleStatePersistence] if `None`.
            infer_name: Whether to infer the graph name from the calling frame.

        Returns:
            The result type from ending the run and the history of the run.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

        return _utils.get_event_loop().run_until_complete(
            self.run(start_node, state=state, deps=deps, persistence=persistence, infer_name=False)
        )

    @asynccontextmanager
    async def iter(
        self,
        start_node: BaseNode[StateT, DepsT, RunEndT],
        *,
        state: StateT = None,
        deps: DepsT = None,
        persistence: BaseStatePersistence[StateT, RunEndT] | None = None,
        span: AbstractContextManager[Any] | None = None,
        infer_name: bool = True,
    ) -> AsyncIterator[GraphRun[StateT, DepsT, RunEndT]]:
        """A contextmanager which can be used to iterate over the graph's nodes as they are executed.

        This method returns a `GraphRun` object which can be used to async-iterate over the nodes of this `Graph` as
        they are executed. This is the API to use if you want to record or interact with the nodes as the graph
        execution unfolds.

        The `GraphRun` can also be used to manually drive the graph execution by calling
        [`GraphRun.next`][pydantic_graph.graph.GraphRun.next].

        The `GraphRun` provides access to the full run history, state, deps, and the final result of the run once
        it has completed.

        For more details, see the API documentation of [`GraphRun`][pydantic_graph.graph.GraphRun].

        Args:
            start_node: the first node to run. Since the graph definition doesn't define the entry point in the graph,
                you need to provide the starting node.
            state: The initial state of the graph.
            deps: The dependencies of the graph.
            persistence: State persistence interface, defaults to
                [`SimpleStatePersistence`][pydantic_graph.SimpleStatePersistence] if `None`.
            span: The span to use for the graph run. If not provided, a new span will be created.
            infer_name: Whether to infer the graph name from the calling frame.

        Returns: A GraphRun that can be async iterated over to drive the graph to completion.
        """
        if infer_name and self.name is None:
            # f_back because `asynccontextmanager` adds one frame
            if frame := inspect.currentframe():  # pragma: no branch
                self._infer_name(frame.f_back)

        if persistence is None:
            persistence = SimpleStatePersistence()
        persistence.set_graph_types(self)

        if self.auto_instrument and span is None:
            span = logfire_api.span('run graph {graph.name}', graph=self)

        with ExitStack() as stack:
            if span is not None:
                stack.enter_context(span)
            yield GraphRun[StateT, DepsT, RunEndT](
                graph=self, start_node=start_node, persistence=persistence, state=state, deps=deps
            )

    @asynccontextmanager
    async def iter_from_persistence(
        self,
        persistence: BaseStatePersistence[StateT, RunEndT],
        *,
        deps: DepsT = None,
        span: AbstractContextManager[Any] | None = None,
        infer_name: bool = True,
    ) -> AsyncIterator[GraphRun[StateT, DepsT, RunEndT]]:
        """A contextmanager to iterate over the graph's nodes as they are executed, created from a persistence object.

        This method has similar functionality to [`iter`][pydantic_graph.graph.Graph.iter],
        but instead of passing the node to run, it will restore the node and state from state persistence.

        Args:
            persistence: The state persistence interface to use.
            deps: The dependencies of the graph.
            span: The span to use for the graph run. If not provided, a new span will be created.
            infer_name: Whether to infer the graph name from the calling frame.

        Returns: A GraphRun that can be async iterated over to drive the graph to completion.
        """
        if infer_name and self.name is None:
            # f_back because `asynccontextmanager` adds one frame
            if frame := inspect.currentframe():  # pragma: no branch
                self._infer_name(frame.f_back)

        persistence.set_graph_types(self)

        snapshot = await persistence.load_next()
        if snapshot is None:
            raise exceptions.GraphRuntimeError('Unable to restore snapshot from state persistence.')

        snapshot.node.set_snapshot_id(snapshot.id)

        if self.auto_instrument and span is None:
            span = logfire_api.span('run graph {graph.name}', graph=self)

        with ExitStack() as stack:
            if span is not None:
                stack.enter_context(span)
            yield GraphRun[StateT, DepsT, RunEndT](
                graph=self,
                start_node=snapshot.node,
                persistence=persistence,
                state=snapshot.state,
                deps=deps,
                snapshot_id=snapshot.id,
            )

    async def initialize(
        self,
        node: BaseNode[StateT, DepsT, RunEndT],
        persistence: BaseStatePersistence[StateT, RunEndT],
        *,
        state: StateT = None,
        infer_name: bool = True,
    ) -> None:
        """Initialize a new graph run in persistence without running it.

        This is useful if you want to set up a graph run to be run later, e.g. via
        [`iter_from_persistence`][pydantic_graph.graph.Graph.iter_from_persistence].

        Args:
            node: The node to run first.
            persistence: State persistence interface.
            state: The start state of the graph.
            infer_name: Whether to infer the graph name from the calling frame.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

        persistence.set_graph_types(self)
        await persistence.snapshot_node(state, node)

    @deprecated('`next` is deprecated, use `async with graph.iter(...) as run:  run.next()` instead')
    async def next(
        self,
        node: BaseNode[StateT, DepsT, RunEndT],
        persistence: BaseStatePersistence[StateT, RunEndT],
        *,
        state: StateT = None,
        deps: DepsT = None,
        infer_name: bool = True,
    ) -> BaseNode[StateT, DepsT, Any] | End[RunEndT]:
        """Run a node in the graph and return the next node to run.

        Args:
            node: The node to run.
            persistence: State persistence interface, defaults to
                [`SimpleStatePersistence`][pydantic_graph.SimpleStatePersistence] if `None`.
            state: The current state of the graph.
            deps: The dependencies of the graph.
            infer_name: Whether to infer the graph name from the calling frame.

        Returns:
            The next node to run or [`End`][pydantic_graph.nodes.End] if the graph has finished.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

        persistence.set_graph_types(self)
        run = GraphRun[StateT, DepsT, RunEndT](
            graph=self,
            start_node=node,
            persistence=persistence,
            state=state,
            deps=deps,
        )
        return await run.next(node)

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

        ```py {title="mermaid_never_42.py" py="3.10"}
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

    def get_nodes(self) -> Sequence[type[BaseNode[StateT, DepsT, RunEndT]]]:
        """Get the nodes in the graph."""
        return [node_def.node for node_def in self.node_defs.values()]

    @cached_property
    def inferred_types(self) -> tuple[type[StateT], type[RunEndT]]:
        # Get the types of the state and run end from the graph.
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
                        if not typing_objects.is_never(t):
                            run_end_type = t
                    if _utils.is_set(state_type) and _utils.is_set(run_end_type):
                        return state_type, run_end_type  # pyright: ignore[reportReturnType]
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
        self,
        node: type[BaseNode[StateT, DepsT, RunEndT]],
        parent_namespace: dict[str, Any] | None,
    ) -> None:
        node_id = node.get_node_id()
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


class GraphRun(Generic[StateT, DepsT, RunEndT]):
    """A stateful, async-iterable run of a [`Graph`][pydantic_graph.graph.Graph].

    You typically get a `GraphRun` instance from calling
    `async with [my_graph.iter(...)][pydantic_graph.graph.Graph.iter] as graph_run:`. That gives you the ability to iterate
    through nodes as they run, either by `async for` iteration or by repeatedly calling `.next(...)`.

    Here's an example of iterating over the graph from [above][pydantic_graph.graph.Graph]:
    ```py {title="iter_never_42.py" noqa="I001" py="3.10"}
    from copy import deepcopy
    from never_42 import Increment, MyState, never_42_graph

    async def main():
        state = MyState(1)
        async with never_42_graph.iter(Increment(), state=state) as graph_run:
            node_states = [(graph_run.next_node, deepcopy(graph_run.state))]
            async for node in graph_run:
                node_states.append((node, deepcopy(graph_run.state)))
            print(node_states)
            '''
            [
                (Increment(), MyState(number=1)),
                (Increment(), MyState(number=1)),
                (Check42(), MyState(number=2)),
                (End(data=2), MyState(number=2)),
            ]
            '''

        state = MyState(41)
        async with never_42_graph.iter(Increment(), state=state) as graph_run:
            node_states = [(graph_run.next_node, deepcopy(graph_run.state))]
            async for node in graph_run:
                node_states.append((node, deepcopy(graph_run.state)))
            print(node_states)
            '''
            [
                (Increment(), MyState(number=41)),
                (Increment(), MyState(number=41)),
                (Check42(), MyState(number=42)),
                (Increment(), MyState(number=42)),
                (Check42(), MyState(number=43)),
                (End(data=43), MyState(number=43)),
            ]
            '''
    ```

    See the [`GraphRun.next` documentation][pydantic_graph.graph.GraphRun.next] for an example of how to manually
    drive the graph run.
    """

    def __init__(
        self,
        *,
        graph: Graph[StateT, DepsT, RunEndT],
        start_node: BaseNode[StateT, DepsT, RunEndT],
        persistence: BaseStatePersistence[StateT, RunEndT],
        state: StateT,
        deps: DepsT,
        snapshot_id: str | None = None,
    ):
        """Create a new run for a given graph, starting at the specified node.

        Typically, you'll use [`Graph.iter`][pydantic_graph.graph.Graph.iter] rather than calling this directly.

        Args:
            graph: The [`Graph`][pydantic_graph.graph.Graph] to run.
            start_node: The node where execution will begin.
            persistence: State persistence interface.
            state: A shared state object or primitive (like a counter, dataclass, etc.) that is available
                to all nodes via `ctx.state`.
            deps: Optional dependencies that each node can access via `ctx.deps`, e.g. database connections,
                configuration, or logging clients.
            snapshot_id: The ID of the snapshot the node came from.
        """
        self.graph = graph
        self.persistence = persistence
        self._snapshot_id: str | None = snapshot_id
        self.state = state
        self.deps = deps

        self._next_node: BaseNode[StateT, DepsT, RunEndT] | End[RunEndT] = start_node
        self._is_started: bool = False

    @property
    def next_node(self) -> BaseNode[StateT, DepsT, RunEndT] | End[RunEndT]:
        """The next node that will be run in the graph.

        This is the next node that will be used during async iteration, or if a node is not passed to `self.next(...)`.
        """
        return self._next_node

    @property
    def result(self) -> GraphRunResult[StateT, RunEndT] | None:
        """The final result of the graph run if the run is completed, otherwise `None`."""
        if not isinstance(self._next_node, End):
            return None  # The GraphRun has not finished running
        return GraphRunResult(
            self._next_node.data,
            state=self.state,
            persistence=self.persistence,
        )

    async def next(
        self, node: BaseNode[StateT, DepsT, RunEndT] | None = None
    ) -> BaseNode[StateT, DepsT, RunEndT] | End[RunEndT]:
        """Manually drive the graph run by passing in the node you want to run next.

        This lets you inspect or mutate the node before continuing execution, or skip certain nodes
        under dynamic conditions. The graph run should stop when you return an [`End`][pydantic_graph.nodes.End] node.

        Here's an example of using `next` to drive the graph from [above][pydantic_graph.graph.Graph]:
        ```py {title="next_never_42.py" noqa="I001" py="3.10"}
        from copy import deepcopy
        from pydantic_graph import End
        from never_42 import Increment, MyState, never_42_graph

        async def main():
            state = MyState(48)
            async with never_42_graph.iter(Increment(), state=state) as graph_run:
                next_node = graph_run.next_node  # start with the first node
                node_states = [(next_node, deepcopy(graph_run.state))]

                while not isinstance(next_node, End):
                    if graph_run.state.number == 50:
                        graph_run.state.number = 42
                    next_node = await graph_run.next(next_node)
                    node_states.append((next_node, deepcopy(graph_run.state)))

                print(node_states)
                '''
                [
                    (Increment(), MyState(number=48)),
                    (Check42(), MyState(number=49)),
                    (End(data=49), MyState(number=49)),
                ]
                '''
        ```

        Args:
            node: The node to run next in the graph. If not specified, uses `self.next_node`, which is initialized to
                the `start_node` of the run and updated each time a new node is returned.

        Returns:
            The next node returned by the graph logic, or an [`End`][pydantic_graph.nodes.End] node if
            the run has completed.
        """
        if node is None:
            # This cast is necessary because self._next_node could be an `End`. You'll get a runtime error if that's
            # the case, but if it is, the only way to get there would be to have tried calling next manually after
            # the run finished. Either way, maybe it would be better to not do this cast...
            node = cast(BaseNode[StateT, DepsT, RunEndT], self._next_node)
            node_snapshot_id = node.get_snapshot_id()
        else:
            node_snapshot_id = node.get_snapshot_id()

        if node_snapshot_id != self._snapshot_id:
            await self.persistence.snapshot_node_if_new(node_snapshot_id, self.state, node)
            self._snapshot_id = node_snapshot_id

        if not isinstance(node, BaseNode):
            # While technically this is not compatible with the documented method signature, it's an easy mistake to
            # make, and we should eagerly provide a more helpful error message than you'd get otherwise.
            raise TypeError(f'`next` must be called with a `BaseNode` instance, got {node!r}.')

        node_id = node.get_node_id()
        if node_id not in self.graph.node_defs:
            raise exceptions.GraphRuntimeError(f'Node `{node}` is not in the graph.')

        with ExitStack() as stack:
            if self.graph.auto_instrument:
                stack.enter_context(_logfire.span('run node {node_id}', node_id=node_id, node=node))

            async with self.persistence.record_run(node_snapshot_id):
                ctx = GraphRunContext(self.state, self.deps)
                self._next_node = await node.run(ctx)

        if isinstance(self._next_node, End):
            self._snapshot_id = self._next_node.get_snapshot_id()
            await self.persistence.snapshot_end(self.state, self._next_node)
        elif isinstance(self._next_node, BaseNode):
            self._snapshot_id = self._next_node.get_snapshot_id()
            await self.persistence.snapshot_node(self.state, self._next_node)
        else:
            raise exceptions.GraphRuntimeError(
                f'Invalid node return type: `{type(self._next_node).__name__}`. Expected `BaseNode` or `End`.'
            )

        return self._next_node

    def __aiter__(self) -> AsyncIterator[BaseNode[StateT, DepsT, RunEndT] | End[RunEndT]]:
        return self

    async def __anext__(self) -> BaseNode[StateT, DepsT, RunEndT] | End[RunEndT]:
        """Use the last returned node as the input to `Graph.next`."""
        if not self._is_started:
            self._is_started = True
            return self._next_node

        if isinstance(self._next_node, End):
            raise StopAsyncIteration

        return await self.next(self._next_node)

    def __repr__(self) -> str:
        return f'<GraphRun graph={self.graph.name or "[unnamed]"}>'


@dataclass
class GraphRunResult(Generic[StateT, RunEndT]):
    """The final result of running a graph."""

    output: RunEndT
    state: StateT
    persistence: BaseStatePersistence[StateT, RunEndT] = field(repr=False)
