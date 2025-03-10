from __future__ import annotations as _annotations

import inspect
import types
from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractContextManager, ExitStack, asynccontextmanager
from dataclasses import dataclass, field
from functools import cached_property
from time import perf_counter
from typing import TYPE_CHECKING, Annotated, Any, Callable, Generic, TypeVar

import logfire_api
import pydantic
import typing_extensions
from logfire_api import LogfireSpan
from typing_inspection import typing_objects

from . import _utils, exceptions, mermaid
from .nodes import BaseNode, DepsT, End, GraphRunContext, NodeDef, RunEndT
from .state import EndStep, HistoryStep, NodeStep, StateT, deep_copy_state, nodes_schema_var

# while waiting for https://github.com/pydantic/logfire/issues/745
try:
    import logfire._internal.stack_info
except ImportError:
    pass
else:
    from pathlib import Path

    logfire._internal.stack_info.NON_USER_CODE_PREFIXES += (str(Path(__file__).parent.absolute()),)


__all__ = ('Graph', 'GraphRun', 'GraphRunResult')

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
    snapshot_state: Callable[[StateT], StateT]
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
        snapshot_state: Callable[[StateT], StateT] = deep_copy_state,
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
            snapshot_state: A function to snapshot the state of the graph, this is used in
                [`NodeStep`][pydantic_graph.state.NodeStep] and [`EndStep`][pydantic_graph.state.EndStep] to record
                the state before each step.
            auto_instrument: Whether to create a span for the graph run and the execution of each node's run method.
        """
        self.name = name
        self._state_type = state_type
        self._run_end_type = run_end_type
        self._auto_instrument = auto_instrument
        self.snapshot_state = snapshot_state

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
        infer_name: bool = True,
        span: LogfireSpan | None = None,
    ) -> GraphRunResult[StateT, T]:
        """Run the graph from a starting node until it ends.

        Args:
            start_node: the first node to run, since the graph definition doesn't define the entry point in the graph,
                you need to provide the starting node.
            state: The initial state of the graph.
            deps: The dependencies of the graph.
            infer_name: Whether to infer the graph name from the calling frame.
            span: The span to use for the graph run. If not provided, a span will be created depending on the value of
                the `_auto_instrument` field.

        Returns:
            A `GraphRunResult` containing information about the run, including its final result.

        Here's an example of running the graph from [above][pydantic_graph.graph.Graph]:

        ```py {title="run_never_42.py" noqa="I001" py="3.10"}
        from never_42 import Increment, MyState, never_42_graph

        async def main():
            state = MyState(1)
            graph_run_result = await never_42_graph.run(Increment(), state=state)
            print(state)
            #> MyState(number=2)
            print(len(graph_run_result.history))
            #> 3

            state = MyState(41)
            graph_run_result = await never_42_graph.run(Increment(), state=state)
            print(state)
            #> MyState(number=43)
            print(len(graph_run_result.history))
            #> 5
        ```
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

        async with self.iter(start_node, state=state, deps=deps, infer_name=infer_name, span=span) as graph_run:
            async for _node in graph_run:
                pass

        final_result = graph_run.result
        assert final_result is not None, 'GraphRun should have a final result'
        return final_result

    @asynccontextmanager
    async def iter(
        self: Graph[StateT, DepsT, T],
        start_node: BaseNode[StateT, DepsT, T],
        *,
        state: StateT = None,
        deps: DepsT = None,
        infer_name: bool = True,
        span: AbstractContextManager[Any] | None = None,
    ) -> AsyncIterator[GraphRun[StateT, DepsT, T]]:
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
            infer_name: Whether to infer the graph name from the calling frame.
            span: The span to use for the graph run. If not provided, a new span will be created.

        Yields:
            A GraphRun that can be async iterated over to drive the graph to completion.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

        if self._auto_instrument and span is None:
            span = logfire_api.span('run graph {graph.name}', graph=self)

        with ExitStack() as stack:
            if span is not None:
                stack.enter_context(span)
            yield GraphRun[StateT, DepsT, T](
                self,
                start_node,
                history=[],
                state=state,
                deps=deps,
                auto_instrument=self._auto_instrument,
            )

    def run_sync(
        self: Graph[StateT, DepsT, T],
        start_node: BaseNode[StateT, DepsT, T],
        *,
        state: StateT = None,
        deps: DepsT = None,
        infer_name: bool = True,
    ) -> GraphRunResult[StateT, T]:
        """Synchronously run the graph.

        This is a convenience method that wraps [`self.run`][pydantic_graph.Graph.run] with `loop.run_until_complete(...)`.
        You therefore can't use this method inside async code or if there's an active event loop.

        Args:
            start_node: the first node to run, since the graph definition doesn't define the entry point in the graph,
                you need to provide the starting node.
            state: The initial state of the graph.
            deps: The dependencies of the graph.
            infer_name: Whether to infer the graph name from the calling frame.

        Returns:
            The result type from ending the run and the history of the run.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())
        return _utils.get_event_loop().run_until_complete(
            self.run(start_node, state=state, deps=deps, infer_name=False)
        )

    async def next(
        self: Graph[StateT, DepsT, T],
        node: BaseNode[StateT, DepsT, T],
        history: list[HistoryStep[StateT, T]],
        *,
        state: StateT = None,
        deps: DepsT = None,
        infer_name: bool = True,
    ) -> BaseNode[StateT, DepsT, Any] | End[T]:
        """Run a node in the graph and return the next node to run.

        Args:
            node: The node to run.
            history: The history of the graph run so far. NOTE: this will be mutated to add the new step.
            state: The current state of the graph.
            deps: The dependencies of the graph.
            infer_name: Whether to infer the graph name from the calling frame.

        Returns:
            The next node to run or [`End`][pydantic_graph.nodes.End] if the graph has finished.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

        if isinstance(node, End):
            # While technically this is not compatible with the documented method signature, it's an easy mistake to
            # make, and we should eagerly provide a more helpful error message than you'd get otherwise.
            raise exceptions.GraphRuntimeError(f'Cannot call `next` with an `End` node: {node!r}.')

        node_id = node.get_id()
        if node_id not in self.node_defs:
            raise exceptions.GraphRuntimeError(f'Node `{node}` is not in the graph.')

        with ExitStack() as stack:
            if self._auto_instrument:
                stack.enter_context(_logfire.span('run node {node_id}', node_id=node_id, node=node))
            ctx = GraphRunContext(state, deps)
            start_ts = _utils.now_utc()
            start = perf_counter()
            next_node = await node.run(ctx)
            duration = perf_counter() - start

        history.append(
            NodeStep(state=state, node=node, start_ts=start_ts, duration=duration, snapshot_state=self.snapshot_state)
        )

        if isinstance(next_node, End):
            history.append(EndStep(result=next_node))
        elif not isinstance(next_node, BaseNode):
            if TYPE_CHECKING:
                typing_extensions.assert_never(next_node)
            else:
                raise exceptions.GraphRuntimeError(
                    f'Invalid node return type: `{type(next_node).__name__}`. Expected `BaseNode` or `End`.'
                )

        return next_node

    def dump_history(
        self: Graph[StateT, DepsT, T], history: list[HistoryStep[StateT, T]], *, indent: int | None = None
    ) -> bytes:
        """Dump the history of a graph run as JSON.

        Args:
            history: The history of the graph run.
            indent: The number of spaces to indent the JSON.

        Returns:
            The JSON representation of the history.
        """
        return self.history_type_adapter.dump_json(history, indent=indent)

    def load_history(self, json_bytes: str | bytes | bytearray) -> list[HistoryStep[StateT, RunEndT]]:
        """Load the history of a graph run from JSON.

        Args:
            json_bytes: The JSON representation of the history.

        Returns:
            The history of the graph run.
        """
        return self.history_type_adapter.validate_json(json_bytes)

    @cached_property
    def history_type_adapter(self) -> pydantic.TypeAdapter[list[HistoryStep[StateT, RunEndT]]]:
        nodes = [node_def.node for node_def in self.node_defs.values()]
        state_t = self._get_state_type()
        end_t = self._get_run_end_type()
        token = nodes_schema_var.set(nodes)
        try:
            ta = pydantic.TypeAdapter(list[Annotated[HistoryStep[state_t, end_t], pydantic.Discriminator('kind')]])
        finally:
            nodes_schema_var.reset(token)
        return ta

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

    def _get_state_type(self) -> type[StateT]:
        if _utils.is_set(self._state_type):
            return self._state_type

        for node_def in self.node_defs.values():
            for base in typing_extensions.get_original_bases(node_def.node):
                if typing_extensions.get_origin(base) is BaseNode:
                    args = typing_extensions.get_args(base)
                    if args:
                        return args[0]
                    # break the inner (bases) loop
                    break
        # state defaults to None, so use that if we can't infer it
        return type(None)  # pyright: ignore[reportReturnType]

    def _get_run_end_type(self) -> type[RunEndT]:
        if _utils.is_set(self._run_end_type):
            return self._run_end_type

        for node_def in self.node_defs.values():
            for base in typing_extensions.get_original_bases(node_def.node):
                if typing_extensions.get_origin(base) is BaseNode:
                    args = typing_extensions.get_args(base)
                    if len(args) == 3:
                        t = args[2]
                        if not typing_objects.is_never(t):
                            return t
                    # break the inner (bases) loop
                    break
        raise exceptions.GraphSetupError('Could not infer run end type from nodes, please set `run_end_type`.')

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
        graph: Graph[StateT, DepsT, RunEndT],
        start_node: BaseNode[StateT, DepsT, RunEndT],
        *,
        history: list[HistoryStep[StateT, RunEndT]],
        state: StateT,
        deps: DepsT,
        auto_instrument: bool,
    ):
        """Create a new run for a given graph, starting at the specified node.

        Typically, you'll use [`Graph.iter`][pydantic_graph.graph.Graph.iter] rather than calling this directly.

        Args:
            graph: The [`Graph`][pydantic_graph.graph.Graph] to run.
            start_node: The node where execution will begin.
            history: A list of [`HistoryStep`][pydantic_graph.state.HistoryStep] objects that describe
                each step of the run. Usually starts empty; can be populated if resuming.
            state: A shared state object or primitive (like a counter, dataclass, etc.) that is available
                to all nodes via `ctx.state`.
            deps: Optional dependencies that each node can access via `ctx.deps`, e.g. database connections,
                configuration, or logging clients.
            auto_instrument: Whether to automatically create instrumentation spans during the run.
        """
        self.graph = graph
        self.history = history
        self.state = state
        self.deps = deps
        self._auto_instrument = auto_instrument

        self._next_node: BaseNode[StateT, DepsT, RunEndT] | End[RunEndT] = start_node

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
            history=self.history,
        )

    async def next(
        self: GraphRun[StateT, DepsT, T], node: BaseNode[StateT, DepsT, T] | None = None
    ) -> BaseNode[StateT, DepsT, T] | End[T]:
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
            if isinstance(self._next_node, End):
                # Note: we could alternatively just return `self._next_node` here, but it's easier to start with an
                # error and relax the behavior later, than vice versa.
                raise exceptions.GraphRuntimeError('This graph run has already ended.')
            node = self._next_node

        history = self.history
        state = self.state
        deps = self.deps

        self._next_node = await self.graph.next(node, history, state=state, deps=deps, infer_name=False)

        return self._next_node

    def __aiter__(self) -> AsyncIterator[BaseNode[StateT, DepsT, RunEndT] | End[RunEndT]]:
        return self

    async def __anext__(self) -> BaseNode[StateT, DepsT, RunEndT] | End[RunEndT]:
        """Use the last returned node as the input to `Graph.next`."""
        if isinstance(self._next_node, End):
            raise StopAsyncIteration
        return await self.next(self._next_node)

    def __repr__(self) -> str:
        return f'<GraphRun name={self.graph.name or "<unnamed>"} step={len(self.history) + 1}>'


@dataclass
class GraphRunResult(Generic[StateT, RunEndT]):
    """The final result of running a graph."""

    output: RunEndT
    state: StateT
    history: list[HistoryStep[StateT, RunEndT]] = field(repr=False)
