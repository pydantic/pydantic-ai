"""Core graph execution engine for the next version of the pydantic-graph library.

This module provides the main `Graph` class and `GraphRun` execution engine that
handles the orchestration of nodes, edges, and parallel execution paths in
the graph-based workflow system.
"""

from __future__ import annotations as _annotations

import asyncio
import inspect
import types
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Sequence
from contextlib import AbstractContextManager, ExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, cast, get_args, get_origin, overload

from typing_extensions import TypeVar, assert_never

from pydantic_graph import exceptions
from pydantic_graph._utils import AbstractSpan, get_traceparent, logfire_span
from pydantic_graph.beta.decision import Decision
from pydantic_graph.beta.id_types import ForkStack, ForkStackItem, GraphRunID, JoinID, NodeID, NodeRunID, TaskID
from pydantic_graph.beta.join import Join, JoinNode, JoinState, ReducerContext
from pydantic_graph.beta.node import (
    EndNode,
    Fork,
    StartNode,
)
from pydantic_graph.beta.node_types import AnyNode
from pydantic_graph.beta.parent_forks import ParentFork
from pydantic_graph.beta.paths import (
    BroadcastMarker,
    DestinationMarker,
    LabelMarker,
    MapMarker,
    Path,
    TransformMarker,
)
from pydantic_graph.beta.step import NodeStep, Step, StepContext, StepNode
from pydantic_graph.beta.util import unpack_type_expression
from pydantic_graph.nodes import BaseNode, End

if TYPE_CHECKING:
    from pydantic_graph.beta.mermaid import StateDiagramDirection


StateT = TypeVar('StateT', infer_variance=True)
"""Type variable for graph state."""

DepsT = TypeVar('DepsT', infer_variance=True)
"""Type variable for graph dependencies."""

InputT = TypeVar('InputT', infer_variance=True)
"""Type variable for graph inputs."""

OutputT = TypeVar('OutputT', infer_variance=True)
"""Type variable for graph outputs."""


@dataclass(init=False)
class EndMarker(Generic[OutputT]):
    """A marker indicating the end of graph execution with a final value.

    EndMarker is used internally to signal that the graph has completed
    execution and carries the final output value.

    Type Parameters:
        OutputT: The type of the final output value
    """

    _value: OutputT
    """The final output value from the graph execution."""

    def __init__(self, value: OutputT):
        # This manually-defined initializer is necessary due to https://github.com/python/mypy/issues/17623.
        self._value = value

    @property
    def value(self) -> OutputT:
        return self._value


@dataclass
class JoinItem:
    """An item representing data flowing into a join operation.

    JoinItem carries input data from a parallel execution path to a join
    node, along with metadata about which execution 'fork' it originated from.
    """

    join_id: JoinID
    """The ID of the join node this item is targeting."""

    inputs: Any
    """The input data for the join operation."""

    fork_stack: ForkStack
    """The stack of ForkStackItems that led to producing this join item."""


@dataclass(repr=False)
class Graph(Generic[StateT, DepsT, InputT, OutputT]):
    """A complete graph definition ready for execution.

    The Graph class represents a complete workflow graph with typed inputs,
    outputs, state, and dependencies. It contains all nodes, edges, and
    metadata needed for execution.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies
        InputT: The type of the input data
        OutputT: The type of the output data
    """

    name: str | None
    """Optional name for the graph, if not provided the name will be inferred from the calling frame on the first call to a graph method."""

    state_type: type[StateT]
    """The type of the graph state."""

    deps_type: type[DepsT]
    """The type of the dependencies."""

    input_type: type[InputT]
    """The type of the input data."""

    output_type: type[OutputT]
    """The type of the output data."""

    auto_instrument: bool
    """Whether to automatically create instrumentation spans."""

    nodes: dict[NodeID, AnyNode]
    """All nodes in the graph indexed by their ID."""

    edges_by_source: dict[NodeID, list[Path]]
    """Outgoing paths from each source node."""

    parent_forks: dict[JoinID, ParentFork[NodeID]]
    """Parent fork information for each join node."""

    def get_parent_fork(self, join_id: JoinID) -> ParentFork[NodeID]:
        """Get the parent fork information for a join node.

        Args:
            join_id: The ID of the join node

        Returns:
            The parent fork information for the join

        Raises:
            RuntimeError: If the join ID is not found or has no parent fork
        """
        result = self.parent_forks.get(join_id)
        if result is None:
            raise RuntimeError(f'Node {join_id} is not a join node or did not have a dominating fork (this is a bug)')
        return result

    async def run(
        self,
        *,
        state: StateT = None,
        deps: DepsT = None,
        inputs: InputT = None,
        span: AbstractContextManager[AbstractSpan] | None = None,
        infer_name: bool = True,
    ) -> OutputT:
        """Execute the graph and return the final output.

        This is the main entry point for graph execution. It runs the graph
        to completion and returns the final output value.

        Args:
            state: The graph state instance
            deps: The dependencies instance
            inputs: The input data for the graph
            span: Optional span for tracing/instrumentation
            infer_name: Whether to infer the graph name from the calling frame.

        Returns:
            The final output from the graph execution
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

        async with self.iter(state=state, deps=deps, inputs=inputs, span=span, infer_name=False) as graph_run:
            # Note: This would probably be better using `async for _ in graph_run`, but this tests the `next` method,
            # which I'm less confident will be implemented correctly if not used on the critical path. We can change it
            # once we have tests, etc.
            event: Any = None
            while True:
                try:
                    event = await graph_run.next(event)
                except StopAsyncIteration:
                    assert isinstance(event, EndMarker), 'Graph run should end with an EndMarker.'
                    return cast(EndMarker[OutputT], event).value

    @asynccontextmanager
    async def iter(
        self,
        *,
        state: StateT = None,
        deps: DepsT = None,
        inputs: InputT = None,
        span: AbstractContextManager[AbstractSpan] | None = None,
        infer_name: bool = True,
    ) -> AsyncIterator[GraphRun[StateT, DepsT, OutputT]]:
        """Create an iterator for step-by-step graph execution.

        This method allows for more fine-grained control over graph execution,
        enabling inspection of intermediate states and results.

        Args:
            state: The graph state instance
            deps: The dependencies instance
            inputs: The input data for the graph
            span: Optional span for tracing/instrumentation
            infer_name: Whether to infer the graph name from the calling frame.

        Yields:
            A GraphRun instance that can be iterated for step-by-step execution
        """
        if infer_name and self.name is None:
            # f_back because `asynccontextmanager` adds one frame
            if frame := inspect.currentframe():  # pragma: no branch
                self._infer_name(frame.f_back)

        with ExitStack() as stack:
            entered_span: AbstractSpan | None = None
            if span is None:
                if self.auto_instrument:
                    entered_span = stack.enter_context(logfire_span('run graph {graph.name}', graph=self))
            else:
                entered_span = stack.enter_context(span)
            traceparent = None if entered_span is None else get_traceparent(entered_span)
            yield GraphRun[StateT, DepsT, OutputT](
                graph=self,
                state=state,
                deps=deps,
                inputs=inputs,
                traceparent=traceparent,
            )

    def render(self, *, title: str | None = None, direction: StateDiagramDirection | None = None) -> str:
        """Render the graph as a Mermaid diagram string.

        Args:
            title: Optional title for the diagram
            direction: Optional direction for the diagram layout

        Returns:
            A string containing the Mermaid diagram representation
        """
        from pydantic_graph.beta.mermaid import build_mermaid_graph

        return build_mermaid_graph(self.nodes, self.edges_by_source).render(title=title, direction=direction)

    def __repr__(self) -> str:
        super_repr = super().__repr__()  # include class and memory address
        # Insert the result of calling `__str__` before the final '>' in the repr
        return f'{super_repr[:-1]}\n{self}\n{super_repr[-1]}'

    def __str__(self) -> str:
        """Return a Mermaid diagram representation of the graph.

        Returns:
            A string containing the Mermaid diagram of the graph
        """
        return self.render()

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
            if parent_frame.f_locals != parent_frame.f_globals:  # pragma: no branch
                # if we couldn't find the agent in locals and globals are a different dict, try globals
                for name, item in parent_frame.f_globals.items():  # pragma: no branch
                    if item is self:
                        self.name = name
                        return


@dataclass
class GraphTask:
    """A single task representing the execution of a node in the graph.

    GraphTask encapsulates all the information needed to execute a specific
    node, including its inputs and the fork context it's executing within.
    """

    # With our current BaseNode thing, next_node_id and next_node_inputs are merged into `next_node` itself
    node_id: NodeID
    """The ID of the node to execute."""

    inputs: Any
    """The input data for the node."""

    fork_stack: ForkStack
    """Stack of forks that have been entered.

    Used by the GraphRun to decide when to proceed through joins.
    """

    task_id: TaskID = field(default_factory=lambda: TaskID(str(uuid.uuid4())))
    """Unique identifier for this task."""


class GraphRun(Generic[StateT, DepsT, OutputT]):
    """A single execution instance of a graph.

    GraphRun manages the execution state for a single run of a graph,
    including task scheduling, fork/join coordination, and result tracking.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies
        OutputT: The type of the output data
    """

    def __init__(
        self,
        graph: Graph[StateT, DepsT, InputT, OutputT],
        *,
        state: StateT,
        deps: DepsT,
        inputs: InputT,
        traceparent: str | None,
    ):
        """Initialize a graph run.

        Args:
            graph: The graph to execute
            state: The graph state instance
            deps: The dependencies instance
            inputs: The input data for the graph
            traceparent: Optional trace parent for instrumentation
        """
        self.graph = graph
        """The graph being executed."""

        self.state = state
        """The graph state instance."""

        self.deps = deps
        """The dependencies instance."""

        self.inputs = inputs
        """The initial input data."""

        self._active_reducers: dict[tuple[JoinID, NodeRunID], JoinState] = {}
        """Active reducers for join operations."""

        self._next: EndMarker[OutputT] | JoinItem | Sequence[GraphTask] | None = None
        """The next item to be processed."""

        run_id = GraphRunID(str(uuid.uuid4()))
        initial_fork_stack: ForkStack = (ForkStackItem(StartNode.id, NodeRunID(run_id), 0),)
        self._first_task = GraphTask(node_id=StartNode.id, inputs=inputs, fork_stack=initial_fork_stack)
        self._iterator = self._iter_graph()

        self.__traceparent = traceparent

    @overload
    def _traceparent(self, *, required: Literal[False]) -> str | None: ...
    @overload
    def _traceparent(self) -> str: ...
    def _traceparent(self, *, required: bool = True) -> str | None:
        """Get the trace parent for instrumentation.

        Args:
            required: Whether to raise an error if no traceparent exists

        Returns:
            The traceparent string, or None if not required and not set

        Raises:
            GraphRuntimeError: If required is True and no traceparent exists
        """
        if self.__traceparent is None and required:  # pragma: no cover
            raise exceptions.GraphRuntimeError('No span was created for this graph run')
        return self.__traceparent

    def __aiter__(self) -> AsyncIterator[EndMarker[OutputT] | JoinItem | Sequence[GraphTask]]:
        """Return self as an async iterator.

        Returns:
            Self for async iteration
        """
        return self

    async def __anext__(self) -> EndMarker[OutputT] | JoinItem | Sequence[GraphTask]:
        """Get the next item in the async iteration.

        Returns:
            The next execution result from the graph
        """
        if self._next is None:
            self._next = await self._iterator.__anext__()
        else:
            self._next = await self._iterator.asend(self._next)
        return self._next

    async def next(
        self, value: EndMarker[OutputT] | JoinItem | Sequence[GraphTask] | None = None
    ) -> EndMarker[OutputT] | JoinItem | Sequence[GraphTask]:
        """Advance the graph execution by one step.

        This method allows for sending a value to the iterator, which is useful
        for resuming iteration or overriding intermediate results.

        Args:
            value: Optional value to send to the iterator

        Returns:
            The next execution result: either an EndMarker, JoinItem, or sequence of GraphTasks
        """
        if self._next is None:
            # Prevent `TypeError: can't send non-None value to a just-started async generator`
            # if `next` is called before the `first_node` has run.
            await self.__anext__()
        if value is not None:
            self._next = value
        return await self.__anext__()

    @property
    def next_task(self) -> EndMarker[OutputT] | JoinItem | Sequence[GraphTask]:
        """Get the next task(s) to be executed.

        Returns:
            The next execution item, or the initial task if none is set
        """
        return self._next or [self._first_task]

    @property
    def output(self) -> OutputT | None:
        """Get the final output if the graph has completed.

        Returns:
            The output value if execution is complete, None otherwise
        """
        if isinstance(self._next, EndMarker):
            return self._next.value
        return None

    async def _iter_graph(  # noqa C901
        self,
    ) -> AsyncGenerator[
        EndMarker[OutputT] | JoinItem | Sequence[GraphTask], EndMarker[OutputT] | JoinItem | Sequence[GraphTask]
    ]:
        tasks_by_id: dict[TaskID, GraphTask] = {}
        pending: set[asyncio.Task[EndMarker[OutputT] | JoinItem | Sequence[GraphTask]]] = set()

        def _start_task(t_: GraphTask) -> None:
            """Helper function to start a new task while doing all necessary tracking."""
            tasks_by_id[t_.task_id] = t_
            task = asyncio.create_task(self._handle_task(t_))
            # Temporal insists on modifying the `name` passed to `create_task`, causing our `task.get_name()`-based lookup further down to fail,
            # so we set it explicitly after creation.
            # https://github.com/temporalio/sdk-python/blob/3fe7e422b008bcb8cd94e985f18ebec2de70e8e6/temporalio/worker/_workflow_instance.py#L2143
            task.set_name(t_.task_id)
            pending.add(task)

        _start_task(self._first_task)

        def _handle_result(result: EndMarker[OutputT] | JoinItem | Sequence[GraphTask]) -> bool:
            if isinstance(result, EndMarker):
                for t in pending:
                    t.cancel()
                return True

            if isinstance(result, JoinItem):
                parent_fork_id = self.graph.get_parent_fork(result.join_id).fork_id
                for i, x in enumerate(result.fork_stack[::-1]):
                    if x.fork_id == parent_fork_id:
                        downstream_fork_stack = result.fork_stack[: len(result.fork_stack) - i]
                        fork_run_id = x.node_run_id
                        break
                else:  # pragma: no cover
                    raise RuntimeError('Parent fork run not found')

                join_node = self.graph.nodes[result.join_id]
                assert isinstance(join_node, Join), f'Expected a `Join` but got {join_node}'
                join_state = self._active_reducers.get((result.join_id, fork_run_id))
                if join_state is None:
                    current = join_node.initial_factory()
                    join_state = self._active_reducers[(result.join_id, fork_run_id)] = JoinState(
                        current, downstream_fork_stack
                    )

                context = ReducerContext(state=self.state, deps=self.deps, join_state=join_state)
                join_state.current = join_node.reduce(context, join_state.current, result.inputs)
                if join_state.cancelled_sibling_tasks:
                    # cancel all concurrently running tasks with the same fork_run_id of the parent fork
                    task_ids_to_cancel = set[TaskID]()
                    for task_id, t in tasks_by_id.items():
                        for item in t.fork_stack:  # pragma: no branch
                            if item.fork_id == parent_fork_id and item.node_run_id == fork_run_id:
                                task_ids_to_cancel.add(task_id)
                                break
                            else:
                                pass
                    for task in list(pending):
                        if task.get_name() in task_ids_to_cancel:
                            task.cancel()
                            pending.remove(task)
            else:
                for new_task in result:
                    _start_task(new_task)
            return False

        while pending or self._active_reducers:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task_result = task.result()
                    source_task = tasks_by_id.pop(TaskID(task.get_name()))
                    maybe_overridden_result = yield task_result
                    if _handle_result(maybe_overridden_result):
                        return

                    for join_id, fork_run_id in self._get_completed_fork_runs(source_task, tasks_by_id.values()):
                        join_state = self._active_reducers.pop((join_id, fork_run_id))
                        join_node = self.graph.nodes[join_id]
                        assert isinstance(join_node, Join), f'Expected a `Join` but got {join_node}'
                        new_tasks = self._handle_edges(join_node, join_state.current, join_state.downstream_fork_stack)
                        maybe_overridden_result = yield new_tasks  # give an opportunity to override these
                        if _handle_result(maybe_overridden_result):
                            return  # pragma: no cover  # TODO: We should cover this

            if self._active_reducers:  # pragma: no branch
                # In this case, there are no pending tasks. We can therefore finalize all active reducers whose
                # downstream fork stacks are not a strict "prefix" of another active reducer. (If it was, finalizing the
                # deeper reducer could produce new tasks in the "prefix" reducer.)
                active_fork_stacks = [join_state.downstream_fork_stack for join_state in self._active_reducers.values()]
                for (join_id, fork_run_id), join_state in list(self._active_reducers.items()):
                    fork_stack = join_state.downstream_fork_stack
                    if any(
                        len(afs) > len(fork_stack) and fork_stack == afs[: len(fork_stack)]
                        for afs in active_fork_stacks
                    ):
                        # this join_state is a strict prefix for one of the other active join_states
                        continue  # pragma: no cover  # TODO: We should cover this
                    self._active_reducers.pop((join_id, fork_run_id))  # we're handling it now, so we can pop it
                    join_node = self.graph.nodes[join_id]
                    assert isinstance(join_node, Join), f'Expected a `Join` but got {join_node}'
                    new_tasks = self._handle_edges(join_node, join_state.current, join_state.downstream_fork_stack)
                    maybe_overridden_result = yield new_tasks  # give an opportunity to override these
                    if _handle_result(maybe_overridden_result):
                        return  # pragma: no cover  # TODO: We should cover this

        raise RuntimeError(  # pragma: no cover
            'Graph run completed, but no result was produced. This is either a bug in the graph or a bug in the graph runner.'
        )

    async def _handle_task(
        self,
        task: GraphTask,
    ) -> Sequence[GraphTask] | JoinItem | EndMarker[OutputT]:
        state = self.state
        deps = self.deps

        node_id = task.node_id
        inputs = task.inputs
        fork_stack = task.fork_stack

        node = self.graph.nodes[node_id]
        if isinstance(node, StartNode | Fork):
            return self._handle_edges(node, inputs, fork_stack)
        elif isinstance(node, Step):
            with ExitStack() as stack:
                if self.graph.auto_instrument:
                    stack.enter_context(logfire_span('run node {node_id}', node_id=node.id, node=node))

                step_context = StepContext[StateT, DepsT, Any](state=state, deps=deps, inputs=inputs)
                output = await node.call(step_context)
            if isinstance(node, NodeStep):
                return self._handle_node(output, fork_stack)
            else:
                return self._handle_edges(node, output, fork_stack)
        elif isinstance(node, Join):
            return JoinItem(node_id, inputs, fork_stack)
        elif isinstance(node, Decision):
            return self._handle_decision(node, inputs, fork_stack)
        elif isinstance(node, EndNode):
            return EndMarker(inputs)
        else:
            assert_never(node)

    def _handle_decision(
        self, decision: Decision[StateT, DepsT, Any], inputs: Any, fork_stack: ForkStack
    ) -> Sequence[GraphTask]:
        for branch in decision.branches:
            match_tester = branch.matches
            if match_tester is not None:
                inputs_match = match_tester(inputs)
            else:
                branch_source = unpack_type_expression(branch.source)

                if branch_source in {Any, object}:
                    inputs_match = True
                elif get_origin(branch_source) is Literal:
                    inputs_match = inputs in get_args(branch_source)
                else:
                    try:
                        inputs_match = isinstance(inputs, branch_source)
                    except TypeError as e:  # pragma: no cover
                        raise RuntimeError(f'Decision branch source {branch_source} is not a valid type.') from e

            if inputs_match:
                return self._handle_path(branch.path, inputs, fork_stack)

        raise RuntimeError(f'No branch matched inputs {inputs} for decision node {decision}.')

    def _handle_node(
        self,
        next_node: BaseNode[StateT, DepsT, Any] | End[Any],
        fork_stack: ForkStack,
    ) -> Sequence[GraphTask] | JoinItem | EndMarker[OutputT]:
        if isinstance(next_node, StepNode):
            return [GraphTask(next_node.step.id, next_node.inputs, fork_stack)]
        elif isinstance(next_node, JoinNode):
            return JoinItem(next_node.join.id, next_node.inputs, fork_stack)
        elif isinstance(next_node, BaseNode):
            node_step = NodeStep(next_node.__class__)
            return [GraphTask(node_step.id, next_node, fork_stack)]
        elif isinstance(next_node, End):
            return EndMarker(next_node.data)
        else:
            assert_never(next_node)

    def _get_completed_fork_runs(
        self,
        t: GraphTask,
        active_tasks: Iterable[GraphTask],
    ) -> list[tuple[JoinID, NodeRunID]]:
        completed_fork_runs: list[tuple[JoinID, NodeRunID]] = []

        fork_run_indices = {fsi.node_run_id: i for i, fsi in enumerate(t.fork_stack)}
        for join_id, fork_run_id in self._active_reducers.keys():
            fork_run_index = fork_run_indices.get(fork_run_id)
            if fork_run_index is None:
                continue  # The fork_run_id is not in the current task's fork stack, so this task didn't complete it.

            # This reducer _may_ now be ready to finalize:
            if self._is_fork_run_completed(active_tasks, join_id, fork_run_id):
                completed_fork_runs.append((join_id, fork_run_id))

        return completed_fork_runs

    def _handle_path(self, path: Path, inputs: Any, fork_stack: ForkStack) -> Sequence[GraphTask]:
        if not path.items:
            return []  # pragma: no cover

        item = path.items[0]
        assert not isinstance(item, MapMarker | BroadcastMarker), (
            'These markers should be removed from paths during graph building'
        )
        if isinstance(item, DestinationMarker):
            return [GraphTask(item.destination_id, inputs, fork_stack)]
        elif isinstance(item, TransformMarker):
            inputs = item.transform(StepContext(state=self.state, deps=self.deps, inputs=inputs))
            return self._handle_path(path.next_path, inputs, fork_stack)
        elif isinstance(item, LabelMarker):
            return self._handle_path(path.next_path, inputs, fork_stack)
        else:
            assert_never(item)

    def _handle_edges(self, node: AnyNode, inputs: Any, fork_stack: ForkStack) -> Sequence[GraphTask]:
        edges = self.graph.edges_by_source.get(node.id, [])
        assert len(edges) == 1 or (isinstance(node, Fork) and not node.is_map), (
            edges,
            node.id,
        )  # this should have already been ensured during graph building

        new_tasks: list[GraphTask] = []

        if isinstance(node, Fork):
            node_run_id = NodeRunID(str(uuid.uuid4()))
            if node.is_map:
                # Eagerly raise a clear error if the input value is not iterable as expected
                try:
                    iter(inputs)
                except TypeError:
                    raise RuntimeError(f'Cannot map non-iterable value: {inputs!r}')

                # If the map specifies a downstream join id, eagerly create a join state for it
                if (join_id := node.downstream_join_id) is not None:
                    join_node = self.graph.nodes[join_id]
                    assert isinstance(join_node, Join)
                    self._active_reducers[(join_id, node_run_id)] = JoinState(join_node.initial_factory(), fork_stack)

                for thread_index, input_item in enumerate(inputs):
                    item_tasks = self._handle_path(
                        edges[0], input_item, fork_stack + (ForkStackItem(node.id, node_run_id, thread_index),)
                    )
                    new_tasks += item_tasks
            else:
                for i, path in enumerate(edges):
                    new_tasks += self._handle_path(path, inputs, fork_stack + (ForkStackItem(node.id, node_run_id, i),))
        else:
            new_tasks += self._handle_path(edges[0], inputs, fork_stack)

        return new_tasks

    def _is_fork_run_completed(self, tasks: Iterable[GraphTask], join_id: JoinID, fork_run_id: NodeRunID) -> bool:
        # Check if any of the tasks in the graph have this fork_run_id in their fork_stack
        # If this is the case, then the fork run is not yet completed
        parent_fork = self.graph.get_parent_fork(join_id)
        for t in tasks:
            if fork_run_id in {x.node_run_id for x in t.fork_stack}:
                if t.node_id in parent_fork.intermediate_nodes or t.node_id == join_id:
                    return False
            else:
                pass
        return True
