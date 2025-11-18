"""Core graph execution engine for the next version of the pydantic-graph library.

This module provides the main `Graph` class and `GraphRun` execution engine that
handles the orchestration of nodes, edges, and parallel execution paths in
the graph-based workflow system.
"""

from __future__ import annotations as _annotations

import sys
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Callable, Iterable, Sequence
from contextlib import AbstractContextManager, AsyncExitStack, ExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeGuard, cast, get_args, get_origin, overload

from anyio import CancelScope, create_memory_object_stream, create_task_group
from anyio.abc import TaskGroup
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from typing_extensions import TypeVar, assert_never

from pydantic_graph import exceptions
from pydantic_graph._utils import AbstractSpan, get_traceparent, infer_obj_name, logfire_span
from pydantic_graph.beta.decision import Decision
from pydantic_graph.beta.id_types import ForkID, ForkStack, ForkStackItem, JoinID, NodeID, NodeRunID, TaskID
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

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup as ExceptionGroup  # pragma: lax no cover
else:
    ExceptionGroup = ExceptionGroup  # pragma: lax no cover

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

    intermediate_join_nodes: dict[JoinID, set[JoinID]]
    """For each join, the set of other joins that appear between it and its parent fork.

    Used to determine which joins are "final" (have no other joins as intermediates) and
    which joins should preserve fork stacks when proceeding downstream."""

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

    def is_final_join(self, join_id: JoinID) -> bool:
        """Check if a join is 'final' (has no downstream joins with the same parent fork).

        A join is non-final if it appears as an intermediate node for another join
        with the same parent fork.

        Args:
            join_id: The ID of the join node

        Returns:
            True if the join is final, False if it's non-final
        """
        # Check if this join appears in any other join's intermediate_join_nodes
        for intermediate_joins in self.intermediate_join_nodes.values():
            if join_id in intermediate_joins:
                return False
        return True

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
            inferred_name = infer_obj_name(self, depth=2)
            if inferred_name is not None:  # pragma: no branch
                self.name = inferred_name

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
            inferred_name = infer_obj_name(self, depth=3)  # depth=3 because asynccontextmanager adds one
            if inferred_name is not None:  # pragma: no branch
                self.name = inferred_name

        with ExitStack() as stack:
            entered_span: AbstractSpan | None = None
            if span is None:
                if self.auto_instrument:
                    entered_span = stack.enter_context(logfire_span('run graph {graph.name}', graph=self))
            else:
                entered_span = stack.enter_context(span)
            traceparent = None if entered_span is None else get_traceparent(entered_span)
            async with GraphRun[StateT, DepsT, OutputT](
                graph=self,
                state=state,
                deps=deps,
                inputs=inputs,
                traceparent=traceparent,
            ) as graph_run:
                yield graph_run

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


@dataclass
class GraphTaskRequest:
    """A request to run a task representing the execution of a node in the graph.

    GraphTaskRequest encapsulates all the information needed to execute a specific
    node, including its inputs and the fork context it's executing within.
    """

    node_id: NodeID
    """The ID of the node to execute."""

    inputs: Any
    """The input data for the node."""

    fork_stack: ForkStack = field(repr=False)
    """Stack of forks that have been entered.

    Used by the GraphRun to decide when to proceed through joins.
    """


@dataclass
class GraphTask(GraphTaskRequest):
    """A task representing the execution of a node in the graph.

    GraphTask encapsulates all the information needed to execute a specific
    node, including its inputs and the fork context it's executing within,
    and has a unique ID to identify the task within the graph run.
    """

    task_id: TaskID = field(repr=False)
    """Unique identifier for this task."""

    @staticmethod
    def from_request(request: GraphTaskRequest, get_task_id: Callable[[], TaskID]) -> GraphTask:
        # Don't call the get_task_id callable, this is already a task
        if isinstance(request, GraphTask):
            return request
        return GraphTask(request.node_id, request.inputs, request.fork_stack, get_task_id())


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

        self._next: EndMarker[OutputT] | Sequence[GraphTask] | None = None
        """The next item to be processed."""

        self._next_task_id = 0
        self._next_node_run_id = 0
        initial_fork_stack: ForkStack = (ForkStackItem(StartNode.id, self._get_next_node_run_id(), 0),)
        self._first_task = GraphTask(
            node_id=StartNode.id, inputs=inputs, fork_stack=initial_fork_stack, task_id=self._get_next_task_id()
        )
        self._iterator_task_group = create_task_group()
        self._iterator_instance = _GraphIterator[StateT, DepsT, OutputT](
            self.graph,
            self.state,
            self.deps,
            self._iterator_task_group,
            self._get_next_node_run_id,
            self._get_next_task_id,
        )
        self._iterator = self._iterator_instance.iter_graph(self._first_task)

        self.__traceparent = traceparent
        self._async_exit_stack = AsyncExitStack()

    async def __aenter__(self):
        self._async_exit_stack.enter_context(_unwrap_exception_groups())
        await self._async_exit_stack.enter_async_context(self._iterator_task_group)
        await self._async_exit_stack.enter_async_context(self._iterator_context())
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        await self._async_exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    @asynccontextmanager
    async def _iterator_context(self):
        try:
            yield
        finally:
            self._iterator_instance.iter_stream_sender.close()
            self._iterator_instance.iter_stream_receiver.close()
            await self._iterator.aclose()

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

    def __aiter__(self) -> AsyncIterator[EndMarker[OutputT] | Sequence[GraphTask]]:
        """Return self as an async iterator.

        Returns:
            Self for async iteration
        """
        return self

    async def __anext__(self) -> EndMarker[OutputT] | Sequence[GraphTask]:
        """Get the next item in the async iteration.

        Returns:
            The next execution result from the graph
        """
        if self._next is None:
            self._next = await anext(self._iterator)
        else:
            self._next = await self._iterator.asend(self._next)
        return self._next

    async def next(
        self, value: EndMarker[OutputT] | Sequence[GraphTaskRequest] | None = None
    ) -> EndMarker[OutputT] | Sequence[GraphTask]:
        """Advance the graph execution by one step.

        This method allows for sending a value to the iterator, which is useful
        for resuming iteration or overriding intermediate results.

        Args:
            value: Optional value to send to the iterator

        Returns:
            The next execution result: either an EndMarker, or sequence of GraphTasks
        """
        if self._next is None:
            # Prevent `TypeError: can't send non-None value to a just-started async generator`
            # if `next` is called before the `first_node` has run.
            await anext(self)
        if value is not None:
            if isinstance(value, EndMarker):
                self._next = value
            else:
                self._next = [GraphTask.from_request(gtr, self._get_next_task_id) for gtr in value]
        return await anext(self)

    @property
    def next_task(self) -> EndMarker[OutputT] | Sequence[GraphTask]:
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

    def _get_next_task_id(self) -> TaskID:
        next_id = TaskID(f'task:{self._next_task_id}')
        self._next_task_id += 1
        return next_id

    def _get_next_node_run_id(self) -> NodeRunID:
        next_id = NodeRunID(f'task:{self._next_node_run_id}')
        self._next_node_run_id += 1
        return next_id


@dataclass
class _GraphTaskAsyncIterable:
    iterable: AsyncIterable[Sequence[GraphTask]]
    fork_stack: ForkStack


@dataclass
class _GraphTaskResult:
    source: GraphTask
    result: EndMarker[Any] | Sequence[GraphTask] | JoinItem
    source_is_finished: bool = True


@dataclass
class _GraphIterator(Generic[StateT, DepsT, OutputT]):
    graph: Graph[StateT, DepsT, Any, OutputT]
    state: StateT
    deps: DepsT
    task_group: TaskGroup
    get_next_node_run_id: Callable[[], NodeRunID]
    get_next_task_id: Callable[[], TaskID]

    cancel_scopes: dict[TaskID, CancelScope] = field(init=False)
    active_tasks: dict[TaskID, GraphTask] = field(init=False)
    active_reducers: dict[tuple[JoinID, NodeRunID], JoinState] = field(init=False)
    iter_stream_sender: MemoryObjectSendStream[_GraphTaskResult] = field(init=False)
    iter_stream_receiver: MemoryObjectReceiveStream[_GraphTaskResult] = field(init=False)

    def __post_init__(self):
        self.cancel_scopes = {}
        self.active_tasks = {}
        self.active_reducers = {}
        self.iter_stream_sender, self.iter_stream_receiver = create_memory_object_stream[_GraphTaskResult]()
        self._next_node_run_id = 1

    async def iter_graph(  # noqa C901
        self, first_task: GraphTask
    ) -> AsyncGenerator[EndMarker[OutputT] | Sequence[GraphTask], EndMarker[OutputT] | Sequence[GraphTask]]:
        async with self.iter_stream_sender:
            try:
                # Fire off the first task
                self.active_tasks[first_task.task_id] = first_task
                self._handle_execution_request([first_task])

                # Handle task results
                async with self.iter_stream_receiver:
                    while self.active_tasks or self.active_reducers:
                        async for task_result in self.iter_stream_receiver:  # pragma: no branch
                            if isinstance(task_result.result, JoinItem):
                                maybe_overridden_result = task_result.result
                            else:
                                maybe_overridden_result = yield task_result.result
                            if isinstance(maybe_overridden_result, EndMarker):
                                # If we got an end marker, this task is definitely done, and we're ready to
                                # start cleaning everything up
                                await self._finish_task(task_result.source.task_id)
                                if self.active_tasks:
                                    # Cancel the remaining tasks
                                    self.task_group.cancel_scope.cancel()
                                return
                            elif isinstance(maybe_overridden_result, JoinItem):
                                result = maybe_overridden_result
                                parent_fork_id = self.graph.get_parent_fork(result.join_id).fork_id
                                for i, x in enumerate(result.fork_stack[::-1]):
                                    if x.fork_id == parent_fork_id:
                                        # For non-final joins (those that are intermediate nodes of other joins),
                                        # preserve the fork stack so downstream joins can still associate with the same fork run
                                        if self.graph.is_final_join(result.join_id):
                                            # Final join: remove the parent fork from the stack
                                            downstream_fork_stack = result.fork_stack[: len(result.fork_stack) - i]
                                        else:
                                            # Non-final join: preserve the fork stack
                                            downstream_fork_stack = result.fork_stack
                                        fork_run_id = x.node_run_id
                                        break
                                else:  # pragma: no cover
                                    raise RuntimeError('Parent fork run not found')

                                join_node = self.graph.nodes[result.join_id]
                                assert isinstance(join_node, Join), f'Expected a `Join` but got {join_node}'
                                join_state = self.active_reducers.get((result.join_id, fork_run_id))
                                if join_state is None:
                                    current = join_node.initial_factory()
                                    join_state = self.active_reducers[(result.join_id, fork_run_id)] = JoinState(
                                        current, downstream_fork_stack
                                    )
                                context = ReducerContext(state=self.state, deps=self.deps, join_state=join_state)
                                join_state.current = join_node.reduce(context, join_state.current, result.inputs)
                                if join_state.cancelled_sibling_tasks:
                                    await self._cancel_sibling_tasks(parent_fork_id, fork_run_id)
                            else:
                                for new_task in maybe_overridden_result:
                                    self.active_tasks[new_task.task_id] = new_task

                            tasks_by_id_values = list(self.active_tasks.values())
                            join_tasks: list[GraphTask] = []

                            for join_id, fork_run_id in self._get_completed_fork_runs(
                                task_result.source, tasks_by_id_values
                            ):
                                join_state = self.active_reducers.pop((join_id, fork_run_id))
                                join_node = self.graph.nodes[join_id]
                                assert isinstance(join_node, Join), f'Expected a `Join` but got {join_node}'
                                new_tasks = self._handle_non_fork_edges(
                                    join_node, join_state.current, join_state.downstream_fork_stack
                                )
                                join_tasks.extend(new_tasks)
                            if join_tasks:
                                for new_task in join_tasks:
                                    self.active_tasks[new_task.task_id] = new_task
                                self._handle_execution_request(join_tasks)

                            if isinstance(maybe_overridden_result, Sequence):
                                if isinstance(task_result.result, Sequence):
                                    new_task_ids = {t.task_id for t in maybe_overridden_result}
                                    for t in task_result.result:
                                        if t.task_id not in new_task_ids:
                                            await self._finish_task(t.task_id)
                                self._handle_execution_request(maybe_overridden_result)

                            if task_result.source_is_finished:
                                await self._finish_task(task_result.source.task_id)

                            if not self.active_tasks:
                                # if there are no active tasks, we'll be waiting forever for the next result..
                                break

                        if self.active_reducers:  # pragma: no branch
                            # In this case, there are no pending tasks. We can therefore finalize all active reducers
                            # that don't have intermediate joins which are also active reducers. If a join J2 has an
                            # intermediate join J1 that shares the same parent fork run, we must finalize J1 first
                            # because it might produce items that feed into J2.
                            for (join_id, fork_run_id), join_state in list(self.active_reducers.items()):
                                # Check if this join has any intermediate joins that are also active reducers
                                should_skip = False
                                intermediate_joins = self.graph.intermediate_join_nodes.get(join_id, set())

                                # Get the parent fork for this join to use for comparison
                                join_parent_fork = self.graph.get_parent_fork(join_id)

                                for intermediate_join_id in intermediate_joins:
                                    # Check if the intermediate join is also an active reducer with matching fork run
                                    for (other_join_id, _), other_join_state in self.active_reducers.items():
                                        if other_join_id == intermediate_join_id:
                                            # Check if they share the same fork run for this join's parent fork
                                            # by finding the parent fork's node_run_id in both fork stacks
                                            join_parent_fork_run_id = None
                                            other_parent_fork_run_id = None

                                            for fsi in join_state.downstream_fork_stack:  # pragma: no branch
                                                if fsi.fork_id == join_parent_fork.fork_id:
                                                    join_parent_fork_run_id = fsi.node_run_id
                                                    break

                                            for fsi in other_join_state.downstream_fork_stack:  # pragma: no branch
                                                if fsi.fork_id == join_parent_fork.fork_id:
                                                    other_parent_fork_run_id = fsi.node_run_id
                                                    break

                                            if (
                                                join_parent_fork_run_id
                                                and other_parent_fork_run_id
                                                and join_parent_fork_run_id == other_parent_fork_run_id
                                            ):  # pragma: no branch
                                                should_skip = True
                                                break
                                    if should_skip:
                                        break

                                if should_skip:
                                    continue

                                self.active_reducers.pop(
                                    (join_id, fork_run_id)
                                )  # we're handling it now, so we can pop it
                                join_node = self.graph.nodes[join_id]
                                assert isinstance(join_node, Join), f'Expected a `Join` but got {join_node}'
                                new_tasks = self._handle_non_fork_edges(
                                    join_node, join_state.current, join_state.downstream_fork_stack
                                )
                                maybe_overridden_result = yield new_tasks
                                if isinstance(maybe_overridden_result, EndMarker):  # pragma: no cover
                                    # This is theoretically reachable but it would be awkward.
                                    # Probably a better way to get coverage here would be to unify the code pat
                                    # with the other `if isinstance(maybe_overridden_result, EndMarker):`
                                    self.task_group.cancel_scope.cancel()
                                    return
                                for new_task in maybe_overridden_result:
                                    self.active_tasks[new_task.task_id] = new_task
                                new_task_ids = {t.task_id for t in maybe_overridden_result}
                                for t in new_tasks:
                                    # Same note as above about how this is theoretically reachable but we should
                                    # just get coverage by unifying the code paths
                                    if t.task_id not in new_task_ids:  # pragma: no cover
                                        await self._finish_task(t.task_id)
                                self._handle_execution_request(maybe_overridden_result)
            except GeneratorExit:
                self.task_group.cancel_scope.cancel()
                return

        raise RuntimeError(  # pragma: no cover
            'Graph run completed, but no result was produced. This is either a bug in the graph or a bug in the graph runner.'
        )

    async def _finish_task(self, task_id: TaskID) -> None:
        # node_id is just included for debugging right now
        scope = self.cancel_scopes.pop(task_id, None)
        if scope is not None:
            scope.cancel()
        self.active_tasks.pop(task_id, None)

    def _handle_execution_request(self, request: Sequence[GraphTask]) -> None:
        for new_task in request:
            self.active_tasks[new_task.task_id] = new_task
        for new_task in request:
            self.task_group.start_soon(self._run_tracked_task, new_task)

    async def _run_tracked_task(self, t_: GraphTask):
        with CancelScope() as scope:
            self.cancel_scopes[t_.task_id] = scope
            result = await self._run_task(t_)
            if isinstance(result, _GraphTaskAsyncIterable):
                async for new_tasks in result.iterable:
                    await self.iter_stream_sender.send(_GraphTaskResult(t_, new_tasks, False))
                await self.iter_stream_sender.send(_GraphTaskResult(t_, []))
            else:
                await self.iter_stream_sender.send(_GraphTaskResult(t_, result))

    async def _run_task(
        self,
        task: GraphTask,
    ) -> EndMarker[OutputT] | Sequence[GraphTask] | _GraphTaskAsyncIterable | JoinItem:
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
            return [GraphTask(next_node.step.id, next_node.inputs, fork_stack, self.get_next_task_id())]
        elif isinstance(next_node, JoinNode):
            return JoinItem(next_node.join.id, next_node.inputs, fork_stack)
        elif isinstance(next_node, BaseNode):
            node_step = NodeStep(next_node.__class__)
            return [GraphTask(node_step.id, next_node, fork_stack, self.get_next_task_id())]
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
        for join_id, fork_run_id in self.active_reducers.keys():
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
            return [GraphTask(item.destination_id, inputs, fork_stack, self.get_next_task_id())]
        elif isinstance(item, TransformMarker):
            inputs = item.transform(StepContext(state=self.state, deps=self.deps, inputs=inputs))
            return self._handle_path(path.next_path, inputs, fork_stack)
        elif isinstance(item, LabelMarker):
            return self._handle_path(path.next_path, inputs, fork_stack)
        else:
            assert_never(item)

    def _handle_edges(
        self, node: AnyNode, inputs: Any, fork_stack: ForkStack
    ) -> Sequence[GraphTask] | _GraphTaskAsyncIterable:
        if isinstance(node, Fork):
            return self._handle_fork_edges(node, inputs, fork_stack)
        else:
            return self._handle_non_fork_edges(node, inputs, fork_stack)

    def _handle_non_fork_edges(self, node: AnyNode, inputs: Any, fork_stack: ForkStack) -> Sequence[GraphTask]:
        edges = self.graph.edges_by_source.get(node.id, [])
        assert len(edges) == 1  # this should have already been ensured during graph building
        return self._handle_path(edges[0], inputs, fork_stack)

    def _handle_fork_edges(
        self, node: Fork[Any, Any], inputs: Any, fork_stack: ForkStack
    ) -> Sequence[GraphTask] | _GraphTaskAsyncIterable:
        edges = self.graph.edges_by_source.get(node.id, [])
        assert len(edges) == 1 or (isinstance(node, Fork) and not node.is_map), (
            edges,
            node.id,
        )  # this should have already been ensured during graph building

        new_tasks: list[GraphTask] = []
        node_run_id = self.get_next_node_run_id()
        if node.is_map:
            # If the map specifies a downstream join id, eagerly create a join state for it
            if (join_id := node.downstream_join_id) is not None:
                join_node = self.graph.nodes[join_id]
                assert isinstance(join_node, Join)
                self.active_reducers[(join_id, node_run_id)] = JoinState(join_node.initial_factory(), fork_stack)

            # Eagerly raise a clear error if the input value is not iterable as expected
            if _is_any_iterable(inputs):
                for thread_index, input_item in enumerate(inputs):
                    item_tasks = self._handle_path(
                        edges[0], input_item, fork_stack + (ForkStackItem(node.id, node_run_id, thread_index),)
                    )
                    new_tasks += item_tasks
            elif _is_any_async_iterable(inputs):

                async def handle_async_iterable() -> AsyncIterator[Sequence[GraphTask]]:
                    thread_index = 0
                    async for input_item in inputs:
                        item_tasks = self._handle_path(
                            edges[0], input_item, fork_stack + (ForkStackItem(node.id, node_run_id, thread_index),)
                        )
                        yield item_tasks
                        thread_index += 1

                return _GraphTaskAsyncIterable(handle_async_iterable(), fork_stack)

            else:
                raise RuntimeError(f'Cannot map non-iterable value: {inputs!r}')
        else:
            for i, path in enumerate(edges):
                new_tasks += self._handle_path(path, inputs, fork_stack + (ForkStackItem(node.id, node_run_id, i),))
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

    async def _cancel_sibling_tasks(self, parent_fork_id: ForkID, node_run_id: NodeRunID):
        task_ids_to_cancel = set[TaskID]()
        for task_id, t in self.active_tasks.items():
            for item in t.fork_stack:  # pragma: no branch
                if item.fork_id == parent_fork_id and item.node_run_id == node_run_id:
                    task_ids_to_cancel.add(task_id)
                    break
                else:
                    pass
        for task_id in task_ids_to_cancel:
            await self._finish_task(task_id)


def _is_any_iterable(x: Any) -> TypeGuard[Iterable[Any]]:
    return isinstance(x, Iterable)


def _is_any_async_iterable(x: Any) -> TypeGuard[AsyncIterable[Any]]:
    return isinstance(x, AsyncIterable)


@contextmanager
def _unwrap_exception_groups():
    # I need to use a helper function for this because I can't figure out a way to get pyright
    # to type-check the ExceptionGroup catching in both 3.13 and 3.10 without emitting type errors in one;
    # if I try to ignore them in one, I get unnecessary-type-ignore errors in the other
    if TYPE_CHECKING:
        yield
    else:
        try:
            yield
        except ExceptionGroup as e:
            exception = e.exceptions[0]
            if exception.__cause__ is None:
                # bizarrely, this prevents recursion errors when formatting the exception for logfire
                exception.__cause__ = None
            raise exception
