"""Builder-based graph API: builder, graph runner, and mermaid rendering.

This module is the canonical home for the builder-based graph API:
[`GraphBuilder`][pydantic_graph.graph_builder.GraphBuilder] for declaratively
constructing executable graphs, [`Graph`][pydantic_graph.graph_builder.Graph]
and [`GraphRun`][pydantic_graph.graph_builder.GraphRun] for executing them, and
the mermaid rendering helpers used by `Graph.render()`.

The same public symbols are re-exported from `pydantic_graph` directly. The
deprecated `pydantic_graph.beta.*` namespace also forwards here.

The contents of three previously-separate modules are bundled here because they
co-occupy the names (`graph`, `mermaid`) that the legacy `BaseNode`-based
runner currently holds at the top level. In v2, after the legacy runner is
removed, `Graph` and the mermaid helpers are expected to move out of this file
to `pydantic_graph.graph` and `pydantic_graph.mermaid` respectively.
"""

from __future__ import annotations as _annotations

import inspect
import sys
from collections import Counter, defaultdict
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Callable, Iterable, Sequence
from contextlib import AbstractContextManager, AsyncExitStack, ExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass, field, replace
from types import NoneType
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeGuard,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from anyio import BrokenResourceError, CancelScope, create_memory_object_stream, create_task_group
from anyio.abc import TaskGroup
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from typing_extensions import Never, TypeAliasType, TypeVar, assert_never

from pydantic_graph import _utils, exceptions
from pydantic_graph._utils import UNSET, AbstractSpan, Unset, get_traceparent, infer_obj_name, logfire_span
from pydantic_graph.basenode import BaseNode, End
from pydantic_graph.decision import Decision, DecisionBranch, DecisionBranchBuilder
from pydantic_graph.exceptions import GraphBuildingError, GraphValidationError
from pydantic_graph.id_types import (
    ForkID,
    ForkStack,
    ForkStackItem,
    JoinID,
    NodeID,
    NodeRunID,
    TaskID,
    generate_placeholder_node_id,
    replace_placeholder_id,
)
from pydantic_graph.join import Join, JoinNode, JoinState, ReducerContext, ReducerFunction
from pydantic_graph.node import EndNode, Fork, StartNode
from pydantic_graph.node_types import AnyDestinationNode, AnyNode, DestinationNode, SourceNode
from pydantic_graph.parent_forks import ParentFork, ParentForkFinder
from pydantic_graph.paths import (
    BroadcastMarker,
    DestinationMarker,
    EdgePath,
    EdgePathBuilder,
    LabelMarker,
    MapMarker,
    Path,
    PathBuilder,
    TransformMarker,
)
from pydantic_graph.step import NodeStep, Step, StepContext, StepFunction, StepNode, StreamFunction
from pydantic_graph.util import TypeOrTypeExpression, get_callable_name, unpack_type_expression

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup as BaseExceptionGroup  # pragma: lax no cover
else:
    BaseExceptionGroup = BaseExceptionGroup  # pragma: lax no cover


# -- TypeVars ----------------------------------------------------------------

StateT = TypeVar('StateT', infer_variance=True)
"""Type variable for graph state."""

DepsT = TypeVar('DepsT', infer_variance=True)
"""Type variable for graph dependencies."""

InputT = TypeVar('InputT', infer_variance=True)
"""Type variable for graph inputs."""

OutputT = TypeVar('OutputT', infer_variance=True)
"""Type variable for graph outputs."""

SourceT = TypeVar('SourceT', infer_variance=True)
SourceNodeT = TypeVar('SourceNodeT', bound=BaseNode[Any, Any, Any], infer_variance=True)
SourceOutputT = TypeVar('SourceOutputT', infer_variance=True)
GraphInputT = TypeVar('GraphInputT', infer_variance=True)
GraphOutputT = TypeVar('GraphOutputT', infer_variance=True)
T = TypeVar('T', infer_variance=True)


# === Graph runner ===


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
class ErrorMarker:
    """A marker indicating that a graph node raised an exception.

    Yielded by the graph iterator instead of raising immediately, allowing the caller
    to recover by sending new tasks via `GraphRun.next()` or `GraphRun.override_next()`.
    If the caller does not override, the error is re-raised on the next iteration.
    """

    error: BaseException
    """The exception raised by the node."""


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
    ) -> AsyncGenerator[GraphRun[StateT, DepsT, OutputT]]:
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
                entered_span = stack.enter_context(span)  # pragma: lax no cover
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

        self._next: EndMarker[OutputT] | ErrorMarker | Sequence[GraphTask] | None = None
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

        Raises:
            Exception: If a node raised an error and the caller has not recovered via
                `override_next()`.
        """
        if self._next is None:
            self._next = await anext(self._iterator)
        else:
            self._next = await self._iterator.asend(self._next)
        if isinstance(self._next, ErrorMarker):
            # A node raised an error. Store it so the caller can recover via
            # override_next() before the next __anext__ call re-raises.
            raise self._next.error
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
            self._set_next(value)
        return await anext(self)

    def override_next(self, value: Sequence[GraphTaskRequest] | EndMarker[OutputT]) -> None:
        """Override the next pending step, allowing the graph to continue after an `End` or error.

        This is used by hook systems (like `after_node_run` or `on_node_run_error`) to redirect
        the graph to a new node when the current step produced an `End` result or raised an error,
        or to signal early completion by passing an `EndMarker`.

        Must only be called between iterations (not while an iteration is in flight).

        Args:
            value: New task requests to execute next, or an `EndMarker` to signal completion.
        """
        self._set_next(value)

    def _set_next(self, value: Sequence[GraphTaskRequest] | EndMarker[OutputT]) -> None:
        if isinstance(value, EndMarker):
            self._next = value
        else:
            self._next = [GraphTask.from_request(gtr, self._get_next_task_id) for gtr in value]

    @property
    def next_task(self) -> EndMarker[OutputT] | ErrorMarker | Sequence[GraphTask]:
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
    error: BaseException | None = None


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

    async def iter_graph(  # noqa: C901
        self, first_task: GraphTask
    ) -> AsyncGenerator[
        EndMarker[OutputT] | ErrorMarker | Sequence[GraphTask], EndMarker[OutputT] | ErrorMarker | Sequence[GraphTask]
    ]:
        async with self.iter_stream_sender:
            try:
                # Fire off the first task
                self.active_tasks[first_task.task_id] = first_task
                self._handle_execution_request([first_task])

                # Handle task results
                async with self.iter_stream_receiver:
                    while self.active_tasks or self.active_reducers:
                        async for task_result in self.iter_stream_receiver:  # pragma: no branch
                            if task_result.error is not None:
                                # Yield ErrorMarker instead of raising, so the caller can
                                # recover via on_node_run_error by sending new tasks.
                                maybe_overridden_result = yield ErrorMarker(task_result.error)
                                if isinstance(maybe_overridden_result, ErrorMarker):
                                    # Caller echoed the error back — actually raise
                                    raise task_result.error
                                # Caller recovered by sending tasks or EndMarker
                            elif isinstance(task_result.result, JoinItem):
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
                                assert not isinstance(maybe_overridden_result, ErrorMarker)
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
                                assert not isinstance(maybe_overridden_result, ErrorMarker)
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

        raise RuntimeError(  # pragma: lax no cover
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
            try:
                result = await self._run_task(t_)
            except BaseException as exc:
                # Send the error through the stream instead of letting it propagate
                # into the task group, which would transform it (e.g. into CancelledError
                # or ExceptionGroup). This preserves the original exception for the caller.
                try:
                    await self.iter_stream_sender.send(_GraphTaskResult(t_, [], error=exc))
                except BrokenResourceError:
                    pass  # pragma: no cover
                return
            try:
                if isinstance(result, _GraphTaskAsyncIterable):
                    async for new_tasks in result.iterable:
                        await self.iter_stream_sender.send(_GraphTaskResult(t_, new_tasks, False))
                    await self.iter_stream_sender.send(_GraphTaskResult(t_, []))
                else:
                    await self.iter_stream_sender.send(_GraphTaskResult(t_, result))
            except BrokenResourceError:
                # Can happen when an asyncio task is cancelled mid-send.
                pass

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
        except BaseExceptionGroup as e:
            exception = e.exceptions[0]
            if exception.__cause__ is None:
                # bizarrely, this prevents recursion errors when formatting the exception for logfire
                exception.__cause__ = None
            raise exception


# === GraphBuilder ===


@dataclass(init=False)
class GraphBuilder(Generic[StateT, DepsT, GraphInputT, GraphOutputT]):
    """A builder for constructing executable graph definitions.

    GraphBuilder provides a fluent interface for defining nodes, edges, and
    routing in a graph workflow. It supports typed state, dependencies, and
    input/output validation.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies
        GraphInputT: The type of the graph input data
        GraphOutputT: The type of the graph output data
    """

    name: str | None
    """Optional name for the graph, if not provided the name will be inferred from the calling frame on the first call to a graph method."""

    state_type: TypeOrTypeExpression[StateT]
    """The type of the graph state."""

    deps_type: TypeOrTypeExpression[DepsT]
    """The type of the dependencies."""

    input_type: TypeOrTypeExpression[GraphInputT]
    """The type of the graph input data."""

    output_type: TypeOrTypeExpression[GraphOutputT]
    """The type of the graph output data."""

    auto_instrument: bool
    """Whether to automatically create instrumentation spans."""

    _nodes: dict[NodeID, AnyNode]
    """Internal storage for nodes in the graph."""

    _edges_by_source: dict[NodeID, list[Path]]
    """Internal storage for edges by source node."""

    _decision_index: int
    """Counter for generating unique decision node IDs."""

    Source = TypeAliasType('Source', SourceNode[StateT, DepsT, OutputT], type_params=(OutputT,))
    Destination = TypeAliasType('Destination', DestinationNode[StateT, DepsT, InputT], type_params=(InputT,))

    def __init__(
        self,
        *,
        name: str | None = None,
        state_type: TypeOrTypeExpression[StateT] = NoneType,
        deps_type: TypeOrTypeExpression[DepsT] = NoneType,
        input_type: TypeOrTypeExpression[GraphInputT] = NoneType,
        output_type: TypeOrTypeExpression[GraphOutputT] = NoneType,
        auto_instrument: bool = True,
    ):
        """Initialize a graph builder.

        Args:
            name: Optional name for the graph, if not provided the name will be inferred from the calling frame on the first call to a graph method.
            state_type: The type of the graph state
            deps_type: The type of the dependencies
            input_type: The type of the graph input data
            output_type: The type of the graph output data
            auto_instrument: Whether to automatically create instrumentation spans
        """
        self.name = name

        self.state_type = state_type
        self.deps_type = deps_type
        self.input_type = input_type
        self.output_type = output_type

        self.auto_instrument = auto_instrument

        self._nodes = {}
        self._edges_by_source = defaultdict(list)
        self._decision_index = 1

        self._start_node = StartNode[GraphInputT]()
        self._end_node = EndNode[GraphOutputT]()

    # Node building
    @property
    def start_node(self) -> StartNode[GraphInputT]:
        """Get the start node for the graph.

        Returns:
            The start node that receives the initial graph input
        """
        return self._start_node

    @property
    def end_node(self) -> EndNode[GraphOutputT]:
        """Get the end node for the graph.

        Returns:
            The end node that produces the final graph output
        """
        return self._end_node

    @overload
    def step(
        self,
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> Callable[[StepFunction[StateT, DepsT, InputT, OutputT]], Step[StateT, DepsT, InputT, OutputT]]: ...
    @overload
    def step(
        self,
        call: StepFunction[StateT, DepsT, InputT, OutputT],
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> Step[StateT, DepsT, InputT, OutputT]: ...
    def step(
        self,
        call: StepFunction[StateT, DepsT, InputT, OutputT] | None = None,
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> (
        Step[StateT, DepsT, InputT, OutputT]
        | Callable[[StepFunction[StateT, DepsT, InputT, OutputT]], Step[StateT, DepsT, InputT, OutputT]]
    ):
        """Create a step from a step function.

        This method can be used as a decorator or called directly to create
        a step node from an async function.

        Args:
            call: The step function to wrap
            node_id: Optional ID for the node
            label: Optional human-readable label

        Returns:
            Either a Step instance or a decorator function
        """
        if call is None:

            def decorator(
                func: StepFunction[StateT, DepsT, InputT, OutputT],
            ) -> Step[StateT, DepsT, InputT, OutputT]:
                return self.step(call=func, node_id=node_id, label=label)

            return decorator

        node_id = node_id or get_callable_name(call)

        step = Step[StateT, DepsT, InputT, OutputT](id=NodeID(node_id), call=call, label=label)

        return step

    @overload
    def stream(
        self,
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> Callable[
        [StreamFunction[StateT, DepsT, InputT, OutputT]], Step[StateT, DepsT, InputT, AsyncIterable[OutputT]]
    ]: ...
    @overload
    def stream(
        self,
        call: StreamFunction[StateT, DepsT, InputT, OutputT],
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> Step[StateT, DepsT, InputT, AsyncIterable[OutputT]]: ...
    @overload
    def stream(
        self,
        call: StreamFunction[StateT, DepsT, InputT, OutputT] | None = None,
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> (
        Step[StateT, DepsT, InputT, AsyncIterable[OutputT]]
        | Callable[
            [StreamFunction[StateT, DepsT, InputT, OutputT]],
            Step[StateT, DepsT, InputT, AsyncIterable[OutputT]],
        ]
    ): ...
    def stream(
        self,
        call: StreamFunction[StateT, DepsT, InputT, OutputT] | None = None,
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> (
        Step[StateT, DepsT, InputT, AsyncIterable[OutputT]]
        | Callable[
            [StreamFunction[StateT, DepsT, InputT, OutputT]],
            Step[StateT, DepsT, InputT, AsyncIterable[OutputT]],
        ]
    ):
        """Create a step from an async iterator (which functions like a "stream").

        This method can be used as a decorator or called directly to create
        a step node from an async function.

        Args:
            call: The step function to wrap
            node_id: Optional ID for the node
            label: Optional human-readable label

        Returns:
            Either a Step instance or a decorator function
        """
        if call is None:

            def decorator(
                func: StreamFunction[StateT, DepsT, InputT, OutputT],
            ) -> Step[StateT, DepsT, InputT, AsyncIterable[OutputT]]:
                return self.stream(call=func, node_id=node_id, label=label)

            return decorator

        # We need to wrap the call so that we can call `await` even though the result is an async iterator
        async def wrapper(ctx: StepContext[StateT, DepsT, InputT]):
            return call(ctx)

        node_id = node_id or get_callable_name(call)

        return self.step(call=wrapper, node_id=node_id, label=label)

    @overload
    def join(
        self,
        reducer: ReducerFunction[StateT, DepsT, InputT, OutputT],
        *,
        initial: OutputT,
        node_id: str | None = None,
        parent_fork_id: str | None = None,
        preferred_parent_fork: Literal['farthest', 'closest'] = 'farthest',
    ) -> Join[StateT, DepsT, InputT, OutputT]: ...
    @overload
    def join(
        self,
        reducer: ReducerFunction[StateT, DepsT, InputT, OutputT],
        *,
        initial_factory: Callable[[], OutputT],
        node_id: str | None = None,
        parent_fork_id: str | None = None,
        preferred_parent_fork: Literal['farthest', 'closest'] = 'farthest',
    ) -> Join[StateT, DepsT, InputT, OutputT]: ...

    def join(
        self,
        reducer: ReducerFunction[StateT, DepsT, InputT, OutputT],
        *,
        initial: OutputT | Unset = UNSET,
        initial_factory: Callable[[], OutputT] | Unset = UNSET,
        node_id: str | None = None,
        parent_fork_id: str | None = None,
        preferred_parent_fork: Literal['farthest', 'closest'] = 'farthest',
    ) -> Join[StateT, DepsT, InputT, OutputT]:
        if initial_factory is UNSET:
            initial_factory = lambda: initial  # pyright: ignore[reportAssignmentType]  # noqa: E731

        return Join[StateT, DepsT, InputT, OutputT](
            id=JoinID(NodeID(node_id or generate_placeholder_node_id(get_callable_name(reducer)))),
            reducer=reducer,
            initial_factory=cast(Callable[[], OutputT], initial_factory),
            parent_fork_id=ForkID(parent_fork_id) if parent_fork_id is not None else None,
            preferred_parent_fork=preferred_parent_fork,
        )

    # Edge building
    def add(self, *edges: EdgePath[StateT, DepsT]) -> None:  # noqa: C901
        """Add one or more edge paths to the graph.

        This method processes edge paths and automatically creates any necessary
        fork nodes for broadcasts and maps.

        Args:
            *edges: The edge paths to add to the graph
        """

        def _handle_path(p: Path):
            """Process a path and create necessary fork nodes.

            Args:
                p: The path to process
            """
            for item in p.items:
                if isinstance(item, BroadcastMarker):
                    new_node = Fork[Any, Any](id=item.fork_id, is_map=False, downstream_join_id=None)
                    self._insert_node(new_node)
                    for path in item.paths:
                        _handle_path(Path(items=[*path.items]))
                elif isinstance(item, MapMarker):
                    new_node = Fork[Any, Any](id=item.fork_id, is_map=True, downstream_join_id=item.downstream_join_id)
                    self._insert_node(new_node)
                elif isinstance(item, DestinationMarker):
                    pass

        def _handle_destination_node(d: AnyDestinationNode):
            if id(d) in destination_ids:
                return  # prevent infinite recursion if there is a cycle of decisions

            destination_ids.add(id(d))
            destinations.append(d)
            self._insert_node(d)
            if isinstance(d, Decision):
                for branch in d.branches:
                    _handle_path(branch.path)
                    for d2 in branch.destinations:
                        _handle_destination_node(d2)

        destination_ids = set[int]()
        destinations: list[AnyDestinationNode] = []
        for edge in edges:
            for source_node in edge.sources:
                self._insert_node(source_node)
                self._edges_by_source[source_node.id].append(edge.path)
            for destination_node in edge.destinations:
                _handle_destination_node(destination_node)
            _handle_path(edge.path)

        # Automatically create edges from step function return hints including `BaseNode`s
        for destination in destinations:
            if not isinstance(destination, Step) or isinstance(destination, NodeStep):
                continue
            parent_namespace = _utils.get_parent_namespace(inspect.currentframe())
            type_hints = get_type_hints(destination.call, localns=parent_namespace, include_extras=True)
            try:
                return_hint = type_hints['return']
            except KeyError:
                pass
            else:
                edge = self._edge_from_return_hint(destination, return_hint)
                if edge is not None:
                    self.add(edge)

    def add_edge(self, source: Source[T], destination: Destination[T], *, label: str | None = None) -> None:
        """Add a simple edge between two nodes.

        Args:
            source: The source node
            destination: The destination node
            label: Optional label for the edge
        """
        builder = self.edge_from(source)
        if label is not None:
            builder = builder.label(label)
        self.add(builder.to(destination))

    def add_mapping_edge(
        self,
        source: Source[Iterable[T]],
        map_to: Destination[T],
        *,
        pre_map_label: str | None = None,
        post_map_label: str | None = None,
        fork_id: ForkID | None = None,
        downstream_join_id: JoinID | None = None,
    ) -> None:
        """Add an edge that maps iterable data across parallel paths.

        Args:
            source: The source node that produces iterable data
            map_to: The destination node that receives individual items
            pre_map_label: Optional label before the map operation
            post_map_label: Optional label after the map operation
            fork_id: Optional ID for the fork node produced for this map operation
            downstream_join_id: Optional ID of a join node that will always be downstream of this map.
                Specifying this ensures correct handling if you try to map an empty iterable.
        """
        builder = self.edge_from(source)
        if pre_map_label is not None:
            builder = builder.label(pre_map_label)
        builder = builder.map(fork_id=fork_id, downstream_join_id=downstream_join_id)
        if post_map_label is not None:
            builder = builder.label(post_map_label)
        self.add(builder.to(map_to))

    # TODO(DavidM): Support adding subgraphs; I think this behaves like a step with the same inputs/outputs but gets rendered as a subgraph in mermaid

    def edge_from(self, *sources: Source[SourceOutputT]) -> EdgePathBuilder[StateT, DepsT, SourceOutputT]:
        """Create an edge path builder starting from the given source nodes.

        Args:
            *sources: The source nodes to start the edge path from

        Returns:
            An EdgePathBuilder for constructing the complete edge path
        """
        return EdgePathBuilder[StateT, DepsT, SourceOutputT](
            sources=sources, path_builder=PathBuilder(working_items=[])
        )

    def decision(self, *, note: str | None = None, node_id: str | None = None) -> Decision[StateT, DepsT, Never]:
        """Create a new decision node.

        Args:
            note: Optional note to describe the decision logic
            node_id: Optional ID for the node produced for this decision logic

        Returns:
            A new Decision node with no branches
        """
        return Decision(id=NodeID(node_id or generate_placeholder_node_id('decision')), branches=[], note=note)

    def match(
        self,
        source: TypeOrTypeExpression[SourceT],
        *,
        matches: Callable[[Any], bool] | None = None,
    ) -> DecisionBranchBuilder[StateT, DepsT, SourceT, SourceT, Never]:
        """Create a decision branch matcher.

        Args:
            source: The type or type expression to match against
            matches: Optional custom matching function

        Returns:
            A DecisionBranchBuilder for constructing the branch
        """
        # Note, the following node_id really is just a placeholder and shouldn't end up in the final graph
        # This is why we don't expose a way for end users to override the value used here.
        node_id = NodeID(generate_placeholder_node_id('match_decision'))
        decision = Decision[StateT, DepsT, Never](id=node_id, branches=[], note=None)
        new_path_builder = PathBuilder[StateT, DepsT, SourceT](working_items=[])
        return DecisionBranchBuilder(decision=decision, source=source, matches=matches, path_builder=new_path_builder)

    def match_node(
        self,
        source: type[SourceNodeT],
        *,
        matches: Callable[[Any], bool] | None = None,
    ) -> DecisionBranch[SourceNodeT]:
        """Create a decision branch for BaseNode subclasses.

        This is similar to match() but specifically designed for matching
        against BaseNode types from the v1 system.

        Args:
            source: The BaseNode subclass to match against
            matches: Optional custom matching function

        Returns:
            A DecisionBranch for the BaseNode type
        """
        node = NodeStep(source)
        path = Path(items=[DestinationMarker(node.id)])
        return DecisionBranch(source=source, matches=matches, path=path, destinations=[node])

    def node(
        self,
        node_type: type[BaseNode[StateT, DepsT, GraphOutputT]],
    ) -> EdgePath[StateT, DepsT]:
        """Create an edge path from a BaseNode class.

        This method integrates v1-style BaseNode classes into the v2 graph
        system by analyzing their type hints and creating appropriate edges.

        Args:
            node_type: The BaseNode subclass to integrate

        Returns:
            An EdgePath representing the node and its connections

        Raises:
            GraphSetupError: If the node type is missing required type hints
        """
        parent_namespace = _utils.get_parent_namespace(inspect.currentframe())
        type_hints = get_type_hints(node_type.run, localns=parent_namespace, include_extras=True)
        try:
            return_hint = type_hints['return']
        except KeyError as e:  # pragma: no cover
            raise exceptions.GraphSetupError(
                f'Node {node_type} is missing a return type hint on its `run` method'
            ) from e

        node = NodeStep(node_type)

        edge = self._edge_from_return_hint(node, return_hint)
        if not edge:  # pragma: no cover
            raise exceptions.GraphSetupError(f'Node {node_type} is missing a return type hint on its `run` method')

        return edge

    # Helpers
    def _insert_node(self, node: AnyNode) -> None:
        """Insert a node into the graph, checking for ID conflicts.

        Args:
            node: The node to insert

        Raises:
            ValueError: If a different node with the same ID already exists
        """
        existing = self._nodes.get(node.id)
        if existing is None:
            self._nodes[node.id] = node
        elif isinstance(existing, NodeStep) and isinstance(node, NodeStep) and existing.node_type is node.node_type:
            pass
        elif existing is not node:
            raise GraphBuildingError(
                f'All nodes must have unique node IDs. {node.id!r} was the ID for {existing} and {node}'
            )

    def _edge_from_return_hint(
        self, node: SourceNode[StateT, DepsT, Any], return_hint: TypeOrTypeExpression[Any]
    ) -> EdgePath[StateT, DepsT] | None:
        """Create edges from a return type hint.

        This method analyzes return type hints from step functions or node methods
        to automatically create appropriate edges in the graph.

        Args:
            node: The source node
            return_hint: The return type hint to analyze

        Returns:
            An EdgePath if edges can be inferred, None otherwise

        Raises:
            GraphSetupError: If the return type hint is invalid or incomplete
        """
        destinations: list[AnyDestinationNode] = []
        union_args = _utils.get_union_args(return_hint)
        for return_type in union_args:
            return_type, annotations = _utils.unpack_annotated(return_type)
            return_type_origin = get_origin(return_type) or return_type
            if return_type_origin is End:
                destinations.append(self.end_node)
            elif return_type_origin is BaseNode:
                raise exceptions.GraphSetupError(  # pragma: no cover
                    f'Node {node} return type hint includes a plain `BaseNode`. '
                    'Edge inference requires each possible returned `BaseNode` subclass to be listed explicitly.'
                )
            elif return_type_origin is StepNode:
                step = cast(
                    Step[StateT, DepsT, Any, Any] | None,
                    next((a for a in annotations if isinstance(a, Step)), None),  # pyright: ignore[reportUnknownArgumentType]
                )
                if step is None:
                    raise exceptions.GraphSetupError(  # pragma: no cover
                        f'Node {node} return type hint includes a `StepNode` without a `Step` annotation. '
                        'When returning `my_step.as_node()`, use `Annotated[StepNode[StateT, DepsT], my_step]` as the return type hint.'
                    )
                destinations.append(step)
            elif return_type_origin is JoinNode:
                join = cast(
                    Join[StateT, DepsT, Any, Any] | None,
                    next((a for a in annotations if isinstance(a, Join)), None),  # pyright: ignore[reportUnknownArgumentType]
                )
                if join is None:
                    raise exceptions.GraphSetupError(  # pragma: no cover
                        f'Node {node} return type hint includes a `JoinNode` without a `Join` annotation. '
                        'When returning `my_join.as_node()`, use `Annotated[JoinNode[StateT, DepsT], my_join]` as the return type hint.'
                    )
                destinations.append(join)
            elif inspect.isclass(return_type_origin) and issubclass(return_type_origin, BaseNode):
                destinations.append(NodeStep(return_type))

        if len(destinations) < len(union_args):
            # Only build edges if all the return types are nodes
            return None

        edge = self.edge_from(node)
        if len(destinations) == 1:
            return edge.to(destinations[0])
        else:
            decision = self.decision()
            for destination in destinations:
                # We don't actually use this decision mechanism, but we need to build the edges for parent-fork finding
                decision = decision.branch(self.match(NoneType).to(destination))
            return edge.to(decision)

    # Graph building
    def build(self, validate_graph_structure: bool = True) -> Graph[StateT, DepsT, GraphInputT, GraphOutputT]:
        """Build the final executable graph from the accumulated nodes and edges.

        This method performs validation, normalization, and analysis of the graph
        structure to create a complete, executable graph instance.

        Args:
            validate_graph_structure: whether to perform validation of the graph structure
                See the docstring of _validate_graph_structure below for more details.

        Returns:
            A complete Graph instance ready for execution

        Raises:
            ValueError: If the graph structure is invalid (e.g., join without parent fork)
        """
        nodes = self._nodes
        edges_by_source = self._edges_by_source

        nodes, edges_by_source = _replace_placeholder_node_ids(nodes, edges_by_source)
        nodes, edges_by_source = _flatten_paths(nodes, edges_by_source)
        nodes, edges_by_source = _normalize_forks(nodes, edges_by_source)
        if validate_graph_structure:
            _validate_graph_structure(nodes, edges_by_source)
        parent_forks = _collect_dominating_forks(nodes, edges_by_source)
        intermediate_join_nodes = _compute_intermediate_join_nodes(nodes, parent_forks)

        return Graph[StateT, DepsT, GraphInputT, GraphOutputT](
            name=self.name,
            state_type=unpack_type_expression(self.state_type),
            deps_type=unpack_type_expression(self.deps_type),
            input_type=unpack_type_expression(self.input_type),
            output_type=unpack_type_expression(self.output_type),
            nodes=nodes,
            edges_by_source=edges_by_source,
            parent_forks=parent_forks,
            intermediate_join_nodes=intermediate_join_nodes,
            auto_instrument=self.auto_instrument,
        )


def _validate_graph_structure(  # noqa: C901
    nodes: dict[NodeID, AnyNode],
    edges_by_source: dict[NodeID, list[Path]],
) -> None:
    """Validate the graph structure for common issues.

    This function raises an error if any of the following criteria are not met:
    1. There are edges from the start node
    2. There are edges to the end node
    3. No non-End node is a dead end (no outgoing edges)
    4. The end node is reachable from the start node
    5. All nodes are reachable from the start node

    Note 1: Under some circumstances it may be reasonable to build a graph that violates one or more of
    the above conditions. We may eventually add support for more granular control over validation,
    but today, if you want to build a graph that violates any of these assumptions you need to pass
    `validate_graph_structure=False` to the call to `GraphBuilder.build`.

    Note 2: Some of the earlier items in the above list are redundant with the later items.
    I've included the earlier items in the list as a reminder to ourselves if/when we add more granular validation
    because you might want to check the earlier items but not the later items, as described in Note 1.

    Args:
        nodes: The nodes in the graph
        edges_by_source: The edges by source node

    Raises:
        GraphBuildingError: If any of the aforementioned structural issues are found.
    """
    how_to_suppress = ' If this is intentional, you can suppress this error by passing `validate_graph_structure=False` to the call to `GraphBuilder.build`.'

    # Extract all destination IDs from edges and decision branches
    all_destinations: set[NodeID] = set()

    def _collect_destinations_from_path(path: Path) -> None:
        for item in path.items:
            if isinstance(item, DestinationMarker):
                all_destinations.add(item.destination_id)

    for paths in edges_by_source.values():
        for path in paths:
            _collect_destinations_from_path(path)

    # Also collect destinations from decision branches
    for node in nodes.values():
        if isinstance(node, Decision):
            for branch in node.branches:
                _collect_destinations_from_path(branch.path)

    # Check 1: Check if there are edges from the start node
    start_edges = edges_by_source.get(StartNode.id, [])
    if not start_edges:
        raise GraphValidationError('The graph has no edges from the start node.' + how_to_suppress)

    # Check 2: Check if there are edges to the end node
    if EndNode.id not in all_destinations:
        raise GraphValidationError('The graph has no edges to the end node.' + how_to_suppress)

    # Check 3: Find all nodes with no outgoing edges (dead ends)
    dead_end_nodes: list[NodeID] = []
    for node_id, node in nodes.items():
        # Skip the end node itself
        if isinstance(node, EndNode):
            continue

        # Check if this node has any outgoing edges
        has_edges = node_id in edges_by_source and len(edges_by_source[node_id]) > 0

        # Also check if it's a decision node with branches
        if isinstance(node, Decision):
            has_edges = has_edges or len(node.branches) > 0

        if not has_edges:
            dead_end_nodes.append(node_id)

    if dead_end_nodes:
        raise GraphValidationError(f'The following nodes have no outgoing edges: {dead_end_nodes}.' + how_to_suppress)

    # Checks 4 and 5: Ensure all nodes (and in particular, the end node) are reachable from the start node
    reachable: set[NodeID] = {StartNode.id}
    to_visit = [StartNode.id]

    while to_visit:
        current_id = to_visit.pop()

        # Add destinations from regular edges
        for path in edges_by_source.get(current_id, []):
            for item in path.items:
                if isinstance(item, DestinationMarker):
                    if item.destination_id not in reachable:
                        reachable.add(item.destination_id)
                        to_visit.append(item.destination_id)

        # Add destinations from decision branches
        current_node = nodes.get(current_id)
        if isinstance(current_node, Decision):
            for branch in current_node.branches:
                for item in branch.path.items:
                    if isinstance(item, DestinationMarker):
                        if item.destination_id not in reachable:
                            reachable.add(item.destination_id)
                            to_visit.append(item.destination_id)

    unreachable_nodes = [node_id for node_id in nodes if node_id not in reachable]
    if unreachable_nodes:
        raise GraphValidationError(
            f'The following nodes are not reachable from the start node: {unreachable_nodes}.' + how_to_suppress
        )


def _flatten_paths(
    nodes: dict[NodeID, AnyNode], edges: dict[NodeID, list[Path]]
) -> tuple[dict[NodeID, AnyNode], dict[NodeID, list[Path]]]:
    new_nodes = nodes.copy()
    new_edges: dict[NodeID, list[Path]] = defaultdict(list)

    paths_to_handle: list[tuple[NodeID, Path]] = []

    def _split_at_first_fork(path: Path) -> tuple[Path, list[tuple[NodeID, Path]]]:
        for i, item in enumerate(path.items):
            if isinstance(item, MapMarker):
                assert item.fork_id in nodes, 'This should have been added to the node during GraphBuilder.add'
                upstream = Path(list(path.items[:i]) + [DestinationMarker(item.fork_id)])
                downstream = Path(path.items[i + 1 :])
                return upstream, [(item.fork_id, downstream)]

            if isinstance(item, BroadcastMarker):
                assert item.fork_id in nodes, 'This should have been added to the node during GraphBuilder.add'
                upstream = Path(list(path.items[:i]) + [DestinationMarker(item.fork_id)])
                return upstream, [(item.fork_id, p) for p in item.paths]
        return path, []

    for node in new_nodes.values():
        if isinstance(node, Decision):
            for branch in node.branches:
                upstream, downstreams = _split_at_first_fork(branch.path)
                branch.path = upstream
                paths_to_handle.extend(downstreams)

    for source_id, edges_from_source in edges.items():
        for path in edges_from_source:
            paths_to_handle.append((source_id, path))

    while paths_to_handle:
        source_id, path = paths_to_handle.pop()
        upstream, downstreams = _split_at_first_fork(path)
        new_edges[source_id].append(upstream)
        paths_to_handle.extend(downstreams)

    return new_nodes, dict(new_edges)


def _normalize_forks(
    nodes: dict[NodeID, AnyNode], edges: dict[NodeID, list[Path]]
) -> tuple[dict[NodeID, AnyNode], dict[NodeID, list[Path]]]:
    """Normalize the graph structure so only broadcast forks have multiple outgoing edges.

    This function ensures that any node with multiple outgoing edges is converted
    to use an explicit broadcast fork, simplifying the graph execution model.

    Args:
        nodes: The nodes in the graph
        edges: The edges by source node

    Returns:
        A tuple of normalized nodes and edges
    """
    new_nodes = nodes.copy()
    new_edges: dict[NodeID, list[Path]] = {}

    paths_to_handle: list[Path] = []

    for source_id, edges_from_source in edges.items():
        paths_to_handle.extend(edges_from_source)

        node = nodes[source_id]
        if isinstance(node, Fork) and not node.is_map:
            new_edges[source_id] = edges_from_source
            continue  # broadcast fork; nothing to do
        if len(edges_from_source) == 1:
            new_edges[source_id] = edges_from_source
            continue
        new_fork = Fork[Any, Any](id=ForkID(NodeID(f'{node.id}_broadcast_fork')), is_map=False, downstream_join_id=None)
        new_nodes[new_fork.id] = new_fork
        new_edges[source_id] = [Path(items=[DestinationMarker(new_fork.id)])]
        new_edges[new_fork.id] = edges_from_source

    return new_nodes, new_edges


def _collect_dominating_forks(
    graph_nodes: dict[NodeID, AnyNode], graph_edges_by_source: dict[NodeID, list[Path]]
) -> dict[JoinID, ParentFork[NodeID]]:
    """Find the dominating fork for each join node in the graph.

    This function analyzes the graph structure to find the parent fork that
    dominates each join node, which is necessary for proper synchronization
    during graph execution.

    Args:
        graph_nodes: All nodes in the graph
        graph_edges_by_source: Edges organized by source node

    Returns:
        A mapping from join IDs to their parent fork information

    Raises:
        ValueError: If any join node lacks a dominating fork
    """
    nodes = set(graph_nodes)
    start_ids: set[NodeID] = {StartNode.id}
    edges: dict[NodeID, list[NodeID]] = defaultdict(list)

    fork_ids: set[NodeID] = set(start_ids)
    for source_id in nodes:
        working_source_id = source_id
        node = graph_nodes.get(source_id)

        if isinstance(node, Fork):
            fork_ids.add(node.id)

        def _handle_path(path: Path, last_source_id: NodeID):
            """Process a path and collect edges and fork information.

            Args:
                path: The path to process
                last_source_id: The current source node ID
            """
            for item in path.items:  # pragma: no branch
                # No need to handle MapMarker or BroadcastMarker here as these should have all been removed
                # by the call to `_flatten_paths`
                if isinstance(item, DestinationMarker):
                    edges[last_source_id].append(item.destination_id)
                    # Destinations should only ever occur as the last item in the list, so no need to update the working_source_id
                    break

        if isinstance(node, Decision):
            for branch in node.branches:
                _handle_path(branch.path, working_source_id)
        else:
            for path in graph_edges_by_source.get(source_id, []):
                _handle_path(path, source_id)

    finder = ParentForkFinder(
        nodes=nodes,
        start_ids=start_ids,
        fork_ids=fork_ids,
        edges=edges,
    )

    joins = [node for node in graph_nodes.values() if isinstance(node, Join)]
    dominating_forks: dict[JoinID, ParentFork[NodeID]] = {}
    for join in joins:
        dominating_fork = finder.find_parent_fork(
            join.id, parent_fork_id=join.parent_fork_id, prefer_closest=join.preferred_parent_fork == 'closest'
        )
        if dominating_fork is None:
            rendered_mermaid_graph = build_mermaid_graph(graph_nodes, graph_edges_by_source).render()
            raise GraphBuildingError(f"""A node in the graph is missing a dominating fork.

For every Join J in the graph, there must be a Fork F between the StartNode and J satisfying:
* Every path from the StartNode to J passes through F
* There are no cycles in the graph including J that don't pass through F.
In this case, F is called a "dominating fork" for J.

This is used to determine when all tasks upstream of this Join are complete and we can proceed with execution.

Mermaid diagram:
{rendered_mermaid_graph}

Join {join.id!r} in this graph has no dominating fork in this graph.""")
        dominating_forks[join.id] = dominating_fork

    return dominating_forks


def _compute_intermediate_join_nodes(
    nodes: dict[NodeID, AnyNode], parent_forks: dict[JoinID, ParentFork[NodeID]]
) -> dict[JoinID, set[JoinID]]:
    """Compute which joins have other joins as intermediate nodes.

    A join J1 is an intermediate node of join J2 if J1 appears in J2's intermediate_nodes
    (as computed relative to J2's parent fork).

    This information is used to determine:
    1. Which joins are "final" (have no other joins in their intermediate_nodes)
    2. When selecting which reducer to proceed with when there are no active tasks

    Args:
        nodes: All nodes in the graph
        parent_forks: Parent fork information for each join

    Returns:
        A mapping from each join to the set of joins that are intermediate to it
    """
    intermediate_join_nodes: dict[JoinID, set[JoinID]] = {}

    for join_id, parent_fork in parent_forks.items():
        intermediate_joins = set[JoinID]()
        for intermediate_node_id in parent_fork.intermediate_nodes:
            # Check if this intermediate node is also a join
            intermediate_node = nodes.get(intermediate_node_id)
            if isinstance(intermediate_node, Join):
                # Add it regardless of whether it has the same parent fork
                intermediate_joins.add(JoinID(intermediate_node_id))
        intermediate_join_nodes[join_id] = intermediate_joins

    return intermediate_join_nodes


def _replace_placeholder_node_ids(nodes: dict[NodeID, AnyNode], edges_by_source: dict[NodeID, list[Path]]):
    node_id_remapping = _build_placeholder_node_id_remapping(nodes)
    replaced_nodes = {
        node_id_remapping.get(name, name): _update_node_with_id_remapping(node, node_id_remapping)
        for name, node in nodes.items()
    }
    replaced_edges_by_source = {
        node_id_remapping.get(source, source): [_update_path_with_id_remapping(p, node_id_remapping) for p in paths]
        for source, paths in edges_by_source.items()
    }
    return replaced_nodes, replaced_edges_by_source


def _build_placeholder_node_id_remapping(nodes: dict[NodeID, AnyNode]) -> dict[NodeID, NodeID]:
    """The determinism of the generated remapping here is dependent on the determinism of the ordering of the `nodes` dict.

    Note: If we want to generate more interesting names, we could try to make use of information about the edges
    into/out of the relevant nodes. I'm not sure if there's a good use case for that though so I didn't bother for now.
    """
    counter = Counter[str]()
    remapping: dict[NodeID, NodeID] = {}
    for node_id in nodes.keys():
        replaced_node_id = replace_placeholder_id(node_id)
        if replaced_node_id == node_id:
            continue
        counter[replaced_node_id] = count = counter[replaced_node_id] + 1
        remapping[node_id] = NodeID(f'{replaced_node_id}_{count}' if count > 1 else replaced_node_id)
    return remapping


def _update_node_with_id_remapping(node: AnyNode, node_id_remapping: dict[NodeID, NodeID]) -> AnyNode:
    # Note: it's a bit awkward that we mutate the provided nodes, but this is necessary to ensure that
    # calls to `.as_node` reference the correct node_ids when relying on compatibility with the v1 API.
    # We only mutate placeholder IDs so I _think_ this should generally be okay. I guess we can
    # rework it more carefully if it causes issues in the future..
    if isinstance(node, Step):
        node.id = node_id_remapping.get(node.id, node.id)
    elif isinstance(node, Join):
        node.id = JoinID(node_id_remapping.get(node.id, node.id))
    elif isinstance(node, Fork):
        node.id = ForkID(node_id_remapping.get(node.id, node.id))
        if node.downstream_join_id is not None:
            node.downstream_join_id = JoinID(node_id_remapping.get(node.downstream_join_id, node.downstream_join_id))
    elif isinstance(node, Decision):
        node.id = node_id_remapping.get(node.id, node.id)
        node.branches = [
            replace(branch, path=_update_path_with_id_remapping(branch.path, node_id_remapping))
            for branch in node.branches
        ]
    return node


def _update_path_with_id_remapping(path: Path, node_id_remapping: dict[NodeID, NodeID]) -> Path:
    # Note: we have already deepcopied the node provided to this function so it should be okay to make mutations,
    # this could change if we change the code surrounding the code paths leading to this function call though.
    for item in path.items:
        if isinstance(item, MapMarker):
            downstream_join_id = item.downstream_join_id
            if downstream_join_id is not None:
                item.downstream_join_id = JoinID(node_id_remapping.get(downstream_join_id, downstream_join_id))
            item.fork_id = ForkID(node_id_remapping.get(item.fork_id, item.fork_id))
        elif isinstance(item, BroadcastMarker):
            item.fork_id = ForkID(node_id_remapping.get(item.fork_id, item.fork_id))
            item.paths = [_update_path_with_id_remapping(p, node_id_remapping) for p in item.paths]
        elif isinstance(item, DestinationMarker):
            item.destination_id = node_id_remapping.get(item.destination_id, item.destination_id)
    return path


# === Mermaid rendering ===

DEFAULT_HIGHLIGHT_CSS = 'fill:#fdff32'
"""The default CSS to use for highlighting nodes."""


StateDiagramDirection = Literal['TB', 'LR', 'RL', 'BT']
"""Used to specify the direction of the state diagram generated by mermaid.

- `'TB'`: Top to bottom, this is the default for mermaid charts.
- `'LR'`: Left to right
- `'RL'`: Right to left
- `'BT'`: Bottom to top
"""

NodeKind = Literal['broadcast', 'map', 'join', 'start', 'end', 'step', 'decision']


@dataclass
class MermaidNode:
    """A mermaid node."""

    id: str
    kind: NodeKind
    label: str | None
    note: str | None


@dataclass
class MermaidEdge:
    """A mermaid edge."""

    start_id: str
    end_id: str
    label: str | None


def build_mermaid_graph(  # noqa: C901
    graph_nodes: dict[NodeID, AnyNode], graph_edges_by_source: dict[NodeID, list[Path]]
) -> MermaidGraph:
    """Build a mermaid graph."""
    nodes: list[MermaidNode] = []
    edges_by_source: dict[str, list[MermaidEdge]] = defaultdict(list)

    def _collect_edges(path: Path, last_source_id: NodeID) -> None:
        working_label: str | None = None
        for item in path.items:
            assert not isinstance(item, MapMarker | BroadcastMarker), 'These should be removed during Graph building'
            if isinstance(item, LabelMarker):
                working_label = item.label
            elif isinstance(item, DestinationMarker):
                edges_by_source[last_source_id].append(MermaidEdge(last_source_id, item.destination_id, working_label))

    for node_id, node in graph_nodes.items():
        kind: NodeKind
        label: str | None = None
        note: str | None = None
        if isinstance(node, StartNode):
            kind = 'start'
        elif isinstance(node, EndNode):
            kind = 'end'
        elif isinstance(node, Step):
            kind = 'step'
            label = node.label
        elif isinstance(node, Join):
            kind = 'join'
        elif isinstance(node, Fork):
            kind = 'map' if node.is_map else 'broadcast'
        elif isinstance(node, Decision):
            kind = 'decision'
            note = node.note
        else:
            assert_never(node)

        source_node = MermaidNode(id=node_id, kind=kind, label=label, note=note)
        nodes.append(source_node)

    for k, v in graph_edges_by_source.items():
        for path in v:
            _collect_edges(path, k)

    for node in graph_nodes.values():
        if isinstance(node, Decision):
            for branch in node.branches:
                _collect_edges(branch.path, node.id)

    # Add edges in the same order that we added nodes
    edges: list[MermaidEdge] = sum([edges_by_source.get(node.id, []) for node in nodes], list[MermaidEdge]())
    return MermaidGraph(nodes, edges)


@dataclass
class MermaidGraph:
    """A mermaid graph."""

    nodes: list[MermaidNode]
    edges: list[MermaidEdge]

    title: str | None = None
    direction: StateDiagramDirection | None = None

    def render(
        self,
        direction: StateDiagramDirection | None = None,
        title: str | None = None,
        edge_labels: bool = True,
    ):
        lines: list[str] = []
        if title:
            lines = ['---', f'title: {title}', '---']
        lines.append('stateDiagram-v2')
        if direction is not None:
            lines.append(f'  direction {direction}')

        nodes, edges = _topological_sort(self.nodes, self.edges)
        for node in nodes:
            # List all nodes in order they were created
            node_lines: list[str] = []
            if node.kind == 'start' or node.kind == 'end':
                pass  # Start and end nodes use special [*] syntax in edges
            elif node.kind == 'step':
                line = f'  {node.id}'
                if node.label:
                    line += f': {node.label}'
                node_lines.append(line)
            elif node.kind == 'join':
                node_lines = [f'  state {node.id} <<join>>']
            elif node.kind == 'broadcast' or node.kind == 'map':
                node_lines = [f'  state {node.id} <<fork>>']
            elif node.kind == 'decision':
                node_lines = [f'  state {node.id} <<choice>>']
                if node.note:
                    node_lines.append(f'  note right of {node.id}\n    {node.note}\n  end note')
            else:  # pragma: no cover
                assert_never(node.kind)
            lines.extend(node_lines)

        lines.append('')

        for edge in edges:
            # Use special [*] syntax for start/end nodes
            render_start_id = '[*]' if edge.start_id == StartNode.id else edge.start_id
            render_end_id = '[*]' if edge.end_id == EndNode.id else edge.end_id
            edge_line = f'  {render_start_id} --> {render_end_id}'
            if edge.label and edge_labels:
                edge_line += f': {edge.label}'
            lines.append(edge_line)

        return '\n'.join(lines)


def _topological_sort(
    nodes: list[MermaidNode], edges: list[MermaidEdge]
) -> tuple[list[MermaidNode], list[MermaidEdge]]:
    """Sort nodes and edges in a logical topological order.

    Uses BFS from the start node to assign depths, then sorts:
    - Nodes by their distance from start
    - Edges by the distance of their source and target nodes
    """
    # Build adjacency list for BFS
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.start_id].append(edge.end_id)

    # BFS to assign depth to each node (distance from start)
    depths: dict[str, int] = {}
    queue: list[tuple[str, int]] = [(StartNode.id, 0)]
    depths[StartNode.id] = 0

    while queue:
        node_id, depth = queue.pop(0)
        for next_id in adjacency[node_id]:
            if next_id not in depths:  # pragma: no branch
                depths[next_id] = depth + 1
                queue.append((next_id, depth + 1))

    # Sort nodes by depth (distance from start), then by id for stability
    # Nodes not reachable from start get infinity depth (sorted to end)
    sorted_nodes = sorted(nodes, key=lambda n: (depths.get(n.id, float('inf')), n.id))

    # Sort edges by source depth, then target depth
    # This ensures edges closer to start come first, edges closer to end come last
    sorted_edges = sorted(
        edges,
        key=lambda e: (
            depths.get(e.start_id, float('inf')),
            depths.get(e.end_id, float('inf')),
            e.start_id,
            e.end_id,
        ),
    )

    return sorted_nodes, sorted_edges
