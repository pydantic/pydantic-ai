from __future__ import annotations as _annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Sequence
from contextlib import AbstractContextManager, ExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, cast, get_args, get_origin, overload

from typing_extensions import TypeVar, assert_never

from pydantic_graph import exceptions
from pydantic_graph._utils import AbstractSpan, get_traceparent, logfire_span
from pydantic_graph.nodes import BaseNode, End
from pydantic_graph.v2.decision import Decision
from pydantic_graph.v2.id_types import ForkStack, ForkStackItem, GraphRunId, JoinId, NodeId, NodeRunId, TaskId
from pydantic_graph.v2.join import Join, Reducer
from pydantic_graph.v2.node import (
    EndNode,
    Fork,
    StartNode,
)
from pydantic_graph.v2.node_types import AnyNode
from pydantic_graph.v2.parent_forks import ParentFork
from pydantic_graph.v2.paths import BroadcastMarker, DestinationMarker, LabelMarker, Path, SpreadMarker, TransformMarker
from pydantic_graph.v2.step import NodeStep, Step, StepContext, StepNode
from pydantic_graph.v2.util import unpack_type_expression

if TYPE_CHECKING:
    from pydantic_graph.v2.mermaid import StateDiagramDirection


StateT = TypeVar('StateT', infer_variance=True)
DepsT = TypeVar('DepsT', infer_variance=True)
InputT = TypeVar('InputT', infer_variance=True)
OutputT = TypeVar('OutputT', infer_variance=True)


@dataclass
class EndMarker(Generic[OutputT]):
    """An end marker."""

    value: OutputT


@dataclass
class JoinItem:
    """A join item."""

    join_id: JoinId
    inputs: Any
    fork_stack: ForkStack


@dataclass(repr=False)
class Graph(Generic[StateT, DepsT, InputT, OutputT]):
    """A graph."""

    state_type: type[StateT]
    deps_type: type[DepsT]
    input_type: type[InputT]
    output_type: type[OutputT]

    auto_instrument: bool

    nodes: dict[NodeId, AnyNode]
    edges_by_source: dict[NodeId, list[Path]]
    parent_forks: dict[JoinId, ParentFork[NodeId]]

    def get_parent_fork(self, join_id: JoinId) -> ParentFork[NodeId]:
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
    ) -> OutputT:
        async with self.iter(state=state, deps=deps, inputs=inputs, span=span) as graph_run:
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
    ) -> AsyncIterator[GraphRun[StateT, DepsT, OutputT]]:
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
        from pydantic_graph.v2.mermaid import build_mermaid_graph

        return build_mermaid_graph(self).render(title=title, direction=direction)

    def __repr__(self):
        return self.render()


@dataclass
class GraphTask:
    """A graph task."""

    # With our current BaseNode thing, next_node_id and next_node_inputs are merged into `next_node` itself
    node_id: NodeId
    inputs: Any
    fork_stack: ForkStack
    """
    Stack of forks that have been entered; used so that the GraphRunner can decide when to proceed through joins
    """

    task_id: TaskId = field(default_factory=lambda: TaskId(str(uuid.uuid4())))


class GraphRun(Generic[StateT, DepsT, OutputT]):
    """A graph run."""

    def __init__(
        self,
        graph: Graph[StateT, DepsT, InputT, OutputT],
        *,
        state: StateT,
        deps: DepsT,
        inputs: InputT,
        traceparent: str | None,
    ):
        self.graph = graph
        self.state = state
        self.deps = deps
        self.inputs = inputs

        self._active_reducers: dict[tuple[JoinId, NodeRunId], Reducer[Any, Any, Any, Any]] = {}

        self._next: EndMarker[OutputT] | JoinItem | Sequence[GraphTask] | None = None

        run_id = GraphRunId(str(uuid.uuid4()))
        initial_fork_stack: ForkStack = (ForkStackItem(StartNode.id, NodeRunId(run_id), 0),)
        self._first_task = GraphTask(node_id=StartNode.id, inputs=inputs, fork_stack=initial_fork_stack)
        self._iterator = self._iter_graph()

        self.__traceparent = traceparent

    @overload
    def _traceparent(self, *, required: Literal[False]) -> str | None: ...
    @overload
    def _traceparent(self) -> str: ...
    def _traceparent(self, *, required: bool = True) -> str | None:
        if self.__traceparent is None and required:  # pragma: no cover
            raise exceptions.GraphRuntimeError('No span was created for this graph run')
        return self.__traceparent

    def __aiter__(self) -> AsyncIterator[EndMarker[OutputT] | JoinItem | Sequence[GraphTask]]:
        return self

    async def __anext__(self) -> EndMarker[OutputT] | JoinItem | Sequence[GraphTask]:
        if self._next is None:
            self._next = await self._iterator.__anext__()
        else:
            self._next = await self._iterator.asend(self._next)
        return self._next

    async def next(
        self, value: EndMarker[OutputT] | JoinItem | Sequence[GraphTask] | None = None
    ) -> EndMarker[OutputT] | JoinItem | Sequence[GraphTask]:
        """Allows for sending a value to the iterator, which is useful for resuming the iteration."""
        if value is not None:
            self._next = value
        return await self.__anext__()

    @property
    def next_step(self) -> EndMarker[OutputT] | JoinItem | Sequence[GraphTask] | None:
        return self._next

    @property
    def output(self) -> OutputT | None:
        if isinstance(self._next, EndMarker):
            return self._next.value
        return None

    async def _iter_graph(
        self,
    ) -> AsyncGenerator[
        EndMarker[OutputT] | JoinItem | Sequence[GraphTask], EndMarker[OutputT] | JoinItem | Sequence[GraphTask]
    ]:
        tasks_by_id: dict[TaskId, GraphTask] = {}
        pending: set[asyncio.Task[EndMarker[OutputT] | JoinItem | Sequence[GraphTask]]] = set()

        def _start_task(t_: GraphTask) -> None:
            """Helper function to start a new task while doing all necessary tracking."""
            tasks_by_id[t_.task_id] = t_
            pending.add(asyncio.create_task(self._handle_task(t_), name=t_.task_id))

        _start_task(self._first_task)

        def _handle_result(result: EndMarker[OutputT] | JoinItem | Sequence[GraphTask]) -> bool:
            if isinstance(result, EndMarker):
                for t in pending:
                    t.cancel()
                return True

            if isinstance(result, JoinItem):
                parent_fork_id = self.graph.get_parent_fork(result.join_id).fork_id
                fork_run_id = [x.node_run_id for x in result.fork_stack[::-1] if x.fork_id == parent_fork_id][0]
                reducer = self._active_reducers.get((result.join_id, fork_run_id))
                if reducer is None:
                    join_node = self.graph.nodes[result.join_id]
                    assert isinstance(join_node, Join)
                    reducer = join_node.create_reducer(StepContext(None, None, result.inputs))
                    self._active_reducers[(result.join_id, fork_run_id)] = reducer
                else:
                    reducer.reduce(StepContext(None, None, result.inputs))
            else:
                for new_task in result:
                    _start_task(new_task)
            return False

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task_result = task.result()
                source_task = tasks_by_id.pop(TaskId(task.get_name()))
                maybe_overridden_result = yield task_result
                if _handle_result(maybe_overridden_result):
                    return

                for join_id, fork_run_id, fork_stack in self._get_completed_fork_runs(
                    source_task, tasks_by_id.values()
                ):
                    reducer = self._active_reducers.pop((join_id, fork_run_id))

                    output = reducer.finalize(StepContext(None, None, None))
                    join_node = self.graph.nodes[join_id]
                    assert isinstance(join_node, Join)  # We could drop this but if it fails it means there is a bug.
                    new_tasks = self._handle_edges(join_node, output, fork_stack)
                    maybe_overridden_result = yield new_tasks  # Need to give an opportunity to override these
                    if _handle_result(maybe_overridden_result):
                        return

        raise RuntimeError(
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
            step_context = StepContext[StateT, DepsT, Any](state, deps, inputs)
            output = await node.call(step_context)
            if isinstance(node, NodeStep):
                return self._handle_node(node, output, fork_stack)
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
                    except TypeError as e:
                        raise RuntimeError(f'Decision branch source {branch_source} is not a valid type.') from e

            if inputs_match:
                return self._handle_path(branch.path, inputs, fork_stack)

        raise RuntimeError(f'No branch matched inputs {inputs} for decision node {decision}.')

    def _handle_node(
        self,
        node_step: NodeStep[StateT, DepsT],
        next_node: BaseNode[StateT, DepsT, Any] | End[Any],
        fork_stack: ForkStack,
    ) -> Sequence[GraphTask] | EndMarker[OutputT]:
        if isinstance(next_node, StepNode):
            return [GraphTask(next_node.step.id, next_node.inputs, fork_stack)]
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
    ) -> list[tuple[JoinId, NodeRunId, ForkStack]]:
        completed_fork_runs: list[tuple[JoinId, NodeRunId, ForkStack]] = []

        fork_run_indices = {fsi.node_run_id: i for i, fsi in enumerate(t.fork_stack)}
        for join_id, fork_run_id in self._active_reducers.keys():
            fork_run_index = fork_run_indices.get(fork_run_id)
            if fork_run_index is None:
                continue  # The fork_run_id is not in the current task's fork stack, so this task didn't complete it.

            new_fork_stack = t.fork_stack[:fork_run_index]
            # This reducer _may_ now be ready to finalize:
            if self._is_fork_run_completed(active_tasks, join_id, fork_run_id):
                completed_fork_runs.append((join_id, fork_run_id, new_fork_stack))

        return completed_fork_runs

    def _handle_path(self, path: Path, inputs: Any, fork_stack: ForkStack) -> Sequence[GraphTask]:
        if not path.items:
            return []

        item = path.items[0]
        if isinstance(item, DestinationMarker):
            return [GraphTask(item.destination_id, inputs, fork_stack)]
        elif isinstance(item, SpreadMarker):
            node_run_id = NodeRunId(str(uuid.uuid4()))
            return [
                GraphTask(
                    item.fork_id, input_item, fork_stack + (ForkStackItem(item.fork_id, node_run_id, thread_index),)
                )
                for thread_index, input_item in enumerate(inputs)
            ]
        elif isinstance(item, BroadcastMarker):
            return [GraphTask(item.fork_id, inputs, fork_stack)]
        elif isinstance(item, TransformMarker):
            inputs = item.transform(StepContext(self.state, self.deps, inputs))
            return self._handle_path(path.next_path, inputs, fork_stack)
        elif isinstance(item, LabelMarker):
            return self._handle_path(path.next_path, inputs, fork_stack)
        else:
            assert_never(item)

    def _handle_edges(self, node: AnyNode, inputs: Any, fork_stack: ForkStack) -> Sequence[GraphTask]:
        edges = self.graph.edges_by_source.get(node.id, [])
        assert len(edges) == 1 or isinstance(node, Fork)  # this should have already been ensured during graph building

        new_tasks: list[GraphTask] = []
        for path in edges:
            new_tasks.extend(self._handle_path(path, inputs, fork_stack))
        return new_tasks

    def _is_fork_run_completed(self, tasks: Iterable[GraphTask], join_id: JoinId, fork_run_id: NodeRunId) -> bool:
        # Check if any of the tasks in the graph have this fork_run_id in their fork_stack
        # If this is the case, then the fork run is not yet completed
        parent_fork = self.graph.get_parent_fork(join_id)
        for t in tasks:
            if fork_run_id in {x.node_run_id for x in t.fork_stack}:
                if t.node_id in parent_fork.intermediate_nodes:
                    return False
        return True
