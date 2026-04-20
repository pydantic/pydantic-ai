from __future__ import annotations as _annotations

import dataclasses
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING, Any, Generic, Literal, overload

from pydantic_graph import BaseNode, End, GraphRunContext
from pydantic_graph.beta.graph import EndMarker, ErrorMarker, GraphRun, GraphTaskRequest, JoinItem
from pydantic_graph.beta.step import NodeStep

from . import (
    _agent_graph,
    _utils,
    exceptions,
    messages as _messages,
    usage as _usage,
)
from .output import OutputDataT
from .tools import AgentDepsT

if TYPE_CHECKING:
    from ._run_context import RunContext
    from .result import FinalResult


@dataclasses.dataclass(repr=False)
class AgentRun(Generic[AgentDepsT, OutputDataT]):
    """A stateful, async-iterable run of an [`Agent`][pydantic_ai.agent.Agent].

    You generally obtain an `AgentRun` instance by calling `async with my_agent.iter(...) as agent_run:`.

    Once you have an instance, you can use it to iterate through the run's nodes as they execute. When an
    [`End`][pydantic_graph.nodes.End] is reached, the run finishes and [`result`][pydantic_ai.agent.AgentRun.result]
    becomes available.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-5.2')

    async def main():
        nodes = []
        # Iterate through the run, recording each node along the way:
        async with agent.iter('What is the capital of France?') as agent_run:
            async for node in agent_run:
                nodes.append(node)
        print(nodes)
        '''
        [
            UserPromptNode(
                user_prompt='What is the capital of France?',
                instructions_functions=[],
                system_prompts=(),
                system_prompt_functions=[],
                system_prompt_dynamic_functions={},
            ),
            ModelRequestNode(
                request=ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='What is the capital of France?',
                            timestamp=datetime.datetime(...),
                        )
                    ],
                    timestamp=datetime.datetime(...),
                    run_id='...',
                )
            ),
            CallToolsNode(
                model_response=ModelResponse(
                    parts=[TextPart(content='The capital of France is Paris.')],
                    usage=RequestUsage(input_tokens=56, output_tokens=7),
                    model_name='gpt-5.2',
                    timestamp=datetime.datetime(...),
                    run_id='...',
                )
            ),
            End(data=FinalResult(output='The capital of France is Paris.')),
        ]
        '''
        print(agent_run.result.output)
        #> The capital of France is Paris.
    ```

    You can also manually drive the iteration using the [`next`][pydantic_ai.agent.AgentRun.next] method for
    more granular control.
    """

    _graph_run: GraphRun[
        _agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any], FinalResult[OutputDataT]
    ]
    _result_override: AgentRunResult[OutputDataT] | None = dataclasses.field(default=None, repr=False, init=False)
    _node_error: BaseException | None = dataclasses.field(default=None, repr=False, init=False)
    """Stores the original exception from node execution, before context manager __aexit__ may transform it."""

    @overload
    def _traceparent(self, *, required: Literal[False]) -> str | None: ...
    @overload
    def _traceparent(self) -> str: ...
    def _traceparent(self, *, required: bool = True) -> str | None:
        traceparent = self._graph_run._traceparent(required=False)  # type: ignore[reportPrivateUsage]
        if traceparent is None and required:  # pragma: no cover
            raise AttributeError('No span was created for this agent run')
        return traceparent

    @property
    def ctx(self) -> GraphRunContext[_agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any]]:
        """The current context of the agent run."""
        return GraphRunContext[_agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any]](
            state=self._graph_run.state, deps=self._graph_run.deps
        )

    @property
    def next_node(
        self,
    ) -> _agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]:
        """The next node that will be run in the agent graph.

        This is the next node that will be used during async iteration, or if a node is not passed to `self.next(...)`.
        """
        task = self._graph_run.next_task
        if isinstance(task, ErrorMarker):
            raise task.error
        return self._task_to_node(task)

    @property
    def result(self) -> AgentRunResult[OutputDataT] | None:
        """The final result of the run if it has ended, otherwise `None`.

        Once the run returns an [`End`][pydantic_graph.nodes.End] node, `result` is populated
        with an [`AgentRunResult`][pydantic_ai.agent.AgentRunResult].
        """
        if self._result_override is not None:
            return self._result_override
        graph_run_output = self._graph_run.output
        if graph_run_output is None:
            return None
        return AgentRunResult(
            graph_run_output.output,
            graph_run_output.tool_name,
            self._graph_run.state,
            self._graph_run.deps.new_message_index,
            self._traceparent(required=False),
        )

    def all_messages(self) -> list[_messages.ModelMessage]:
        """Return all messages for the run so far.

        Messages from older runs are included.
        """
        return self.ctx.state.message_history

    def all_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:
        """Return all messages from [`all_messages`][pydantic_ai.agent.AgentRun.all_messages] as JSON bytes.

        Returns:
            JSON bytes representing the messages.
        """
        return _messages.ModelMessagesTypeAdapter.dump_json(self.all_messages())

    def new_messages(self) -> list[_messages.ModelMessage]:
        """Return the messages produced during this run so far.

        Messages provided via `message_history` and messages from older runs are excluded.
        """
        return self.all_messages()[self.ctx.deps.new_message_index :]

    def new_messages_json(self) -> bytes:
        """Return new messages from [`new_messages`][pydantic_ai.agent.AgentRun.new_messages] as JSON bytes.

        Returns:
            JSON bytes representing the new messages.
        """
        return _messages.ModelMessagesTypeAdapter.dump_json(self.new_messages())

    def __aiter__(
        self,
    ) -> AsyncIterator[_agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]]:
        """Provide async-iteration over the nodes in the agent run."""
        if self.ctx.deps.root_capability.has_wrap_node_run:
            warnings.warn(
                'A capability has `wrap_node_run` hooks, but bare `async for node in agent_run` '
                'does not fire them. Use `agent_run.next(node)` to advance the run, or use '
                '`agent.run()` which drives via `next()` automatically.',
                UserWarning,
                stacklevel=2,
            )
        return self

    async def __anext__(
        self,
    ) -> _agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]:
        """Advance to the next node automatically based on the last returned node.

        Note: this uses the graph run's internal iteration which does NOT call
        node hooks (`before_node_run`, `wrap_node_run`, `after_node_run`,
        `on_node_run_error`). Use `next()` for capability-hooked iteration, or
        use `agent.run()` which drives via `next()` automatically.
        """
        if self._result_override is not None:
            raise StopAsyncIteration
        try:
            task = await anext(self._graph_run)
        except BaseException as exc:
            self._node_error = exc
            raise
        return self._task_to_node(task)

    def _task_to_node(
        self, task: EndMarker[FinalResult[OutputDataT]] | JoinItem | Sequence[GraphTaskRequest]
    ) -> _agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]:
        if isinstance(task, Sequence) and len(task) == 1:
            first_task = task[0]
            if isinstance(first_task.inputs, BaseNode):  # pragma: no branch
                base_node: BaseNode[  # pyright: ignore[reportUnknownVariableType]
                    _agent_graph.GraphAgentState,
                    _agent_graph.GraphAgentDeps[AgentDepsT, OutputDataT],
                    FinalResult[OutputDataT],
                ] = first_task.inputs  # pyright: ignore[reportUnknownMemberType]
                if _agent_graph.is_agent_node(node=base_node):  # pragma: no branch
                    return base_node
        if isinstance(task, EndMarker):
            return End(task.value)
        raise exceptions.AgentRunError(f'Unexpected node: {task}')  # pragma: no cover

    def _node_to_task(self, node: _agent_graph.AgentNode[AgentDepsT, OutputDataT]) -> GraphTaskRequest:
        return GraphTaskRequest(NodeStep(type(node)).id, inputs=node, fork_stack=())

    def _sync_graph_state(self, result: _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]) -> None:
        """Synchronize the graph runner's state to match a hook-modified result.

        After a capability hook changes the result (e.g. `on_node_run_error` recovering,
        or `after_node_run` converting End↔node), the graph runner's internal `_next` must
        be updated so that `output` and `next_node` reflect the hook's decision.
        """
        if isinstance(result, End):
            self._graph_run.override_next(EndMarker(result.data))
        else:
            self._graph_run.override_next([self._node_to_task(result)])

    async def _advance_graph(
        self,
        node: _agent_graph.AgentNode[AgentDepsT, Any],
    ) -> _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]:
        """Execute a single graph step without firing capability hooks."""
        task = [self._node_to_task(node)]
        try:
            task = await self._graph_run.next(task)
        except StopAsyncIteration:
            pass
        return self._task_to_node(task)

    async def _wrap_and_advance(
        self,
        run_context: RunContext[AgentDepsT],
        node: _agent_graph.AgentNode[AgentDepsT, Any],
        step_fn: Callable[
            [_agent_graph.AgentNode[AgentDepsT, Any]],
            Awaitable[_agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]],
        ],
    ) -> _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]:
        """Execute `wrap_node_run(step_fn)` → `on_node_run_error` → `after_node_run`.

        This is the portion of the hook lifecycle after `before_node_run` has already fired.
        Used by both `_run_node_with_hooks` and directly by `run_stream()` which calls
        `before_node_run` separately (before streaming).
        """
        cap = self.ctx.deps.root_capability
        try:
            result = await cap.wrap_node_run(run_context, node=node, handler=step_fn)
        except Exception as e:
            result = await cap.on_node_run_error(run_context, node=node, error=e)
            # on_node_run_error recovered by returning a result.
            # The graph runner is in ErrorMarker state; update it to match.
            self._sync_graph_state(result)
        pre_hook_result = result
        result = await cap.after_node_run(run_context, node=node, result=result)

        # If after_node_run changed the result, sync the graph runner state so
        # agent_run.result correctly reflects whether the run is finished.
        if result is not pre_hook_result:
            self._sync_graph_state(result)

        return result

    async def _run_node_with_hooks(
        self,
        node: _agent_graph.AgentNode[AgentDepsT, Any],
        step_fn: Callable[
            [_agent_graph.AgentNode[AgentDepsT, Any]],
            Awaitable[_agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]],
        ],
    ) -> _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]:
        """Run a node through the full capability hook lifecycle with a custom step function.

        Fires hooks in order: `before_node_run` → `wrap_node_run(step_fn)` → `after_node_run`,
        with `on_node_run_error` handling exceptions from `wrap_node_run`.
        """
        run_context = _agent_graph.build_run_context(self.ctx)
        cap = self.ctx.deps.root_capability
        node = await cap.before_node_run(run_context, node=node)
        return await self._wrap_and_advance(run_context, node, step_fn)

    async def next(
        self,
        node: _agent_graph.AgentNode[AgentDepsT, OutputDataT],
    ) -> _agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]:
        """Manually drive the agent run by passing in the node you want to run next.

        This lets you inspect or mutate the node before continuing execution, or skip certain nodes
        under dynamic conditions. The agent run should be stopped when you return an [`End`][pydantic_graph.nodes.End]
        node.

        Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_graph import End

        agent = Agent('openai:gpt-5.2')

        async def main():
            async with agent.iter('What is the capital of France?') as agent_run:
                next_node = agent_run.next_node  # start with the first node
                nodes = [next_node]
                while not isinstance(next_node, End):
                    next_node = await agent_run.next(next_node)
                    nodes.append(next_node)
                # Once `next_node` is an End, we've finished:
                print(nodes)
                '''
                [
                    UserPromptNode(
                        user_prompt='What is the capital of France?',
                        instructions_functions=[],
                        system_prompts=(),
                        system_prompt_functions=[],
                        system_prompt_dynamic_functions={},
                    ),
                    ModelRequestNode(
                        request=ModelRequest(
                            parts=[
                                UserPromptPart(
                                    content='What is the capital of France?',
                                    timestamp=datetime.datetime(...),
                                )
                            ],
                            timestamp=datetime.datetime(...),
                            run_id='...',
                        )
                    ),
                    CallToolsNode(
                        model_response=ModelResponse(
                            parts=[TextPart(content='The capital of France is Paris.')],
                            usage=RequestUsage(input_tokens=56, output_tokens=7),
                            model_name='gpt-5.2',
                            timestamp=datetime.datetime(...),
                            run_id='...',
                        )
                    ),
                    End(data=FinalResult(output='The capital of France is Paris.')),
                ]
                '''
                print('Final result:', agent_run.result.output)
                #> Final result: The capital of France is Paris.
        ```

        Args:
            node: The node to run next in the graph.

        Returns:
            The next node returned by the graph logic, or an [`End`][pydantic_graph.nodes.End] node if
            the run has completed.
        """
        # Note: It might be nice to expose a synchronous interface for iteration, but we shouldn't do it
        # on this class, or else IDEs won't warn you if you accidentally use `for` instead of `async for` to iterate.
        return await self._run_node_with_hooks(node, self._advance_graph)

    # TODO (v2): Make this a property
    def usage(self) -> _usage.RunUsage:
        """Get usage statistics for the run so far, including token usage, model requests, and so on."""
        return self._graph_run.state.usage

    @property
    def metadata(self) -> dict[str, Any] | None:
        """Metadata associated with this agent run, if configured."""
        return self._graph_run.state.metadata

    @property
    def run_id(self) -> str:
        """The unique identifier for the agent run."""
        return self._graph_run.state.run_id

    def __repr__(self) -> str:  # pragma: no cover
        result = self._graph_run.output
        result_repr = '<run not finished>' if result is None else repr(result.output)
        return f'<{type(self).__name__} result={result_repr} usage={self.usage()}>'


@dataclasses.dataclass
class AgentRunResult(Generic[OutputDataT]):
    """The final result of an agent run."""

    output: OutputDataT
    """The output data from the agent run."""

    _output_tool_name: str | None = dataclasses.field(repr=False, compare=False, default=None)
    _state: _agent_graph.GraphAgentState = dataclasses.field(
        repr=False, compare=False, default_factory=_agent_graph.GraphAgentState
    )
    _new_message_index: int = dataclasses.field(repr=False, compare=False, default=0)
    _traceparent_value: str | None = dataclasses.field(repr=False, compare=False, default=None)

    @overload
    def _traceparent(self, *, required: Literal[False]) -> str | None: ...
    @overload
    def _traceparent(self) -> str: ...
    def _traceparent(self, *, required: bool = True) -> str | None:
        if self._traceparent_value is None and required:  # pragma: no cover
            raise AttributeError('No span was created for this agent run')
        return self._traceparent_value

    def _set_output_tool_return(self, return_content: str) -> list[_messages.ModelMessage]:
        """Set return content for the output tool.

        Useful if you want to continue the conversation and want to set the response to the output tool call.
        """
        if not self._output_tool_name:
            raise ValueError('Cannot set output tool return content when the return type is `str`.')

        messages = self._state.message_history
        last_message = messages[-1]
        for idx, part in enumerate(last_message.parts):
            if isinstance(part, _messages.ToolReturnPart) and part.tool_name == self._output_tool_name:
                # Only do deepcopy when we have to modify
                copied_messages = list(messages)
                copied_last = deepcopy(last_message)
                copied_last.parts[idx].content = return_content  # type: ignore[misc]
                copied_messages[-1] = copied_last
                return copied_messages

        raise LookupError(f'No tool call found with tool name {self._output_tool_name!r}.')

    def all_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
        """Return the history of _messages.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            List of messages.
        """
        if output_tool_return_content is not None:
            return self._set_output_tool_return(output_tool_return_content)
        else:
            return self._state.message_history

    def all_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:
        """Return all messages from [`all_messages`][pydantic_ai.agent.AgentRunResult.all_messages] as JSON bytes.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            JSON bytes representing the messages.
        """
        return _messages.ModelMessagesTypeAdapter.dump_json(
            self.all_messages(output_tool_return_content=output_tool_return_content)
        )

    def new_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
        """Return the messages produced during this run.

        Messages provided via `message_history` and messages from older runs are excluded.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            List of new messages.
        """
        return self.all_messages(output_tool_return_content=output_tool_return_content)[self._new_message_index :]

    def new_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:
        """Return new messages from [`new_messages`][pydantic_ai.agent.AgentRunResult.new_messages] as JSON bytes.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            JSON bytes representing the new messages.
        """
        return _messages.ModelMessagesTypeAdapter.dump_json(
            self.new_messages(output_tool_return_content=output_tool_return_content)
        )

    @property
    def response(self) -> _messages.ModelResponse:
        """Return the last response from the message history."""
        # The response may not be the very last item if it contained an output tool call. See `CallToolsNode._handle_final_result`.
        for message in reversed(self.all_messages()):
            if isinstance(message, _messages.ModelResponse):
                return message
        raise ValueError('No response found in the message history')  # pragma: no cover

    # TODO (v2): Make this a property
    def usage(self) -> _usage.RunUsage:
        """Return the usage of the whole run."""
        return self._state.usage

    # TODO (v2): Make this a property
    def timestamp(self) -> datetime:
        """Return the timestamp of last response."""
        return self.response.timestamp

    @property
    def metadata(self) -> dict[str, Any] | None:
        """Metadata associated with this agent run, if configured."""
        return self._state.metadata

    @property
    def run_id(self) -> str:
        """The unique identifier for the agent run."""
        return self._state.run_id


@dataclasses.dataclass(repr=False)
class AgentRunResultEvent(Generic[OutputDataT]):
    """An event indicating the agent run ended and containing the final result of the agent run."""

    result: AgentRunResult[OutputDataT]
    """The result of the run."""

    _: dataclasses.KW_ONLY

    event_kind: Literal['agent_run_result'] = 'agent_run_result'
    """Event type identifier, used as a discriminator."""

    __repr__ = _utils.dataclasses_no_defaults_repr
