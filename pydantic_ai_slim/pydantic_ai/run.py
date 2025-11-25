from __future__ import annotations as _annotations

import dataclasses
from collections.abc import AsyncIterator, Sequence
from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING, Any, Generic, Literal, overload

from pydantic_graph import BaseNode, End, GraphRunContext
from pydantic_graph.beta.graph import EndMarker, GraphRun, GraphTaskRequest, JoinItem
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

    agent = Agent('openai:gpt-4o')

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
                    run_id='...',
                )
            ),
            CallToolsNode(
                model_response=ModelResponse(
                    parts=[TextPart(content='The capital of France is Paris.')],
                    usage=RequestUsage(input_tokens=56, output_tokens=7),
                    model_name='gpt-4o',
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
        return self._task_to_node(task)

    @property
    def result(self) -> AgentRunResult[OutputDataT] | None:
        """The final result of the run if it has ended, otherwise `None`.

        Once the run returns an [`End`][pydantic_graph.nodes.End] node, `result` is populated
        with an [`AgentRunResult`][pydantic_ai.agent.AgentRunResult].
        """
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
        """Return new messages for the run so far.

        Messages from older runs are excluded.
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
        return self

    async def __anext__(
        self,
    ) -> _agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]:
        """Advance to the next node automatically based on the last returned node."""
        task = await anext(self._graph_run)
        return self._task_to_node(task)

    def _task_to_node(
        self, task: EndMarker[FinalResult[OutputDataT]] | JoinItem | Sequence[GraphTaskRequest]
    ) -> _agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]:
        if isinstance(task, Sequence) and len(task) == 1:
            first_task = task[0]
            if isinstance(first_task.inputs, BaseNode):  # pragma: no branch
                base_node: BaseNode[
                    _agent_graph.GraphAgentState,
                    _agent_graph.GraphAgentDeps[AgentDepsT, OutputDataT],
                    FinalResult[OutputDataT],
                ] = first_task.inputs  # type: ignore[reportUnknownMemberType]
                if _agent_graph.is_agent_node(node=base_node):  # pragma: no branch
                    return base_node
        if isinstance(task, EndMarker):
            return End(task.value)
        raise exceptions.AgentRunError(f'Unexpected node: {task}')  # pragma: no cover

    def _node_to_task(self, node: _agent_graph.AgentNode[AgentDepsT, OutputDataT]) -> GraphTaskRequest:
        return GraphTaskRequest(NodeStep(type(node)).id, inputs=node, fork_stack=())

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

        agent = Agent('openai:gpt-4o')

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
                            run_id='...',
                        )
                    ),
                    CallToolsNode(
                        model_response=ModelResponse(
                            parts=[TextPart(content='The capital of France is Paris.')],
                            usage=RequestUsage(input_tokens=56, output_tokens=7),
                            model_name='gpt-4o',
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
        task = [self._node_to_task(node)]
        try:
            task = await self._graph_run.next(task)
        except StopAsyncIteration:
            pass
        return self._task_to_node(task)

    # TODO (v2): Make this a property
    def usage(self) -> _usage.RunUsage:
        """Get usage statistics for the run so far, including token usage, model requests, and so on."""
        return self._graph_run.state.usage

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
        """Return new messages associated with this run.

        Messages from older runs are excluded.

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
