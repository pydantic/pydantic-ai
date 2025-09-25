from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from typing import Any, Generic, overload
from uuid import uuid4

from hatchet_sdk import DurableContext, Hatchet, TriggerWorkflowOptions
from hatchet_sdk.runnables.workflow import BaseWorkflow
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from typing_extensions import Never

from pydantic_ai import _utils, messages as _messages, models, usage as _usage
from pydantic_ai.agent import AbstractAgent, AgentRun, AgentRunResult, EventStreamHandler, RunOutputDataT, WrapperAgent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model
from pydantic_ai.messages import ModelMessage
from pydantic_ai.output import OutputDataT, OutputSpec
from pydantic_ai.result import StreamedRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import (
    AgentDepsT,
    DeferredToolResults,
    Tool,
    ToolFuncEither,
)
from pydantic_ai.toolsets import AbstractToolset

from ._model import HatchetModel
from ._run_context import HatchetRunContext
from ._utils import TaskConfig


class RunAgentInput(BaseModel, Generic[RunOutputDataT, AgentDepsT]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_prompt: str | Sequence[_messages.UserContent] | None = None
    output_type: OutputSpec[RunOutputDataT] | None = None
    message_history: list[_messages.ModelMessage] | None = None
    deferred_tool_results: DeferredToolResults | None = None
    model: models.Model | models.KnownModelName | str | None = None
    deps: AgentDepsT
    model_settings: ModelSettings | None = None
    usage_limits: _usage.UsageLimits | None = None
    usage: _usage.RunUsage | None = None
    infer_name: bool = True
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None
    event_stream_handler: EventStreamHandler[AgentDepsT] | None = None
    deprecated_kwargs: dict[str, Any] = Field(default_factory=dict)


class HatchetAgent(WrapperAgent[AgentDepsT, OutputDataT]):
    def __init__(
        self,
        wrapped: AbstractAgent[AgentDepsT, OutputDataT],
        hatchet: Hatchet,
        *,
        name: str | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        mcp_task_config: TaskConfig | None = None,
        model_task_config: TaskConfig | None = None,
        run_context_type: type[HatchetRunContext[AgentDepsT]] = HatchetRunContext[AgentDepsT],
    ):
        """Wrap an agent to enable it with Hatchet durable tasks, by automatically offloading model requests, tool calls, and MCP server communication to Hatchet tasks.

        After wrapping, the original agent can still be used as normal outside of the Hatchet workflow.

        Args:
            wrapped: The agent to wrap.
            hatchet: The Hatchet instance to use for creating tasks.
            name: Optional unique agent name to use in the Hatchet tasks' names. If not provided, the agent's `name` will be used.
            event_stream_handler: Optional event stream handler to use for this agent.
            mcp_task_config: The base Hatchet task config to use for MCP server tasks. If no config is provided, use the default settings.
            model_task_config: The Hatchet task config to use for model request tasks. If no config is provided, use the default settings.
            run_context_type: The `HatchetRunContext` (sub)class that's used to serialize and deserialize the run context.
        """
        super().__init__(wrapped)

        self._name = name or wrapped.name
        self._event_stream_handler = event_stream_handler
        self.run_context_type: type[HatchetRunContext[AgentDepsT]] = run_context_type

        self._hatchet = hatchet

        if not self._name:
            raise UserError(
                "An agent needs to have a unique `name` in order to be used with Hatchet. The name will be used to identify the agent's workflows and tasks."
            )

        if not isinstance(wrapped.model, Model):
            raise UserError(
                'An agent needs to have a `model` in order to be used with Hatchet, it cannot be set at agent run time.'
            )

        self._model = HatchetModel(
            wrapped.model,
            task_name_prefix=self._name,
            task_config=model_task_config or TaskConfig(),
            hatchet=self._hatchet,
            event_stream_handler=self.event_stream_handler,
            deps_type=self.deps_type,
            run_context_type=self.run_context_type,
        )
        hatchet_agent_name = self._name

        def hatchetize_toolset(toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            from ._toolset import hatchetize_toolset

            return hatchetize_toolset(
                toolset,
                hatchet=hatchet,
                task_name_prefix=hatchet_agent_name,
                task_config=mcp_task_config or TaskConfig(),
                deps_type=self.deps_type,
                run_context_type=run_context_type,
            )

        self._toolsets = [toolset.visit_and_replace(hatchetize_toolset) for toolset in wrapped.toolsets]

        @hatchet.durable_task(name=f'{self._name}.run', input_validator=RunAgentInput[Any, Any])
        async def wrapped_run_workflow(
            input: RunAgentInput[RunOutputDataT, AgentDepsT],
            _ctx: DurableContext,
        ) -> AgentRunResult[Any]:
            with self._hatchet_overrides():
                return await super(WrapperAgent, self).run(
                    input.user_prompt,
                    output_type=input.output_type,
                    message_history=input.message_history,
                    deferred_tool_results=input.deferred_tool_results,
                    model=input.model,
                    deps=input.deps,
                    model_settings=input.model_settings,
                    usage_limits=input.usage_limits,
                    usage=input.usage,
                    infer_name=input.infer_name,
                    toolsets=input.toolsets,
                    event_stream_handler=input.event_stream_handler,
                    **input.deprecated_kwargs,
                )

        self.hatchet_wrapped_run_workflow = wrapped_run_workflow

        @hatchet.durable_task(name=f'{self._name}.run_stream', input_validator=RunAgentInput[Any, Any])
        async def wrapped_run_stream_workflow(
            input: RunAgentInput[RunOutputDataT, AgentDepsT],
            _ctx: DurableContext,
        ) -> AgentRunResult[Any]:
            return await wrapped.run(
                input.user_prompt,
                output_type=input.output_type,
                message_history=input.message_history,
                deferred_tool_results=input.deferred_tool_results,
                model=self._model,
                deps=input.deps,
                model_settings=input.model_settings,
                usage_limits=input.usage_limits,
                usage=input.usage,
                infer_name=input.infer_name,
                toolsets=self._toolsets,
                event_stream_handler=input.event_stream_handler,
                **input.deprecated_kwargs,
            )

        self.hatchet_wrapped_run_stream_workflow = wrapped_run_stream_workflow

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, value: str | None) -> None:  # pragma: no cover
        raise UserError(
            'The agent name cannot be changed after creation. If you need to change the name, create a new agent.'
        )

    @property
    def model(self) -> Model:
        return self._model

    @property
    def event_stream_handler(self) -> EventStreamHandler[AgentDepsT] | None:
        return self._event_stream_handler or super().event_stream_handler

    @property
    def toolsets(self) -> Sequence[AbstractToolset[AgentDepsT]]:
        with self._hatchet_overrides():
            return super().toolsets

    @contextmanager
    def _hatchet_overrides(self) -> Iterator[None]:
        with super().override(model=self._model, toolsets=self._toolsets, tools=[]):
            yield

    @property
    def workflows(self) -> list[BaseWorkflow[Any]]:
        workflows: list[BaseWorkflow[Any]] = [
            self.hatchet_wrapped_run_workflow,
            self.hatchet_wrapped_run_stream_workflow,
            self._model.hatchet_wrapped_request_task,
            self._model.hatchet_wrapped_request_stream_task,
        ]

        for toolset in self._toolsets:
            from ._toolset import HatchetWrapperToolset

            if isinstance(toolset, HatchetWrapperToolset):
                workflows.extend(toolset.hatchet_tasks)

        return workflows

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[OutputDataT]: ...

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[RunOutputDataT]: ...

    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AgentRunResult[Any]:
        """Run the agent with a user prompt in async mode."""
        agent_run_id = uuid4()

        result = await self.hatchet_wrapped_run_workflow.aio_run(
            RunAgentInput[RunOutputDataT, AgentDepsT](
                user_prompt=user_prompt,
                output_type=output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
                event_stream_handler=event_stream_handler,
                deprecated_kwargs=_deprecated_kwargs,
            ),
            options=TriggerWorkflowOptions(
                additional_metadata={
                    'hatchet__agent_name': self._name,
                    'hatchet__agent_run_id': str(agent_run_id),
                }
            ),
        )

        if isinstance(result, dict):
            return TypeAdapter(AgentRunResult[Any]).validate_python(result)

        return result

    @overload
    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[OutputDataT]: ...

    @overload
    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[RunOutputDataT]: ...

    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AgentRunResult[Any]:
        """Run the agent with a user prompt in sync mode."""
        agent_run_id = uuid4()

        result = self.hatchet_wrapped_run_workflow.run(
            RunAgentInput[RunOutputDataT, AgentDepsT](
                user_prompt=user_prompt,
                output_type=output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
                event_stream_handler=event_stream_handler,
                deprecated_kwargs=_deprecated_kwargs,
            ),
            options=TriggerWorkflowOptions(
                additional_metadata={
                    'hatchet__agent_name': self._name,
                    'hatchet__agent_run_id': str(agent_run_id),
                }
            ),
        )

        if isinstance(result, dict):
            return TypeAdapter(AgentRunResult[Any]).validate_python(result)

        return result

    @overload
    def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AbstractAsyncContextManager[StreamedRunResult[AgentDepsT, OutputDataT]]: ...

    @overload
    def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AbstractAsyncContextManager[StreamedRunResult[AgentDepsT, RunOutputDataT]]: ...

    @asynccontextmanager
    async def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AsyncIterator[StreamedRunResult[AgentDepsT, Any]]:
        """Run the agent with a user prompt in async mode, returning a streamed response.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            async with agent.run_stream('What is the capital of the UK?') as response:
                print(await response.get_output())
                #> The capital of the UK is London.
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            event_stream_handler: Optional event stream handler to use for this run. It will receive all the events up until the final result is found, which you can then read or stream from inside the context manager.

        Returns:
            The result of the run.
        """
        if self._hatchet.is_in_task_run:
            raise UserError(
                '`agent.run_stream()` cannot currently be used inside a Hatchet workflow. '
                'Set an `event_stream_handler` on the agent and use `agent.run()` instead. '
                'Please file an issue if this is not sufficient for your use case.'
            )

        # Execute the streaming via Hatchet workflow
        agent_run_id = uuid4()

        ref = await self.hatchet_wrapped_run_stream_workflow.aio_run_no_wait(
            RunAgentInput[RunOutputDataT, AgentDepsT](
                user_prompt=user_prompt,
                output_type=output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
                event_stream_handler=event_stream_handler,
                deprecated_kwargs=_deprecated_kwargs,
            ),
            options=TriggerWorkflowOptions(
                additional_metadata={
                    'hatchet__agent_name': self._name,
                    'hatchet__agent_run_id': str(agent_run_id),
                    'hatchet__stream_mode': True,
                }
            ),
        )

        all_messages: list[ModelMessage] = []

        async for x in self._hatchet.runs.subscribe_to_stream(ref.workflow_run_id):
            print('\nx', x)

        result = await ref.aio_result()

        if isinstance(result, dict):
            result = TypeAdapter(AgentRunResult[Any]).validate_python(result)

        messages = result.all_messages()
        new_message_index = result._new_message_index

        streamed_result = StreamedRunResult(
            all_messages=messages,
            new_message_index=new_message_index,
            run_result=result,
        )

        yield streamed_result

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, OutputDataT]]: ...

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, RunOutputDataT]]: ...

    @asynccontextmanager
    async def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AsyncIterator[AgentRun[AgentDepsT, Any]]:
        """A contextmanager which can be used to iterate over the agent graph's nodes as they are executed.

        This method builds an internal agent graph (using system prompts, tools and output schemas) and then returns an
        `AgentRun` object. The `AgentRun` can be used to async-iterate over the nodes of the graph as they are
        executed. This is the API to use if you want to consume the outputs coming from each LLM model response, or the
        stream of events coming from the execution of tools.

        The `AgentRun` also provides methods to access the full message history, new messages, and usage statistics,
        and the final result of the run once it has completed.

        For more details, see the documentation of `AgentRun`.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            nodes = []
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
                        ]
                    )
                ),
                CallToolsNode(
                    model_response=ModelResponse(
                        parts=[TextPart(content='The capital of France is Paris.')],
                        usage=RequestUsage(input_tokens=56, output_tokens=7),
                        model_name='gpt-4o',
                        timestamp=datetime.datetime(...),
                    )
                ),
                End(data=FinalResult(output='The capital of France is Paris.')),
            ]
            '''
            print(agent_run.result.output)
            #> The capital of France is Paris.
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.

        Returns:
            The result of the run.
        """
        if self._hatchet.is_in_task_run:
            raise UserError(
                '`agent.iter()` cannot currently be used inside a Hatchet workflow. '
                'Set an `event_stream_handler` on the agent and use `agent.run()` instead. '
                'Please file an issue if this is not sufficient for your use case.'
            )

        async with super().iter(
            user_prompt=user_prompt,
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            **_deprecated_kwargs,
        ) as run:
            yield run

    @contextmanager
    def override(
        self,
        *,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        model: models.Model | models.KnownModelName | str | _utils.Unset = _utils.UNSET,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | _utils.Unset = _utils.UNSET,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | _utils.Unset = _utils.UNSET,
    ) -> Iterator[None]:
        """Context manager to temporarily override agent dependencies, model, toolsets, or tools.

        This is particularly useful when testing.
        You can find an example of this [here](../testing.md#overriding-model-via-pytest-fixtures).

        Args:
            deps: The dependencies to use instead of the dependencies passed to the agent run.
            model: The model to use instead of the model passed to the agent run.
            toolsets: The toolsets to use instead of the toolsets passed to the agent constructor and agent run.
            tools: The tools to use instead of the tools registered with the agent.
        """
        with super().override(deps=deps, model=model, toolsets=toolsets, tools=tools):
            yield
