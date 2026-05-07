from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any, overload

from .. import (
    _instructions,
    _utils,
    messages as _messages,
    models,
    usage as _usage,
)
from .._json_schema import JsonSchema
from .._template import TemplateStr
from ..capabilities import AgentCapability
from ..output import OutputDataT, OutputSpec
from ..run import AgentRun
from ..settings import ModelSettings
from ..tools import (
    AgentBuiltinTool,
    AgentDepsT,
    DeferredToolResults,
    Tool,
    ToolFuncEither,
)
from ..toolsets import AbstractToolset
from .abstract import AbstractAgent, AgentMetadata, AgentModelSettings, EventStreamHandler, RunOutputDataT

if TYPE_CHECKING:
    from ..capabilities import CombinedCapability
    from .spec import AgentSpec


class WrapperAgent(AbstractAgent[AgentDepsT, OutputDataT]):
    """Agent which wraps another agent.

    Does nothing on its own, used as a base class.
    """

    def __init__(self, wrapped: AbstractAgent[AgentDepsT, OutputDataT]):
        self.wrapped = wrapped

    @property
    def model(self) -> models.Model | models.KnownModelName | str | None:
        return self.wrapped.model

    @property
    def name(self) -> str | None:
        return self.wrapped.name

    @name.setter
    def name(self, value: str | None) -> None:
        self.wrapped.name = value

    @property
    def description(self) -> str | None:
        return self.wrapped.description

    @description.setter
    def description(self, value: TemplateStr[AgentDepsT] | str | None) -> None:
        self.wrapped.description = value

    @property
    def deps_type(self) -> type:
        return self.wrapped.deps_type

    @property
    def output_type(self) -> OutputSpec[OutputDataT]:
        return self.wrapped.output_type

    @property
    def event_stream_handler(self) -> EventStreamHandler[AgentDepsT] | None:
        return self.wrapped.event_stream_handler

    @property
    def root_capability(self) -> CombinedCapability[AgentDepsT]:
        return self.wrapped.root_capability

    @property
    def toolsets(self) -> Sequence[AbstractToolset[AgentDepsT]]:
        return self.wrapped.toolsets

    async def __aenter__(self) -> AbstractAgent[AgentDepsT, OutputDataT]:
        return await self.wrapped.__aenter__()

    async def __aexit__(self, *args: Any) -> bool | None:
        return await self.wrapped.__aexit__(*args)

    def output_json_schema(self, output_type: OutputSpec[OutputDataT | RunOutputDataT] | None = None) -> JsonSchema:
        return self.wrapped.output_json_schema(output_type=output_type)

    async def system_prompt_parts(
        self,
        *,
        deps: AgentDepsT = None,
        model: models.Model | models.KnownModelName | str | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        prompt: str | Sequence[_messages.UserContent] | None = None,
        usage: _usage.RunUsage | None = None,
        model_settings: ModelSettings | None = None,
    ) -> list[_messages.SystemPromptPart]:
        return await self.wrapped.system_prompt_parts(
            deps=deps,
            model=model,
            message_history=message_history,
            prompt=prompt,
            usage=usage,
            model_settings=model_settings,
        )

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AgentBuiltinTool[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, OutputDataT]]: ...

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AgentBuiltinTool[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, RunOutputDataT]]: ...

    @asynccontextmanager
    async def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AgentBuiltinTool[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
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

        agent = Agent('openai:gpt-5.2')

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
                        ],
                        timestamp=datetime.datetime(...),
                        run_id='...',
                        conversation_id='...',
                    )
                ),
                CallToolsNode(
                    model_response=ModelResponse(
                        parts=[TextPart(content='The capital of France is Paris.')],
                        usage=RequestUsage(input_tokens=56, output_tokens=7),
                        model_name='gpt-5.2',
                        timestamp=datetime.datetime(...),
                        run_id='...',
                        conversation_id='...',
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
            conversation_id: ID of the conversation this run belongs to. Pass `'new'` to start a fresh conversation, ignoring any `conversation_id` already on `message_history`. If omitted, falls back to the most recent `conversation_id` on `message_history` or a freshly generated UUID7.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            metadata: Optional metadata to attach to this run.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools for this run.
            capabilities: Optional additional [capabilities](https://ai.pydantic.dev/capabilities/) for this run, merged with the agent's configured capabilities.
            spec: Optional agent spec to apply for this run.

        Returns:
            The result of the run.
        """
        async with self.wrapped.iter(
            user_prompt=user_prompt,
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            conversation_id=conversation_id,
            model=model,
            instructions=instructions,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            metadata=metadata,
            infer_name=infer_name,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
            capabilities=capabilities,
            spec=spec,
        ) as run:
            yield run

    @contextmanager
    def override(
        self,
        *,
        name: str | _utils.Unset = _utils.UNSET,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        model: models.Model | models.KnownModelName | str | _utils.Unset = _utils.UNSET,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | _utils.Unset = _utils.UNSET,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | _utils.Unset = _utils.UNSET,
        builtin_tools: Sequence[AgentBuiltinTool[AgentDepsT]] | _utils.Unset = _utils.UNSET,
        instructions: _instructions.AgentInstructions[AgentDepsT] | _utils.Unset = _utils.UNSET,
        model_settings: AgentModelSettings[AgentDepsT] | _utils.Unset = _utils.UNSET,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> Iterator[None]:
        """Context manager to temporarily override agent configuration.

        This is particularly useful when testing.
        You can find an example of this [here](../testing.md#overriding-model-via-pytest-fixtures).

        Args:
            name: The name to use instead of the name passed to the agent constructor and agent run.
            deps: The dependencies to use instead of the dependencies passed to the agent run.
            model: The model to use instead of the model passed to the agent run.
            toolsets: The toolsets to use instead of the toolsets passed to the agent constructor and agent run.
            tools: The tools to use instead of the tools registered with the agent.
            builtin_tools: The builtin tools to use instead of the agent's configured builtin tools.
                Per-run `builtin_tools` are still added to these.
            instructions: The instructions to use instead of the instructions registered with the agent.
            model_settings: The model settings to use instead of the model settings passed to the agent constructor.
                When set, any per-run `model_settings` argument is ignored.
            spec: Optional agent spec to apply as overrides.
        """
        with self.wrapped.override(
            name=name,
            deps=deps,
            model=model,
            toolsets=toolsets,
            tools=tools,
            builtin_tools=builtin_tools,
            instructions=instructions,
            model_settings=model_settings,
            spec=spec,
        ):
            yield
