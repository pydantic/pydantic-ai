from __future__ import annotations as _annotations

import asyncio
import dataclasses
import inspect
from collections.abc import AsyncIterator, Awaitable, Generator, Iterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from copy import deepcopy
from types import FrameType
from typing import Any, Callable, Generic, cast, final, overload

import logfire_api
from typing_extensions import TypeVar, deprecated

from pydantic_graph import BaseNode, End, Graph, GraphRun, GraphRunContext, GraphRunner
from pydantic_graph.graph import GraphRunResult

from . import (
    _agent_graph,
    _result,
    _system_prompt,
    _utils,
    exceptions,
    messages as _messages,
    models,
    result,
    usage as _usage,
)
from ._agent_graph import EndStrategy, capture_run_messages  # imported for re-export
from .result import MarkFinalResult, ResultDataT, StreamedRunResult
from .settings import ModelSettings, merge_model_settings
from .tools import (
    AgentDepsT,
    DocstringFormat,
    RunContext,
    Tool,
    ToolFuncContext,
    ToolFuncEither,
    ToolFuncPlain,
    ToolParams,
    ToolPrepareFunc,
)

__all__ = 'Agent', 'capture_run_messages', 'EndStrategy'

_logfire = logfire_api.Logfire(otel_scope='pydantic-ai')

# while waiting for https://github.com/pydantic/logfire/issues/745
try:
    import logfire._internal.stack_info
except ImportError:
    pass
else:
    from pathlib import Path

    logfire._internal.stack_info.NON_USER_CODE_PREFIXES += (str(Path(__file__).parent.absolute()),)

T = TypeVar('T')
NoneType = type(None)
RunResultDataT = TypeVar('RunResultDataT')
"""Type variable for the result data of a run where `result_type` was customized on the run call."""


@final
@dataclasses.dataclass(init=False)
class Agent(Generic[AgentDepsT, ResultDataT]):
    """Class for defining "agents" - a way to have a specific type of "conversation" with an LLM.

    Agents are generic in the dependency type they take [`AgentDepsT`][pydantic_ai.tools.AgentDepsT]
    and the result data type they return, [`ResultDataT`][pydantic_ai.result.ResultDataT].

    By default, if neither generic parameter is customised, agents have type `Agent[None, str]`.

    Minimal usage example:

    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')
    result = agent.run_sync('What is the capital of France?')
    print(result.data)
    #> Paris
    ```
    """

    # we use dataclass fields in order to conveniently know what attributes are available
    model: models.Model | models.KnownModelName | None
    """The default model configured for this agent."""

    name: str | None
    """The name of the agent, used for logging.

    If `None`, we try to infer the agent name from the call frame when the agent is first run.
    """
    end_strategy: EndStrategy
    """Strategy for handling tool calls when a final result is found."""

    model_settings: ModelSettings | None
    """Optional model request settings to use for this agents's runs, by default.

    Note, if `model_settings` is provided by `run`, `run_sync`, or `run_stream`, those settings will
    be merged with this value, with the runtime argument taking priority.
    """

    result_type: type[ResultDataT] = dataclasses.field(repr=False)
    """
    The type of the result data, used to validate the result data, defaults to `str`.
    """

    _deps_type: type[AgentDepsT] = dataclasses.field(repr=False)
    _result_tool_name: str = dataclasses.field(repr=False)
    _result_tool_description: str | None = dataclasses.field(repr=False)
    _result_schema: _result.ResultSchema[ResultDataT] | None = dataclasses.field(repr=False)
    _result_validators: list[_result.ResultValidator[AgentDepsT, ResultDataT]] = dataclasses.field(repr=False)
    _system_prompts: tuple[str, ...] = dataclasses.field(repr=False)
    _system_prompt_functions: list[_system_prompt.SystemPromptRunner[AgentDepsT]] = dataclasses.field(repr=False)
    _system_prompt_dynamic_functions: dict[str, _system_prompt.SystemPromptRunner[AgentDepsT]] = dataclasses.field(
        repr=False
    )
    _function_tools: dict[str, Tool[AgentDepsT]] = dataclasses.field(repr=False)
    _default_retries: int = dataclasses.field(repr=False)
    _max_result_retries: int = dataclasses.field(repr=False)
    _override_deps: _utils.Option[AgentDepsT] = dataclasses.field(default=None, repr=False)
    _override_model: _utils.Option[models.Model] = dataclasses.field(default=None, repr=False)

    def __init__(
        self,
        model: models.Model | models.KnownModelName | None = None,
        *,
        result_type: type[ResultDataT] = str,
        system_prompt: str | Sequence[str] = (),
        deps_type: type[AgentDepsT] = NoneType,
        name: str | None = None,
        model_settings: ModelSettings | None = None,
        retries: int = 1,
        result_tool_name: str = 'final_result',
        result_tool_description: str | None = None,
        result_retries: int | None = None,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
        defer_model_check: bool = False,
        end_strategy: EndStrategy = 'early',
    ):
        """Create an agent.

        Args:
            model: The default model to use for this agent, if not provide,
                you must provide the model when calling it.
            result_type: The type of the result data, used to validate the result data, defaults to `str`.
            system_prompt: Static system prompts to use for this agent, you can also register system
                prompts via a function with [`system_prompt`][pydantic_ai.Agent.system_prompt].
            deps_type: The type used for dependency injection, this parameter exists solely to allow you to fully
                parameterize the agent, and therefore get the best out of static type checking.
                If you're not using deps, but want type checking to pass, you can set `deps=None` to satisfy Pyright
                or add a type hint `: Agent[None, <return type>]`.
            name: The name of the agent, used for logging. If `None`, we try to infer the agent name from the call frame
                when the agent is first run.
            model_settings: Optional model request settings to use for this agent's runs, by default.
            retries: The default number of retries to allow before raising an error.
            result_tool_name: The name of the tool to use for the final result.
            result_tool_description: The description of the final result tool.
            result_retries: The maximum number of retries to allow for result validation, defaults to `retries`.
            tools: Tools to register with the agent, you can also register tools via the decorators
                [`@agent.tool`][pydantic_ai.Agent.tool] and [`@agent.tool_plain`][pydantic_ai.Agent.tool_plain].
            defer_model_check: by default, if you provide a [named][pydantic_ai.models.KnownModelName] model,
                it's evaluated to create a [`Model`][pydantic_ai.models.Model] instance immediately,
                which checks for the necessary environment variables. Set this to `false`
                to defer the evaluation until the first run. Useful if you want to
                [override the model][pydantic_ai.Agent.override] for testing.
            end_strategy: Strategy for handling tool calls that are requested alongside a final result.
                See [`EndStrategy`][pydantic_ai.agent.EndStrategy] for more information.
        """
        if model is None or defer_model_check:
            self.model = model
        else:
            self.model = models.infer_model(model)

        self.end_strategy = end_strategy
        self.name = name
        self.model_settings = model_settings
        self.result_type = result_type

        self._deps_type = deps_type

        self._result_tool_name = result_tool_name
        self._result_tool_description = result_tool_description
        self._result_schema: _result.ResultSchema[ResultDataT] | None = _result.ResultSchema[result_type].build(
            result_type, result_tool_name, result_tool_description
        )
        self._result_validators: list[_result.ResultValidator[AgentDepsT, ResultDataT]] = []

        self._system_prompts = (system_prompt,) if isinstance(system_prompt, str) else tuple(system_prompt)
        self._system_prompt_functions: list[_system_prompt.SystemPromptRunner[AgentDepsT]] = []
        self._system_prompt_dynamic_functions: dict[str, _system_prompt.SystemPromptRunner[AgentDepsT]] = {}

        self._function_tools: dict[str, Tool[AgentDepsT]] = {}

        self._default_retries = retries
        self._max_result_retries = result_retries if result_retries is not None else retries
        for tool in tools:
            if isinstance(tool, Tool):
                self._register_tool(tool)
            else:
                self._register_tool(Tool(tool))

    @overload
    def run(
        self,
        user_prompt: str,
        *,
        result_type: None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> AgentRunner[AgentDepsT, ResultDataT]: ...

    @overload
    def run(
        self,
        user_prompt: str,
        *,
        result_type: type[RunResultDataT],
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> AgentRunner[AgentDepsT, RunResultDataT]: ...

    def run(
        self,
        user_prompt: str,
        *,
        result_type: type[RunResultDataT] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> AgentRunner[AgentDepsT, Any]:
        """Run the agent with a user prompt.

        This method builds an internal agent graph (using system prompts, tools and result schemas) and then
        returns an `AgentRun` object. The `AgentRun` functions as a handle that can be used to iterate over the graph
        and obtain the final result. The AgentRun also provides methods to access the full message history,
        new messages, and usage statistics.

        The returned `AgentRun` object should always be immediately used in one of two ways:
        * Via `await` (i.e., `await agent.run(...)`), which will execute the graph run and return the final result
            * This is the API you should use if you just want the end result and are not interested in the execution details
            or consuming streaming updates
        * As a context manager (i.e., `with agent.run(...) as agent_run:`), which will return an async iterator over
        the graph nodes, and which can also be used as an async generator to override the execution of the graph.
            * This is the API you should use if you want to consume the graph execution in a streaming manner,
            or if you want to consume the stream of events coming from individual requests to the LLM, or the stream
            of events coming from the execution of tools.

        For more details, see the documentation of `AgentRun`.

        Example (with `await`):
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            agent_run = await agent.run('What is the capital of France?')
            print(agent_run.data)
            #> Paris
        ```

        Example (with a context manager):
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            nodes = []
            with agent.run('What is the capital of France?') as agent_run:
                async for node in agent_run:
                    nodes.append(node)
            print(nodes)
            '''
            [
                ModelRequestNode(
                    request=ModelRequest(
                        parts=[
                            UserPromptPart(
                                content='What is the capital of France?',
                                timestamp=datetime.datetime(...),
                                part_kind='user-prompt',
                            )
                        ],
                        kind='request',
                    )
                ),
                HandleResponseNode(
                    model_response=ModelResponse(
                        parts=[TextPart(content='Paris', part_kind='text')],
                        model_name='function:model_logic',
                        timestamp=datetime.datetime(...),
                        kind='response',
                    )
                ),
                End(data=MarkFinalResult(data='Paris', tool_name=None)),
            ]
            '''
            print(agent_run.final_result.data)
            #> Paris
        ```

        Args:
            result_type: Custom result type to use for this run, `result_type` may only be used if the agent has no
                result validators since result validators would expect an argument that matches the agent's result type.
            user_prompt: User input to start/continue the conversation.
            message_history: History of the conversation so far.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.

        Returns:
            The result of the run.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())
        model_used = self._get_model(model)

        deps = self._get_deps(deps)
        new_message_index = len(message_history) if message_history else 0
        result_schema: _result.ResultSchema[RunResultDataT] | None = self._prepare_result_schema(result_type)

        # Build the graph
        graph = self._build_graph(result_type)

        # Build the initial state
        state = _agent_graph.GraphAgentState(
            message_history=message_history[:] if message_history else [],
            usage=usage or _usage.Usage(),
            retries=0,
            run_step=0,
        )

        # We consider it a user error if a user tries to restrict the result type while having a result validator that
        # may change the result type from the restricted type to something else. Therefore, we consider the following
        # typecast reasonable, even though it is possible to violate it with otherwise-type-checked code.
        result_validators = cast(list[_result.ResultValidator[AgentDepsT, RunResultDataT]], self._result_validators)

        # TODO: Instead of this, copy the function tools to ensure they don't share current_retry state between agent
        #  runs. Requires some changes to `Tool` to make them copyable though.
        for v in self._function_tools.values():
            v.current_retry = 0

        model_settings = merge_model_settings(self.model_settings, model_settings)
        usage_limits = usage_limits or _usage.UsageLimits()

        # Build the deps object for the graph
        run_span = _logfire.span(
            '{agent_name} run {prompt=}',
            prompt=user_prompt,
            agent=self,
            model_name=model_used.model_name if model_used else 'no-model',
            agent_name=self.name or 'agent',
        )
        graph_deps = _agent_graph.GraphAgentDeps[AgentDepsT, RunResultDataT](
            user_deps=deps,
            prompt=user_prompt,
            new_message_index=new_message_index,
            model=model_used,
            model_settings=model_settings,
            usage_limits=usage_limits,
            max_result_retries=self._max_result_retries,
            end_strategy=self.end_strategy,
            result_schema=result_schema,
            result_tools=self._result_schema.tool_defs() if self._result_schema else [],
            result_validators=result_validators,
            function_tools=self._function_tools,
            run_span=run_span,
        )
        start_node = _agent_graph.UserPromptNode[AgentDepsT](
            user_prompt=user_prompt,
            system_prompts=self._system_prompts,
            system_prompt_functions=self._system_prompt_functions,
            system_prompt_dynamic_functions=self._system_prompt_dynamic_functions,
        )

        return AgentRunner(
            graph.run(
                start_node,
                state=state,
                deps=graph_deps,
                infer_name=False,
                span=run_span,
            )
        )

    @overload
    def run_sync(
        self,
        user_prompt: str,
        *,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> AgentRunResult[AgentDepsT, ResultDataT]: ...

    @overload
    def run_sync(
        self,
        user_prompt: str,
        *,
        result_type: type[RunResultDataT] | None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> AgentRunResult[AgentDepsT, RunResultDataT]: ...

    def run_sync(
        self,
        user_prompt: str,
        *,
        result_type: type[RunResultDataT] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> AgentRunResult[AgentDepsT, Any]:
        """Run the agent with a user prompt synchronously.

        This is a convenience method that wraps [`self.run`][pydantic_ai.Agent.run] with `loop.run_until_complete(...)`.
        You therefore can't use this method inside async code or if there's an active event loop.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        result_sync = agent.run_sync('What is the capital of Italy?')
        print(result_sync.data)
        #> Rome
        ```

        Args:
            result_type: Custom result type to use for this run, `result_type` may only be used if the agent has no
                result validators since result validators would expect an argument that matches the agent's result type.
            user_prompt: User input to start/continue the conversation.
            message_history: History of the conversation so far.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.

        Returns:
            The result of the run.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())
        return asyncio.get_event_loop().run_until_complete(
            self.run(
                user_prompt,
                result_type=result_type,
                message_history=message_history,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=False,
            )
        )

    @overload
    def run_stream(
        self,
        user_prompt: str,
        *,
        result_type: None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> AbstractAsyncContextManager[result.StreamedRunResult[AgentDepsT, ResultDataT]]: ...

    @overload
    def run_stream(
        self,
        user_prompt: str,
        *,
        result_type: type[RunResultDataT],
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> AbstractAsyncContextManager[result.StreamedRunResult[AgentDepsT, RunResultDataT]]: ...

    @asynccontextmanager
    async def run_stream(  # noqa
        self,
        user_prompt: str,
        *,
        result_type: type[RunResultDataT] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> AsyncIterator[result.StreamedRunResult[AgentDepsT, Any]]:
        """Run the agent with a user prompt in async mode, returning a streamed response.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            async with agent.run_stream('What is the capital of the UK?') as response:
                print(await response.get_data())
                #> London
        ```

        Args:
            result_type: Custom result type to use for this run, `result_type` may only be used if the agent has no
                result validators since result validators would expect an argument that matches the agent's result type.
            user_prompt: User input to start/continue the conversation.
            message_history: History of the conversation so far.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.

        Returns:
            The result of the run.
        """
        if infer_name and self.name is None:
            # f_back because `asynccontextmanager` adds one frame
            if frame := inspect.currentframe():  # pragma: no branch
                self._infer_name(frame.f_back)

        yielded = False
        with self.run(
            user_prompt,
            result_type=result_type,
            message_history=message_history,
            model=model,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=False,
        ) as agent_run:
            first_node = await agent_run.__anext__()
            assert isinstance(first_node, _agent_graph.ModelRequestNode)  # the first node should be a request node
            node: BaseNode[Any, Any, Any] = cast(BaseNode[Any, Any, Any], first_node)
            while True:
                if isinstance(node, _agent_graph.ModelRequestNode):
                    node = cast(_agent_graph.ModelRequestNode[AgentDepsT, Any], node)
                    graph_ctx = agent_run.get_graph_ctx()
                    async with node.stream(graph_ctx) as streamed_response:

                        async def stream_to_final(
                            s: models.StreamedResponse,
                        ) -> MarkFinalResult[models.StreamedResponse] | None:
                            result_schema = graph_ctx.deps.result_schema
                            async for maybe_part_event in streamed_response:
                                if isinstance(maybe_part_event, _messages.PartStartEvent):
                                    new_part = maybe_part_event.part
                                    if isinstance(new_part, _messages.TextPart):
                                        if _agent_graph.allow_text_result(result_schema):
                                            return MarkFinalResult(s, None)
                                    elif isinstance(new_part, _messages.ToolCallPart):
                                        if result_schema is not None and (match := result_schema.find_tool([new_part])):
                                            call, _ = match
                                            return MarkFinalResult(s, call.tool_name)
                            return None

                        final_result_details = await stream_to_final(streamed_response)
                        if final_result_details is not None:
                            if yielded:
                                raise exceptions.AgentRunError('Agent run produced final results')
                            yielded = True

                            messages = graph_ctx.state.message_history.copy()

                            async def on_complete() -> None:
                                """Called when the stream has completed.

                                The model response will have been added to messages by now
                                by `StreamedRunResult._marked_completed`.
                                """
                                last_message = messages[-1]
                                assert isinstance(last_message, _messages.ModelResponse)
                                tool_calls = [
                                    part for part in last_message.parts if isinstance(part, _messages.ToolCallPart)
                                ]

                                parts: list[_messages.ModelRequestPart] = []
                                async for _event in _agent_graph.process_function_tools(
                                    tool_calls,
                                    final_result_details.tool_name,
                                    graph_ctx,
                                    parts,
                                ):
                                    pass
                                # TODO: Should we do something here related to the retry count?
                                #   Maybe we should move the incrementing of the retry count to where we actually make a request?
                                # if any(isinstance(part, _messages.RetryPromptPart) for part in parts):
                                #     ctx.state.increment_retries(ctx.deps.max_result_retries)
                                if parts:
                                    messages.append(_messages.ModelRequest(parts))

                            yield StreamedRunResult(
                                messages,
                                graph_ctx.deps.new_message_index,
                                graph_ctx.deps.usage_limits,
                                streamed_response,
                                graph_ctx.deps.result_schema,
                                _agent_graph.build_run_context(graph_ctx),
                                graph_ctx.deps.result_validators,
                                final_result_details.tool_name,
                                on_complete,
                            )
                            break
                next_node = await agent_run.next(node)
                if not isinstance(next_node, BaseNode):
                    raise exceptions.AgentRunError('Should have produced a StreamedRunResult before getting here')
                node = cast(BaseNode[Any, Any, Any], next_node)

        if not yielded:
            raise exceptions.AgentRunError('Agent run finished without producing a final result')

    @contextmanager
    def override(
        self,
        *,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        model: models.Model | models.KnownModelName | _utils.Unset = _utils.UNSET,
    ) -> Iterator[None]:
        """Context manager to temporarily override agent dependencies and model.

        This is particularly useful when testing.
        You can find an example of this [here](../testing-evals.md#overriding-model-via-pytest-fixtures).

        Args:
            deps: The dependencies to use instead of the dependencies passed to the agent run.
            model: The model to use instead of the model passed to the agent run.
        """
        if _utils.is_set(deps):
            override_deps_before = self._override_deps
            self._override_deps = _utils.Some(deps)
        else:
            override_deps_before = _utils.UNSET

        # noinspection PyTypeChecker
        if _utils.is_set(model):
            override_model_before = self._override_model
            # noinspection PyTypeChecker
            self._override_model = _utils.Some(models.infer_model(model))  # pyright: ignore[reportArgumentType]
        else:
            override_model_before = _utils.UNSET

        try:
            yield
        finally:
            if _utils.is_set(override_deps_before):
                self._override_deps = override_deps_before
            if _utils.is_set(override_model_before):
                self._override_model = override_model_before

    @overload
    def system_prompt(
        self, func: Callable[[RunContext[AgentDepsT]], str], /
    ) -> Callable[[RunContext[AgentDepsT]], str]: ...

    @overload
    def system_prompt(
        self, func: Callable[[RunContext[AgentDepsT]], Awaitable[str]], /
    ) -> Callable[[RunContext[AgentDepsT]], Awaitable[str]]: ...

    @overload
    def system_prompt(self, func: Callable[[], str], /) -> Callable[[], str]: ...

    @overload
    def system_prompt(self, func: Callable[[], Awaitable[str]], /) -> Callable[[], Awaitable[str]]: ...

    @overload
    def system_prompt(
        self, /, *, dynamic: bool = False
    ) -> Callable[[_system_prompt.SystemPromptFunc[AgentDepsT]], _system_prompt.SystemPromptFunc[AgentDepsT]]: ...

    def system_prompt(
        self,
        func: _system_prompt.SystemPromptFunc[AgentDepsT] | None = None,
        /,
        *,
        dynamic: bool = False,
    ) -> (
        Callable[[_system_prompt.SystemPromptFunc[AgentDepsT]], _system_prompt.SystemPromptFunc[AgentDepsT]]
        | _system_prompt.SystemPromptFunc[AgentDepsT]
    ):
        """Decorator to register a system prompt function.

        Optionally takes [`RunContext`][pydantic_ai.tools.RunContext] as its only argument.
        Can decorate a sync or async functions.

        The decorator can be used either bare (`agent.system_prompt`) or as a function call
        (`agent.system_prompt(...)`), see the examples below.

        Overloads for every possible signature of `system_prompt` are included so the decorator doesn't obscure
        the type of the function, see `tests/typed_agent.py` for tests.

        Args:
            func: The function to decorate
            dynamic: If True, the system prompt will be reevaluated even when `messages_history` is provided,
                see [`SystemPromptPart.dynamic_ref`][pydantic_ai.messages.SystemPromptPart.dynamic_ref]

        Example:
        ```python
        from pydantic_ai import Agent, RunContext

        agent = Agent('test', deps_type=str)

        @agent.system_prompt
        def simple_system_prompt() -> str:
            return 'foobar'

        @agent.system_prompt(dynamic=True)
        async def async_system_prompt(ctx: RunContext[str]) -> str:
            return f'{ctx.deps} is the best'
        ```
        """
        if func is None:

            def decorator(
                func_: _system_prompt.SystemPromptFunc[AgentDepsT],
            ) -> _system_prompt.SystemPromptFunc[AgentDepsT]:
                runner = _system_prompt.SystemPromptRunner[AgentDepsT](func_, dynamic=dynamic)
                self._system_prompt_functions.append(runner)
                if dynamic:
                    self._system_prompt_dynamic_functions[func_.__qualname__] = runner
                return func_

            return decorator
        else:
            assert not dynamic, "dynamic can't be True in this case"
            self._system_prompt_functions.append(_system_prompt.SystemPromptRunner[AgentDepsT](func, dynamic=dynamic))
            return func

    @overload
    def result_validator(
        self, func: Callable[[RunContext[AgentDepsT], ResultDataT], ResultDataT], /
    ) -> Callable[[RunContext[AgentDepsT], ResultDataT], ResultDataT]: ...

    @overload
    def result_validator(
        self, func: Callable[[RunContext[AgentDepsT], ResultDataT], Awaitable[ResultDataT]], /
    ) -> Callable[[RunContext[AgentDepsT], ResultDataT], Awaitable[ResultDataT]]: ...

    @overload
    def result_validator(
        self, func: Callable[[ResultDataT], ResultDataT], /
    ) -> Callable[[ResultDataT], ResultDataT]: ...

    @overload
    def result_validator(
        self, func: Callable[[ResultDataT], Awaitable[ResultDataT]], /
    ) -> Callable[[ResultDataT], Awaitable[ResultDataT]]: ...

    def result_validator(
        self, func: _result.ResultValidatorFunc[AgentDepsT, ResultDataT], /
    ) -> _result.ResultValidatorFunc[AgentDepsT, ResultDataT]:
        """Decorator to register a result validator function.

        Optionally takes [`RunContext`][pydantic_ai.tools.RunContext] as its first argument.
        Can decorate a sync or async functions.

        Overloads for every possible signature of `result_validator` are included so the decorator doesn't obscure
        the type of the function, see `tests/typed_agent.py` for tests.

        Example:
        ```python
        from pydantic_ai import Agent, ModelRetry, RunContext

        agent = Agent('test', deps_type=str)

        @agent.result_validator
        def result_validator_simple(data: str) -> str:
            if 'wrong' in data:
                raise ModelRetry('wrong response')
            return data

        @agent.result_validator
        async def result_validator_deps(ctx: RunContext[str], data: str) -> str:
            if ctx.deps in data:
                raise ModelRetry('wrong response')
            return data

        result = agent.run_sync('foobar', deps='spam')
        print(result.data)
        #> success (no tool calls)
        ```
        """
        self._result_validators.append(_result.ResultValidator[AgentDepsT, Any](func))
        return func

    @overload
    def tool(self, func: ToolFuncContext[AgentDepsT, ToolParams], /) -> ToolFuncContext[AgentDepsT, ToolParams]: ...

    @overload
    def tool(
        self,
        /,
        *,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
    ) -> Callable[[ToolFuncContext[AgentDepsT, ToolParams]], ToolFuncContext[AgentDepsT, ToolParams]]: ...

    def tool(
        self,
        func: ToolFuncContext[AgentDepsT, ToolParams] | None = None,
        /,
        *,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
    ) -> Any:
        """Decorator to register a tool function which takes [`RunContext`][pydantic_ai.tools.RunContext] as its first argument.

        Can decorate a sync or async functions.

        The docstring is inspected to extract both the tool description and description of each parameter,
        [learn more](../tools.md#function-tools-and-schema).

        We can't add overloads for every possible signature of tool, since the return type is a recursive union
        so the signature of functions decorated with `@agent.tool` is obscured.

        Example:
        ```python
        from pydantic_ai import Agent, RunContext

        agent = Agent('test', deps_type=int)

        @agent.tool
        def foobar(ctx: RunContext[int], x: int) -> int:
            return ctx.deps + x

        @agent.tool(retries=2)
        async def spam(ctx: RunContext[str], y: float) -> float:
            return ctx.deps + y

        result = agent.run_sync('foobar', deps=1)
        print(result.data)
        #> {"foobar":1,"spam":1.0}
        ```

        Args:
            func: The tool function to register.
            retries: The number of retries to allow for this tool, defaults to the agent's default retries,
                which defaults to 1.
            prepare: custom method to prepare the tool definition for each step, return `None` to omit this
                tool from a given step. This is useful if you want to customise a tool at call time,
                or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
            docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
                Defaults to `'auto'`, such that the format is inferred from the structure of the docstring.
            require_parameter_descriptions: If True, raise an error if a parameter description is missing. Defaults to False.
        """
        if func is None:

            def tool_decorator(
                func_: ToolFuncContext[AgentDepsT, ToolParams],
            ) -> ToolFuncContext[AgentDepsT, ToolParams]:
                # noinspection PyTypeChecker
                self._register_function(func_, True, retries, prepare, docstring_format, require_parameter_descriptions)
                return func_

            return tool_decorator
        else:
            # noinspection PyTypeChecker
            self._register_function(func, True, retries, prepare, docstring_format, require_parameter_descriptions)
            return func

    @overload
    def tool_plain(self, func: ToolFuncPlain[ToolParams], /) -> ToolFuncPlain[ToolParams]: ...

    @overload
    def tool_plain(
        self,
        /,
        *,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
    ) -> Callable[[ToolFuncPlain[ToolParams]], ToolFuncPlain[ToolParams]]: ...

    def tool_plain(
        self,
        func: ToolFuncPlain[ToolParams] | None = None,
        /,
        *,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
    ) -> Any:
        """Decorator to register a tool function which DOES NOT take `RunContext` as an argument.

        Can decorate a sync or async functions.

        The docstring is inspected to extract both the tool description and description of each parameter,
        [learn more](../tools.md#function-tools-and-schema).

        We can't add overloads for every possible signature of tool, since the return type is a recursive union
        so the signature of functions decorated with `@agent.tool` is obscured.

        Example:
        ```python
        from pydantic_ai import Agent, RunContext

        agent = Agent('test')

        @agent.tool
        def foobar(ctx: RunContext[int]) -> int:
            return 123

        @agent.tool(retries=2)
        async def spam(ctx: RunContext[str]) -> float:
            return 3.14

        result = agent.run_sync('foobar', deps=1)
        print(result.data)
        #> {"foobar":123,"spam":3.14}
        ```

        Args:
            func: The tool function to register.
            retries: The number of retries to allow for this tool, defaults to the agent's default retries,
                which defaults to 1.
            prepare: custom method to prepare the tool definition for each step, return `None` to omit this
                tool from a given step. This is useful if you want to customise a tool at call time,
                or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
            docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
                Defaults to `'auto'`, such that the format is inferred from the structure of the docstring.
            require_parameter_descriptions: If True, raise an error if a parameter description is missing. Defaults to False.
        """
        if func is None:

            def tool_decorator(func_: ToolFuncPlain[ToolParams]) -> ToolFuncPlain[ToolParams]:
                # noinspection PyTypeChecker
                self._register_function(
                    func_, False, retries, prepare, docstring_format, require_parameter_descriptions
                )
                return func_

            return tool_decorator
        else:
            self._register_function(func, False, retries, prepare, docstring_format, require_parameter_descriptions)
            return func

    def _register_function(
        self,
        func: ToolFuncEither[AgentDepsT, ToolParams],
        takes_ctx: bool,
        retries: int | None,
        prepare: ToolPrepareFunc[AgentDepsT] | None,
        docstring_format: DocstringFormat,
        require_parameter_descriptions: bool,
    ) -> None:
        """Private utility to register a function as a tool."""
        retries_ = retries if retries is not None else self._default_retries
        tool = Tool[AgentDepsT](
            func,
            takes_ctx=takes_ctx,
            max_retries=retries_,
            prepare=prepare,
            docstring_format=docstring_format,
            require_parameter_descriptions=require_parameter_descriptions,
        )
        self._register_tool(tool)

    def _register_tool(self, tool: Tool[AgentDepsT]) -> None:
        """Private utility to register a tool instance."""
        if tool.max_retries is None:
            # noinspection PyTypeChecker
            tool = dataclasses.replace(tool, max_retries=self._default_retries)

        if tool.name in self._function_tools:
            raise exceptions.UserError(f'Tool name conflicts with existing tool: {tool.name!r}')

        if self._result_schema and tool.name in self._result_schema.tools:
            raise exceptions.UserError(f'Tool name conflicts with result schema name: {tool.name!r}')

        self._function_tools[tool.name] = tool

    def _get_model(self, model: models.Model | models.KnownModelName | None) -> models.Model:
        """Create a model configured for this agent.

        Args:
            model: model to use for this run, required if `model` was not set when creating the agent.

        Returns:
            The model used
        """
        model_: models.Model
        if some_model := self._override_model:
            # we don't want `override()` to cover up errors from the model not being defined, hence this check
            if model is None and self.model is None:
                raise exceptions.UserError(
                    '`model` must be set either when creating the agent or when calling it. '
                    '(Even when `override(model=...)` is customizing the model that will actually be called)'
                )
            model_ = some_model.value
        elif model is not None:
            model_ = models.infer_model(model)
        elif self.model is not None:
            # noinspection PyTypeChecker
            model_ = self.model = models.infer_model(self.model)
        else:
            raise exceptions.UserError('`model` must be set either when creating the agent or when calling it.')

        return model_

    def _get_deps(self: Agent[T, ResultDataT], deps: T) -> T:
        """Get deps for a run.

        If we've overridden deps via `_override_deps`, use that, otherwise use the deps passed to the call.

        We could do runtime type checking of deps against `self._deps_type`, but that's a slippery slope.
        """
        if some_deps := self._override_deps:
            return some_deps.value
        else:
            return deps

    def _infer_name(self, function_frame: FrameType | None) -> None:
        """Infer the agent name from the call frame.

        Usage should be `self._infer_name(inspect.currentframe())`.
        """
        assert self.name is None, 'Name already set'
        if function_frame is not None:  # pragma: no branch
            if parent_frame := function_frame.f_back:  # pragma: no branch
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

    @property
    @deprecated(
        'The `last_run_messages` attribute has been removed, use `capture_run_messages` instead.', category=None
    )
    def last_run_messages(self) -> list[_messages.ModelMessage]:
        raise AttributeError('The `last_run_messages` attribute has been removed, use `capture_run_messages` instead.')

    def _build_graph(
        self, result_type: type[RunResultDataT] | None
    ) -> Graph[_agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any], MarkFinalResult[Any]]:
        return _agent_graph.build_agent_graph(self.name, self._deps_type, result_type or self.result_type)

    def _prepare_result_schema(
        self, result_type: type[RunResultDataT] | None
    ) -> _result.ResultSchema[RunResultDataT] | None:
        if result_type is not None:
            if self._result_validators:
                raise exceptions.UserError('Cannot set a custom run `result_type` when the agent has result validators')
            return _result.ResultSchema[result_type].build(
                result_type, self._result_tool_name, self._result_tool_description
            )
        else:
            return self._result_schema  # pyright: ignore[reportReturnType]


@dataclasses.dataclass
class AgentRunner(Generic[AgentDepsT, ResultDataT]):
    graph_runner: GraphRunner[
        _agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any], MarkFinalResult[ResultDataT]
    ]

    def __await__(self) -> Generator[Any, Any, AgentRunResult[AgentDepsT, ResultDataT]]:
        """Run the agent graph until it ends, and return self."""

        async def _run():
            graph_run_result = await self.graph_runner
            return AgentRunResult(graph_run_result)

        return _run().__await__()

    def __enter__(self) -> AgentRun[AgentDepsT, ResultDataT]:
        graph_run = self.graph_runner.__enter__()
        return AgentRun(graph_run)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.graph_runner.__exit__(exc_type, exc_val, exc_tb)


@dataclasses.dataclass
class AgentRun(Generic[AgentDepsT, ResultDataT]):
    """A stateful, iterable run of an agent."""

    _graph_run: GraphRun[
        _agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any], MarkFinalResult[ResultDataT]
    ]

    @property
    def final_result(self) -> AgentRunResult[AgentDepsT, ResultDataT]:
        graph_run_result = self._graph_run.final_result
        return AgentRunResult(graph_run_result)

    def __aiter__(
        self,
    ) -> AsyncIterator[
        BaseNode[
            _agent_graph.GraphAgentState,
            _agent_graph.GraphAgentDeps[AgentDepsT, Any],
            MarkFinalResult[ResultDataT],
        ]
        | End[MarkFinalResult[ResultDataT]]
    ]:
        return self

    async def __anext__(
        self,
    ) -> (
        BaseNode[
            _agent_graph.GraphAgentState,
            _agent_graph.GraphAgentDeps[AgentDepsT, Any],
            MarkFinalResult[ResultDataT],
        ]
        | End[MarkFinalResult[ResultDataT]]
    ):
        """Use the last returned node as the input to `Graph.next`."""
        return await self._graph_run.__anext__()

    async def next(
        self,
        node: BaseNode[
            _agent_graph.GraphAgentState,
            _agent_graph.GraphAgentDeps[AgentDepsT, Any],
            MarkFinalResult[ResultDataT],
        ],
    ) -> (
        BaseNode[
            _agent_graph.GraphAgentState,
            _agent_graph.GraphAgentDeps[AgentDepsT, Any],
            MarkFinalResult[ResultDataT],
        ]
        | End[MarkFinalResult[ResultDataT]]
    ):
        # TODO: It would be nice to expose a synchronous interface for this, to be able to
        #   synchronously iterate over the agent graph. I don't think this would be hard to do,
        #   but I'm having a hard time coming up with an API that fits nicely along side the current `run_sync`.
        #   The use of `await` provides an easy way to signal that you just want the result, but it's less
        #   clear to me what the analogous thing should be for synchronous code.
        return await self._graph_run.next(node)

    def usage(self) -> _usage.Usage:
        return self._graph_run.state.usage

    def get_graph_ctx(
        self,
    ) -> GraphRunContext[_agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any]]:
        return GraphRunContext[_agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any]](
            self._graph_run.state, self._graph_run.deps
        )


@dataclasses.dataclass
class AgentRunResult(Generic[AgentDepsT, ResultDataT]):
    """The final result of an agent run."""

    graph_run_result: GraphRunResult[
        _agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any], MarkFinalResult[ResultDataT]
    ]

    @property
    def result(self) -> MarkFinalResult[ResultDataT]:
        return self.graph_run_result.result

    @property
    def data(self) -> ResultDataT:
        return self.result.data

    @property
    def _result_tool_name(self) -> str | None:
        return self.result.tool_name

    def _set_result_tool_return(self, return_content: str) -> list[_messages.ModelMessage]:
        """Set return content for the result tool.

        Useful if you want to continue the conversation and want to set the response to the result tool call.
        """
        if not self.result.tool_name:
            raise ValueError('Cannot set result tool return content when the return type is `str`.')
        messages = deepcopy(self.graph_run_result.state.message_history)
        last_message = messages[-1]
        for part in last_message.parts:
            if isinstance(part, _messages.ToolReturnPart) and part.tool_name == self.result.tool_name:
                part.content = return_content
                return messages
        raise LookupError(f'No tool call found with tool name {self.result.tool_name!r}.')

    def all_messages(self, *, result_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
        if result_tool_return_content is not None:
            return self._set_result_tool_return(result_tool_return_content)
        else:
            return self.graph_run_result.state.message_history

    def all_messages_json(self, *, result_tool_return_content: str | None = None) -> bytes:
        """Return all messages from [`all_messages`][pydantic_ai.result._BaseRunResult.all_messages] as JSON bytes.

        Args:
            result_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the result tool call if you want to continue
                the conversation and want to set the response to the result tool call. If `None`, the last message will
                not be modified.

        Returns:
            JSON bytes representing the messages.
        """
        return _messages.ModelMessagesTypeAdapter.dump_json(
            self.all_messages(result_tool_return_content=result_tool_return_content)
        )

    @property
    def _new_message_index(self) -> int:
        return self.graph_run_result.deps.new_message_index

    def new_messages(self, *, result_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
        return self.all_messages(result_tool_return_content=result_tool_return_content)[self._new_message_index :]

    def new_messages_json(self, *, result_tool_return_content: str | None = None) -> bytes:
        """Return new messages from [`new_messages`][pydantic_ai.result._BaseRunResult.new_messages] as JSON bytes.

        Args:
            result_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the result tool call if you want to continue
                the conversation and want to set the response to the result tool call. If `None`, the last message will
                not be modified.

        Returns:
            JSON bytes representing the new messages.
        """
        return _messages.ModelMessagesTypeAdapter.dump_json(
            self.new_messages(result_tool_return_content=result_tool_return_content)
        )

    def usage(self) -> _usage.Usage:
        return self.graph_run_result.state.usage
