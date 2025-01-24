from __future__ import annotations as _annotations

import asyncio
import dataclasses
import inspect
from collections.abc import AsyncIterator, Awaitable, Iterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from contextvars import ContextVar
from types import FrameType
from typing import Any, Callable, Generic, Literal, cast, final, overload

import logfire_api
from typing_extensions import TypeVar, assert_never, deprecated

from . import (
    _result,
    _system_prompt,
    _utils,
    exceptions,
    messages as _messages,
    models,
    result,
    usage as _usage,
)
from .result import ResultDataT
from .settings import ModelSettings, merge_model_settings
from .tools import (
    AgentDepsT,
    DocstringFormat,
    RunContext,
    Tool,
    ToolDefinition,
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
"""An invariant TypeVar."""
NoneType = type(None)
EndStrategy = Literal['early', 'exhaustive']
"""The strategy for handling multiple tool calls when a final result is found.

- `'early'`: Stop processing other tool calls once a final result is found
- `'exhaustive'`: Process all tool calls even after finding a final result
"""
RunResultData = TypeVar('RunResultData')
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
    _result_tool_name: str = dataclasses.field(repr=False)
    _result_tool_description: str | None = dataclasses.field(repr=False)
    _result_schema: _result.ResultSchema[ResultDataT] | None = dataclasses.field(repr=False)
    _result_validators: list[_result.ResultValidator[AgentDepsT, ResultDataT]] = dataclasses.field(repr=False)
    _system_prompts: tuple[str, ...] = dataclasses.field(repr=False)
    _function_tools: dict[str, Tool[AgentDepsT]] = dataclasses.field(repr=False)
    _default_retries: int = dataclasses.field(repr=False)
    _system_prompt_functions: list[_system_prompt.SystemPromptRunner[AgentDepsT]] = dataclasses.field(repr=False)
    _system_prompt_dynamic_functions: dict[str, _system_prompt.SystemPromptRunner[AgentDepsT]] = dataclasses.field(
        repr=False
    )
    _deps_type: type[AgentDepsT] = dataclasses.field(repr=False)
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
        self._result_tool_name = result_tool_name
        self._result_tool_description = result_tool_description
        self._result_schema = _result.ResultSchema[result_type].build(
            result_type, result_tool_name, result_tool_description
        )

        self._system_prompts = (system_prompt,) if isinstance(system_prompt, str) else tuple(system_prompt)
        self._function_tools = {}
        self._default_retries = retries
        for tool in tools:
            if isinstance(tool, Tool):
                self._register_tool(tool)
            else:
                self._register_tool(Tool(tool))
        self._deps_type = deps_type
        self._system_prompt_functions = []
        self._system_prompt_dynamic_functions = {}
        self._max_result_retries = result_retries if result_retries is not None else retries
        self._result_validators = []

    @overload
    async def run(
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
    ) -> result.RunResult[ResultDataT]: ...

    @overload
    async def run(
        self,
        user_prompt: str,
        *,
        result_type: type[RunResultData],
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> result.RunResult[RunResultData]: ...

    async def run(
        self,
        user_prompt: str,
        *,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        result_type: type[RunResultData] | None = None,
        infer_name: bool = True,
    ) -> result.RunResult[Any]:
        """Run the agent with a user prompt in async mode.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            result = await agent.run('What is the capital of France?')
            print(result.data)
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
        model_used = await self._get_model(model)

        deps = self._get_deps(deps)
        new_message_index = len(message_history) if message_history else 0
        result_schema = self._prepare_result_schema(result_type)

        with _logfire.span(
            '{agent_name} run {prompt=}',
            prompt=user_prompt,
            agent=self,
            model_name=model_used.name(),
            agent_name=self.name or 'agent',
        ) as run_span:
            run_context = RunContext(deps, model_used, usage or _usage.Usage(), user_prompt)
            messages = await self._prepare_messages(user_prompt, message_history, run_context)
            run_context.messages = messages

            for tool in self._function_tools.values():
                tool.current_retry = 0

            model_settings = merge_model_settings(self.model_settings, model_settings)
            usage_limits = usage_limits or _usage.UsageLimits()

            while True:
                usage_limits.check_before_request(run_context.usage)

                run_context.run_step += 1
                with _logfire.span('preparing model and tools {run_step=}', run_step=run_context.run_step):
                    agent_model = await self._prepare_model(run_context, result_schema)

                with _logfire.span('model request', run_step=run_context.run_step) as model_req_span:
                    model_response, request_usage = await agent_model.request(messages, model_settings)
                    model_req_span.set_attribute('response', model_response)
                    model_req_span.set_attribute('usage', request_usage)

                messages.append(model_response)
                run_context.usage.incr(request_usage, requests=1)
                usage_limits.check_tokens(run_context.usage)

                with _logfire.span('handle model response', run_step=run_context.run_step) as handle_span:
                    final_result, tool_responses = await self._handle_model_response(
                        model_response, run_context, result_schema
                    )

                    if tool_responses:
                        # Add parts to the conversation as a new message
                        messages.append(_messages.ModelRequest(tool_responses))

                    # Check if we got a final result
                    if final_result is not None:
                        result_data = final_result.data
                        result_tool_name = final_result.tool_name
                        run_span.set_attribute('all_messages', messages)
                        run_span.set_attribute('usage', run_context.usage)
                        handle_span.set_attribute('result', result_data)
                        handle_span.message = 'handle model response -> final result'
                        return result.RunResult(
                            messages, new_message_index, result_data, result_tool_name, run_context.usage
                        )
                    else:
                        # continue the conversation
                        handle_span.set_attribute('tool_responses', tool_responses)
                        tool_responses_str = ' '.join(r.part_kind for r in tool_responses)
                        handle_span.message = f'handle model response -> {tool_responses_str}'

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
    ) -> result.RunResult[ResultDataT]: ...

    @overload
    def run_sync(
        self,
        user_prompt: str,
        *,
        result_type: type[RunResultData] | None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> result.RunResult[RunResultData]: ...

    def run_sync(
        self,
        user_prompt: str,
        *,
        result_type: type[RunResultData] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> result.RunResult[Any]:
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
        result_type: type[RunResultData],
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> AbstractAsyncContextManager[result.StreamedRunResult[AgentDepsT, RunResultData]]: ...

    @asynccontextmanager
    async def run_stream(
        self,
        user_prompt: str,
        *,
        result_type: type[RunResultData] | None = None,
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
        model_used = await self._get_model(model)

        deps = self._get_deps(deps)
        new_message_index = len(message_history) if message_history else 0
        result_schema = self._prepare_result_schema(result_type)

        with _logfire.span(
            '{agent_name} run stream {prompt=}',
            prompt=user_prompt,
            agent=self,
            model_name=model_used.name(),
            agent_name=self.name or 'agent',
        ) as run_span:
            run_context = RunContext(deps, model_used, usage or _usage.Usage(), user_prompt)
            messages = await self._prepare_messages(user_prompt, message_history, run_context)
            run_context.messages = messages

            for tool in self._function_tools.values():
                tool.current_retry = 0

            model_settings = merge_model_settings(self.model_settings, model_settings)
            usage_limits = usage_limits or _usage.UsageLimits()

            while True:
                run_context.run_step += 1
                usage_limits.check_before_request(run_context.usage)

                with _logfire.span('preparing model and tools {run_step=}', run_step=run_context.run_step):
                    agent_model = await self._prepare_model(run_context, result_schema)

                with _logfire.span('model request {run_step=}', run_step=run_context.run_step) as model_req_span:
                    async with agent_model.request_stream(messages, model_settings) as model_response:
                        run_context.usage.requests += 1
                        model_req_span.set_attribute('response_type', model_response.__class__.__name__)
                        # We want to end the "model request" span here, but we can't exit the context manager
                        # in the traditional way
                        model_req_span.__exit__(None, None, None)

                        with _logfire.span('handle model response') as handle_span:
                            maybe_final_result = await self._handle_streamed_response(
                                model_response, run_context, result_schema
                            )

                            # Check if we got a final result
                            if isinstance(maybe_final_result, _MarkFinalResult):
                                result_stream = maybe_final_result.data
                                result_tool_name = maybe_final_result.tool_name
                                handle_span.message = 'handle model response -> final result'

                                async def on_complete():
                                    """Called when the stream has completed.

                                    The model response will have been added to messages by now
                                    by `StreamedRunResult._marked_completed`.
                                    """
                                    last_message = messages[-1]
                                    assert isinstance(last_message, _messages.ModelResponse)
                                    tool_calls = [
                                        part for part in last_message.parts if isinstance(part, _messages.ToolCallPart)
                                    ]
                                    parts = await self._process_function_tools(
                                        tool_calls, result_tool_name, run_context, result_schema
                                    )
                                    if any(isinstance(part, _messages.RetryPromptPart) for part in parts):
                                        self._incr_result_retry(run_context)
                                    if parts:
                                        messages.append(_messages.ModelRequest(parts))
                                    run_span.set_attribute('all_messages', messages)

                                # The following is not guaranteed to be true, but we consider it a user error if
                                # there are result validators that might convert the result data from an overridden
                                # `result_type` to a type that is not valid as such.
                                result_validators = cast(
                                    list[_result.ResultValidator[AgentDepsT, RunResultData]], self._result_validators
                                )

                                yield result.StreamedRunResult(
                                    messages,
                                    new_message_index,
                                    usage_limits,
                                    result_stream,
                                    result_schema,
                                    run_context,
                                    result_validators,
                                    result_tool_name,
                                    on_complete,
                                )
                                return
                            else:
                                # continue the conversation
                                model_response_msg, tool_responses = maybe_final_result
                                # if we got a model response add that to messages
                                messages.append(model_response_msg)
                                if tool_responses:
                                    # if we got one or more tool response parts, add a model request message
                                    messages.append(_messages.ModelRequest(tool_responses))

                                handle_span.set_attribute('tool_responses', tool_responses)
                                tool_responses_str = ' '.join(r.part_kind for r in tool_responses)
                                handle_span.message = f'handle model response -> {tool_responses_str}'
                                # the model_response should have been fully streamed by now, we can add its usage
                                model_response_usage = model_response.usage()
                                run_context.usage.incr(model_response_usage)
                                usage_limits.check_tokens(run_context.usage)

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
            self._system_prompt_functions.append(_system_prompt.SystemPromptRunner(func, dynamic=dynamic))
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

    async def _get_model(self, model: models.Model | models.KnownModelName | None) -> models.Model:
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

    async def _prepare_model(
        self, run_context: RunContext[AgentDepsT], result_schema: _result.ResultSchema[RunResultData] | None
    ) -> models.AgentModel:
        """Build tools and create an agent model."""
        function_tools: list[ToolDefinition] = []

        async def add_tool(tool: Tool[AgentDepsT]) -> None:
            ctx = run_context.replace_with(retry=tool.current_retry, tool_name=tool.name)
            if tool_def := await tool.prepare_tool_def(ctx):
                function_tools.append(tool_def)

        await asyncio.gather(*map(add_tool, self._function_tools.values()))

        return await run_context.model.agent_model(
            function_tools=function_tools,
            allow_text_result=self._allow_text_result(result_schema),
            result_tools=result_schema.tool_defs() if result_schema is not None else [],
        )

    async def _reevaluate_dynamic_prompts(
        self, messages: list[_messages.ModelMessage], run_context: RunContext[AgentDepsT]
    ) -> None:
        """Reevaluate any `SystemPromptPart` with dynamic_ref in the provided messages by running the associated runner function."""
        # Only proceed if there's at least one dynamic runner.
        if self._system_prompt_dynamic_functions:
            for msg in messages:
                if isinstance(msg, _messages.ModelRequest):
                    for i, part in enumerate(msg.parts):
                        if isinstance(part, _messages.SystemPromptPart) and part.dynamic_ref:
                            # Look up the runner by its ref
                            if runner := self._system_prompt_dynamic_functions.get(part.dynamic_ref):
                                updated_part_content = await runner.run(run_context)
                                msg.parts[i] = _messages.SystemPromptPart(
                                    updated_part_content, dynamic_ref=part.dynamic_ref
                                )

    def _prepare_result_schema(
        self, result_type: type[RunResultData] | None
    ) -> _result.ResultSchema[RunResultData] | None:
        if result_type is not None:
            if self._result_validators:
                raise exceptions.UserError('Cannot set a custom run `result_type` when the agent has result validators')
            return _result.ResultSchema[result_type].build(
                result_type, self._result_tool_name, self._result_tool_description
            )
        else:
            return self._result_schema  # pyright: ignore[reportReturnType]

    async def _prepare_messages(
        self,
        user_prompt: str,
        message_history: list[_messages.ModelMessage] | None,
        run_context: RunContext[AgentDepsT],
    ) -> list[_messages.ModelMessage]:
        try:
            ctx_messages = _messages_ctx_var.get()
        except LookupError:
            messages: list[_messages.ModelMessage] = []
        else:
            if ctx_messages.used:
                messages = []
            else:
                messages = ctx_messages.messages
                ctx_messages.used = True

        if message_history:
            # Shallow copy messages
            messages.extend(message_history)
            # Reevaluate any dynamic system prompt parts
            await self._reevaluate_dynamic_prompts(messages, run_context)
            messages.append(_messages.ModelRequest([_messages.UserPromptPart(user_prompt)]))
        else:
            parts = await self._sys_parts(run_context)
            parts.append(_messages.UserPromptPart(user_prompt))
            messages.append(_messages.ModelRequest(parts))

        return messages

    async def _handle_model_response(
        self,
        model_response: _messages.ModelResponse,
        run_context: RunContext[AgentDepsT],
        result_schema: _result.ResultSchema[RunResultData] | None,
    ) -> tuple[_MarkFinalResult[RunResultData] | None, list[_messages.ModelRequestPart]]:
        """Process a non-streamed response from the model.

        Returns:
            A tuple of `(final_result, request parts)`. If `final_result` is not `None`, the conversation should end.
        """
        texts: list[str] = []
        tool_calls: list[_messages.ToolCallPart] = []
        for part in model_response.parts:
            if isinstance(part, _messages.TextPart):
                # ignore empty content for text parts, see #437
                if part.content:
                    texts.append(part.content)
            else:
                tool_calls.append(part)

        # At the moment, we prioritize at least executing tool calls if they are present.
        # In the future, we'd consider making this configurable at the agent or run level.
        # This accounts for cases like anthropic returns that might contain a text response
        # and a tool call response, where the text response just indicates the tool call will happen.
        if tool_calls:
            return await self._handle_structured_response(tool_calls, run_context, result_schema)
        elif texts:
            text = '\n\n'.join(texts)
            return await self._handle_text_response(text, run_context, result_schema)
        else:
            raise exceptions.UnexpectedModelBehavior('Received empty model response')

    async def _handle_text_response(
        self, text: str, run_context: RunContext[AgentDepsT], result_schema: _result.ResultSchema[RunResultData] | None
    ) -> tuple[_MarkFinalResult[RunResultData] | None, list[_messages.ModelRequestPart]]:
        """Handle a plain text response from the model for non-streaming responses."""
        if self._allow_text_result(result_schema):
            result_data_input = cast(RunResultData, text)
            try:
                result_data = await self._validate_result(result_data_input, run_context, None)
            except _result.ToolRetryError as e:
                self._incr_result_retry(run_context)
                return None, [e.tool_retry]
            else:
                return _MarkFinalResult(result_data, None), []
        else:
            self._incr_result_retry(run_context)
            response = _messages.RetryPromptPart(
                content='Plain text responses are not permitted, please call one of the functions instead.',
            )
            return None, [response]

    async def _handle_structured_response(
        self,
        tool_calls: list[_messages.ToolCallPart],
        run_context: RunContext[AgentDepsT],
        result_schema: _result.ResultSchema[RunResultData] | None,
    ) -> tuple[_MarkFinalResult[RunResultData] | None, list[_messages.ModelRequestPart]]:
        """Handle a structured response containing tool calls from the model for non-streaming responses."""
        assert tool_calls, 'Expected at least one tool call'

        # first look for the result tool call
        final_result: _MarkFinalResult[RunResultData] | None = None

        parts: list[_messages.ModelRequestPart] = []
        if result_schema is not None:
            if match := result_schema.find_tool(tool_calls):
                call, result_tool = match
                try:
                    result_data = result_tool.validate(call)
                    result_data = await self._validate_result(result_data, run_context, call)
                except _result.ToolRetryError as e:
                    parts.append(e.tool_retry)
                else:
                    final_result = _MarkFinalResult(result_data, call.tool_name)

        # Then build the other request parts based on end strategy
        parts += await self._process_function_tools(
            tool_calls, final_result and final_result.tool_name, run_context, result_schema
        )

        if any(isinstance(part, _messages.RetryPromptPart) for part in parts):
            self._incr_result_retry(run_context)

        return final_result, parts

    async def _process_function_tools(
        self,
        tool_calls: list[_messages.ToolCallPart],
        result_tool_name: str | None,
        run_context: RunContext[AgentDepsT],
        result_schema: _result.ResultSchema[RunResultData] | None,
    ) -> list[_messages.ModelRequestPart]:
        """Process function (non-result) tool calls in parallel.

        Also add stub return parts for any other tools that need it.
        """
        parts: list[_messages.ModelRequestPart] = []
        tasks: list[asyncio.Task[_messages.ModelRequestPart]] = []

        stub_function_tools = bool(result_tool_name) and self.end_strategy == 'early'

        # we rely on the fact that if we found a result, it's the first result tool in the last
        found_used_result_tool = False
        for call in tool_calls:
            if call.tool_name == result_tool_name and not found_used_result_tool:
                found_used_result_tool = True
                parts.append(
                    _messages.ToolReturnPart(
                        tool_name=call.tool_name,
                        content='Final result processed.',
                        tool_call_id=call.tool_call_id,
                    )
                )
            elif tool := self._function_tools.get(call.tool_name):
                if stub_function_tools:
                    parts.append(
                        _messages.ToolReturnPart(
                            tool_name=call.tool_name,
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=call.tool_call_id,
                        )
                    )
                else:
                    tasks.append(asyncio.create_task(tool.run(call, run_context), name=call.tool_name))
            elif result_schema is not None and call.tool_name in result_schema.tools:
                # if tool_name is in _result_schema, it means we found a result tool but an error occurred in
                # validation, we don't add another part here
                if result_tool_name is not None:
                    parts.append(
                        _messages.ToolReturnPart(
                            tool_name=call.tool_name,
                            content='Result tool not used - a final result was already processed.',
                            tool_call_id=call.tool_call_id,
                        )
                    )
            else:
                parts.append(self._unknown_tool(call.tool_name, result_schema))

        # Run all tool tasks in parallel
        if tasks:
            with _logfire.span('running {tools=}', tools=[t.get_name() for t in tasks]):
                task_results: Sequence[_messages.ModelRequestPart] = await asyncio.gather(*tasks)
                parts.extend(task_results)
        return parts

    async def _handle_streamed_response(
        self,
        streamed_response: models.StreamedResponse,
        run_context: RunContext[AgentDepsT],
        result_schema: _result.ResultSchema[RunResultData] | None,
    ) -> _MarkFinalResult[models.StreamedResponse] | tuple[_messages.ModelResponse, list[_messages.ModelRequestPart]]:
        """Process a streamed response from the model.

        Returns:
            Either a final result or a tuple of the model response and the tool responses for the next request.
            If a final result is returned, the conversation should end.
        """
        received_text = False

        async for maybe_part_event in streamed_response:
            if isinstance(maybe_part_event, _messages.PartStartEvent):
                new_part = maybe_part_event.part
                if isinstance(new_part, _messages.TextPart):
                    received_text = True
                    if self._allow_text_result(result_schema):
                        return _MarkFinalResult(streamed_response, None)
                elif isinstance(new_part, _messages.ToolCallPart):
                    if result_schema is not None and (match := result_schema.find_tool([new_part])):
                        call, _ = match
                        return _MarkFinalResult(streamed_response, call.tool_name)
                else:
                    assert_never(new_part)

        tasks: list[asyncio.Task[_messages.ModelRequestPart]] = []
        parts: list[_messages.ModelRequestPart] = []
        model_response = streamed_response.get()
        if not model_response.parts:
            raise exceptions.UnexpectedModelBehavior('Received empty model response')
        for p in model_response.parts:
            if isinstance(p, _messages.ToolCallPart):
                if tool := self._function_tools.get(p.tool_name):
                    tasks.append(asyncio.create_task(tool.run(p, run_context), name=p.tool_name))
                else:
                    parts.append(self._unknown_tool(p.tool_name, result_schema))

        if received_text and not tasks and not parts:
            # Can only get here if self._allow_text_result returns `False` for the provided result_schema
            self._incr_result_retry(run_context)
            model_response = _messages.RetryPromptPart(
                content='Plain text responses are not permitted, please call one of the functions instead.',
            )
            return streamed_response.get(), [model_response]

        with _logfire.span('running {tools=}', tools=[t.get_name() for t in tasks]):
            task_results: Sequence[_messages.ModelRequestPart] = await asyncio.gather(*tasks)
            parts.extend(task_results)

        if any(isinstance(part, _messages.RetryPromptPart) for part in parts):
            self._incr_result_retry(run_context)

        return model_response, parts

    async def _validate_result(
        self,
        result_data: RunResultData,
        run_context: RunContext[AgentDepsT],
        tool_call: _messages.ToolCallPart | None,
    ) -> RunResultData:
        if self._result_validators:
            agent_result_data = cast(ResultDataT, result_data)
            for validator in self._result_validators:
                agent_result_data = await validator.validate(agent_result_data, tool_call, run_context)
            return cast(RunResultData, agent_result_data)
        else:
            return result_data

    def _incr_result_retry(self, run_context: RunContext[AgentDepsT]) -> None:
        run_context.retry += 1
        if run_context.retry > self._max_result_retries:
            raise exceptions.UnexpectedModelBehavior(
                f'Exceeded maximum retries ({self._max_result_retries}) for result validation'
            )

    async def _sys_parts(self, run_context: RunContext[AgentDepsT]) -> list[_messages.ModelRequestPart]:
        """Build the initial messages for the conversation."""
        messages: list[_messages.ModelRequestPart] = [_messages.SystemPromptPart(p) for p in self._system_prompts]
        for sys_prompt_runner in self._system_prompt_functions:
            prompt = await sys_prompt_runner.run(run_context)
            if sys_prompt_runner.dynamic:
                messages.append(_messages.SystemPromptPart(prompt, dynamic_ref=sys_prompt_runner.function.__qualname__))
            else:
                messages.append(_messages.SystemPromptPart(prompt))
        return messages

    def _unknown_tool(
        self,
        tool_name: str,
        result_schema: _result.ResultSchema[RunResultData] | None,
    ) -> _messages.RetryPromptPart:
        names = list(self._function_tools.keys())
        if result_schema:
            names.extend(result_schema.tool_names())
        if names:
            msg = f'Available tools: {", ".join(names)}'
        else:
            msg = 'No tools available.'
        return _messages.RetryPromptPart(content=f'Unknown tool name: {tool_name!r}. {msg}')

    def _get_deps(self: Agent[T, Any], deps: T) -> T:
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

    @staticmethod
    def _allow_text_result(result_schema: _result.ResultSchema[RunResultData] | None) -> bool:
        return result_schema is None or result_schema.allow_text_result

    @property
    @deprecated(
        'The `last_run_messages` attribute has been removed, use `capture_run_messages` instead.', category=None
    )
    def last_run_messages(self) -> list[_messages.ModelMessage]:
        raise AttributeError('The `last_run_messages` attribute has been removed, use `capture_run_messages` instead.')


@dataclasses.dataclass
class _RunMessages:
    messages: list[_messages.ModelMessage]
    used: bool = False


_messages_ctx_var: ContextVar[_RunMessages] = ContextVar('var')


@contextmanager
def capture_run_messages() -> Iterator[list[_messages.ModelMessage]]:
    """Context manager to access the messages used in a [`run`][pydantic_ai.Agent.run], [`run_sync`][pydantic_ai.Agent.run_sync], or [`run_stream`][pydantic_ai.Agent.run_stream] call.

    Useful when a run may raise an exception, see [model errors](../agents.md#model-errors) for more information.

    Examples:
    ```python
    from pydantic_ai import Agent, capture_run_messages

    agent = Agent('test')

    with capture_run_messages() as messages:
        try:
            result = agent.run_sync('foobar')
        except Exception:
            print(messages)
            raise
    ```

    !!! note
        If you call `run`, `run_sync`, or `run_stream` more than once within a single `capture_run_messages` context,
        `messages` will represent the messages exchanged during the first call only.
    """
    try:
        yield _messages_ctx_var.get().messages
    except LookupError:
        messages: list[_messages.ModelMessage] = []
        token = _messages_ctx_var.set(_RunMessages(messages))
        try:
            yield messages
        finally:
            _messages_ctx_var.reset(token)


@dataclasses.dataclass
class _MarkFinalResult(Generic[ResultDataT]):
    """Marker class to indicate that the result is the final result.

    This allows us to use `isinstance`, which wouldn't be possible if we were returning `ResultDataT` directly.

    It also avoids problems in the case where the result type is itself `None`, but is set.
    """

    data: ResultDataT
    """The final result data."""
    tool_name: str | None
    """Name of the final result tool, None if the result is a string."""
