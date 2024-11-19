from __future__ import annotations as _annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, cast, final, overload

import logfire_api
from typing_extensions import assert_never

from . import (
    _result,
    _retriever as _r,
    _system_prompt,
    _utils,
    exceptions,
    messages as _messages,
    models,
    result,
)
from .dependencies import AgentDeps, CallContext, RetrieverContextFunc, RetrieverParams, RetrieverPlainFunc
from .result import ResultData

__all__ = ('Agent',)

_logfire = logfire_api.Logfire(otel_scope='pydantic-ai')

NoneType = type(None)


@final
@dataclass(init=False)
class Agent(Generic[AgentDeps, ResultData]):
    """Class for defining "agents" - a way to have a specific type of "conversation" with an LLM.

    Agents are generic in the dependency type they take [`AgentDeps`][pydantic_ai.dependencies.AgentDeps]
    and the result data type they return, [`ResultData`][pydantic_ai.result.ResultData].

    By default, if neither generic parameter is customised, agents have type `Agent[None, str]`.

    Minimal usage example:

    ```py
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')
    result = agent.run_sync('What is the capital of France?')
    print(result.data)
    #> Paris
    ```
    """

    # dataclass fields mostly for my sanity — knowing what attributes are available
    model: models.Model | models.KnownModelName | None
    """The default model configured for this agent."""
    _result_schema: _result.ResultSchema[ResultData] | None = field(repr=False)
    _result_validators: list[_result.ResultValidator[AgentDeps, ResultData]] = field(repr=False)
    _allow_text_result: bool = field(repr=False)
    _system_prompts: tuple[str, ...] = field(repr=False)
    _retrievers: dict[str, _r.Retriever[AgentDeps, Any]] = field(repr=False)
    _default_retries: int = field(repr=False)
    _system_prompt_functions: list[_system_prompt.SystemPromptRunner[AgentDeps]] = field(repr=False)
    _deps_type: type[AgentDeps] = field(repr=False)
    _max_result_retries: int = field(repr=False)
    _current_result_retry: int = field(repr=False)
    _override_deps: _utils.Option[AgentDeps] = field(default=None, repr=False)
    _override_model: _utils.Option[models.Model] = field(default=None, repr=False)
    last_run_messages: list[_messages.Message] | None = None
    """The messages from the last run, useful when a run raised an exception.

    Note: these are not used by the agent, e.g. in future runs, they are just stored for developers' convenience.
    """

    def __init__(
        self,
        model: models.Model | models.KnownModelName | None = None,
        result_type: type[ResultData] = str,
        *,
        system_prompt: str | Sequence[str] = (),
        deps_type: type[AgentDeps] = NoneType,
        retries: int = 1,
        result_tool_name: str = 'final_result',
        result_tool_description: str | None = None,
        result_retries: int | None = None,
        defer_model_check: bool = False,
    ):
        """Create an agent.

        Args:
            model: The default model to use for this agent, if not provide,
                you must provide the model when calling the agent.
            result_type: The type of the result data, used to validate the result data, defaults to `str`.
            system_prompt: Static system prompts to use for this agent, you can also register system
                prompts via a function with [`system_prompt`][pydantic_ai.Agent.system_prompt].
            deps_type: The type used for dependency injection, this parameter exists solely to allow you to fully
                parameterize the agent, and therefore get the best out of static type checking.
                If you're not using deps, but want type checking to pass, you can set `deps=None` to satisfy Pyright
                or add a type hint `: Agent[None, <return type>]`.
            retries: The default number of retries to allow before raising an error.
            result_tool_name: The name of the tool to use for the final result.
            result_tool_description: The description of the final result tool.
            result_retries: The maximum number of retries to allow for result validation, defaults to `retries`.
            defer_model_check: by default, if you provide a [named][pydantic_ai.models.KnownModelName] model,
                it's evaluated to create a [`Model`][pydantic_ai.models.Model] instance immediately,
                which checks for the necessary environment variables. Set this to `false`
                to defer the evaluation until the first run. Useful if you want to
                [override the model][pydantic_ai.Agent.override_model] for testing.
        """
        if model is None or defer_model_check:
            self.model = model
        else:
            self.model = models.infer_model(model)

        self._result_schema = _result.ResultSchema[result_type].build(
            result_type, result_tool_name, result_tool_description
        )
        # if the result tool is None, or its schema allows `str`, we allow plain text results
        self._allow_text_result = self._result_schema is None or self._result_schema.allow_text_result

        self._system_prompts = (system_prompt,) if isinstance(system_prompt, str) else tuple(system_prompt)
        self._retrievers: dict[str, _r.Retriever[AgentDeps, Any]] = {}
        self._deps_type = deps_type
        self._default_retries = retries
        self._system_prompt_functions = []
        self._max_result_retries = result_retries if result_retries is not None else retries
        self._current_result_retry = 0
        self._result_validators = []

    async def run(
        self,
        user_prompt: str,
        *,
        message_history: list[_messages.Message] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDeps = None,
    ) -> result.RunResult[ResultData]:
        """Run the agent with a user prompt in async mode.

        Args:
            user_prompt: User input to start/continue the conversation.
            message_history: History of the conversation so far.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.

        Returns:
            The result of the run.
        """
        model_used, custom_model, agent_model = await self._get_agent_model(model)

        deps = self._get_deps(deps)

        new_message_index, messages = await self._prepare_messages(deps, user_prompt, message_history)
        self.last_run_messages = messages

        for retriever in self._retrievers.values():
            retriever.reset()

        cost = result.Cost()

        with _logfire.span(
            'agent run {prompt=}',
            prompt=user_prompt,
            agent=self,
            custom_model=custom_model,
            model_name=model_used.name(),
        ) as run_span:
            run_step = 0
            while True:
                run_step += 1
                with _logfire.span('model request {run_step=}', run_step=run_step) as model_req_span:
                    model_response, request_cost = await agent_model.request(messages)
                    model_req_span.set_attribute('response', model_response)
                    model_req_span.set_attribute('cost', request_cost)
                    model_req_span.message = f'model request -> {model_response.role}'

                messages.append(model_response)
                cost += request_cost

                with _logfire.span('handle model response') as handle_span:
                    either = await self._handle_model_response(model_response, deps)

                    if isinstance(either, _MarkFinalResult):
                        # we have a final result, end the conversation
                        result_data = either.data
                        run_span.set_attribute('all_messages', messages)
                        run_span.set_attribute('cost', cost)
                        handle_span.set_attribute('result', result_data)
                        handle_span.message = 'handle model response -> final result'
                        return result.RunResult(messages, new_message_index, result_data, cost)
                    else:
                        # continue the conversation
                        tool_responses = either
                        handle_span.set_attribute('tool_responses', tool_responses)
                        response_msgs = ' '.join(m.role for m in tool_responses)
                        handle_span.message = f'handle model response -> {response_msgs}'
                        messages.extend(tool_responses)

    def run_sync(
        self,
        user_prompt: str,
        *,
        message_history: list[_messages.Message] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDeps = None,
    ) -> result.RunResult[ResultData]:
        """Run the agent with a user prompt synchronously.

        This is a convenience method that wraps `self.run` with `asyncio.run()`.

        Args:
            user_prompt: User input to start/continue the conversation.
            message_history: History of the conversation so far.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.

        Returns:
            The result of the run.
        """
        return asyncio.run(self.run(user_prompt, message_history=message_history, model=model, deps=deps))

    @asynccontextmanager
    async def run_stream(
        self,
        user_prompt: str,
        *,
        message_history: list[_messages.Message] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDeps = None,
    ) -> AsyncIterator[result.StreamedRunResult[AgentDeps, ResultData]]:
        """Run the agent with a user prompt in async mode, returning a streamed response.

        Args:
            user_prompt: User input to start/continue the conversation.
            message_history: History of the conversation so far.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.

        Returns:
            The result of the run.
        """
        model_used, custom_model, agent_model = await self._get_agent_model(model)

        deps = self._get_deps(deps)

        new_message_index, messages = await self._prepare_messages(deps, user_prompt, message_history)
        self.last_run_messages = messages

        for retriever in self._retrievers.values():
            retriever.reset()

        cost = result.Cost()

        with _logfire.span(
            'agent run stream {prompt=}',
            prompt=user_prompt,
            agent=self,
            custom_model=custom_model,
            model_name=model_used.name(),
        ) as run_span:
            run_step = 0
            while True:
                run_step += 1
                with _logfire.span('model request {run_step=}', run_step=run_step) as model_req_span:
                    async with agent_model.request_stream(messages) as model_response:
                        model_req_span.set_attribute('response_type', model_response.__class__.__name__)
                        # We want to end the "model request" span here, but we can't exit the context manager
                        # in the traditional way
                        model_req_span.__exit__(None, None, None)

                        with _logfire.span('handle model response') as handle_span:
                            either = await self._handle_streamed_model_response(model_response, deps)

                            if isinstance(either, _MarkFinalResult):
                                result_stream = either.data
                                run_span.set_attribute('all_messages', messages)
                                handle_span.set_attribute('result_type', result_stream.__class__.__name__)
                                handle_span.message = 'handle model response -> final result'
                                yield result.StreamedRunResult(
                                    messages,
                                    new_message_index,
                                    cost,
                                    result_stream,
                                    self._result_schema,
                                    deps,
                                    self._result_validators,
                                )
                                return
                            else:
                                tool_responses = either
                                handle_span.set_attribute('tool_responses', tool_responses)
                                response_msgs = ' '.join(m.role for m in tool_responses)
                                handle_span.message = f'handle model response -> {response_msgs}'
                                messages.extend(tool_responses)
                                # the model_response should have been fully streamed by now, we can add it's cost
                                cost += model_response.cost()

    @contextmanager
    def override_deps(self, overriding_deps: AgentDeps) -> Iterator[None]:
        """Context manager to temporarily override agent dependencies, this is particularly useful when testing.

        Args:
            overriding_deps: The dependencies to use instead of the dependencies passed to the agent run.
        """
        override_deps_before = self._override_deps
        self._override_deps = _utils.Some(overriding_deps)
        try:
            yield
        finally:
            self._override_deps = override_deps_before

    @contextmanager
    def override_model(self, overriding_model: models.Model | models.KnownModelName) -> Iterator[None]:
        """Context manager to temporarily override the model used by the agent.

        Args:
            overriding_model: The model to use instead of the model passed to the agent run.
        """
        override_model_before = self._override_model
        self._override_model = _utils.Some(models.infer_model(overriding_model))
        try:
            yield
        finally:
            self._override_model = override_model_before

    @overload
    def system_prompt(
        self, func: Callable[[CallContext[AgentDeps]], str], /
    ) -> Callable[[CallContext[AgentDeps]], str]: ...

    @overload
    def system_prompt(
        self, func: Callable[[CallContext[AgentDeps]], Awaitable[str]], /
    ) -> Callable[[CallContext[AgentDeps]], Awaitable[str]]: ...

    @overload
    def system_prompt(self, func: Callable[[], str], /) -> Callable[[], str]: ...

    @overload
    def system_prompt(self, func: Callable[[], Awaitable[str]], /) -> Callable[[], Awaitable[str]]: ...

    def system_prompt(
        self, func: _system_prompt.SystemPromptFunc[AgentDeps], /
    ) -> _system_prompt.SystemPromptFunc[AgentDeps]:
        """Decorator to register a system prompt function.

        Optionally takes [`CallContext`][pydantic_ai.dependencies.CallContext] as it's only argument.
        Can decorate a sync or async functions.

        Overloads for every possible signature of `system_prompt` are included so the decorator doesn't obscure
        the type of the function, see `tests/typed_agent.py` for tests.

        Example:
        ```py
        from pydantic_ai import Agent, CallContext

        agent = Agent('test', deps_type=str)

        @agent.system_prompt
        def simple_system_prompt() -> str:
            return 'foobar'

        @agent.system_prompt
        async def async_system_prompt(ctx: CallContext[str]) -> str:
            return f'{ctx.deps} is the best'

        result = agent.run_sync('foobar', deps='spam')
        print(result.data)
        #> success (no retriever calls)
        ```
        """
        self._system_prompt_functions.append(_system_prompt.SystemPromptRunner(func))
        return func

    @overload
    def result_validator(
        self, func: Callable[[CallContext[AgentDeps], ResultData], ResultData], /
    ) -> Callable[[CallContext[AgentDeps], ResultData], ResultData]: ...

    @overload
    def result_validator(
        self, func: Callable[[CallContext[AgentDeps], ResultData], Awaitable[ResultData]], /
    ) -> Callable[[CallContext[AgentDeps], ResultData], Awaitable[ResultData]]: ...

    @overload
    def result_validator(self, func: Callable[[ResultData], ResultData], /) -> Callable[[ResultData], ResultData]: ...

    @overload
    def result_validator(
        self, func: Callable[[ResultData], Awaitable[ResultData]], /
    ) -> Callable[[ResultData], Awaitable[ResultData]]: ...

    def result_validator(
        self, func: _result.ResultValidatorFunc[AgentDeps, ResultData], /
    ) -> _result.ResultValidatorFunc[AgentDeps, ResultData]:
        """Decorator to register a result validator function.

        Optionally takes [`CallContext`][pydantic_ai.dependencies.CallContext] as it's first argument.
        Can decorate a sync or async functions.

        Overloads for every possible signature of `result_validator` are included so the decorator doesn't obscure
        the type of the function, see `tests/typed_agent.py` for tests.

        Example:
        ```py
        from pydantic_ai import Agent, CallContext, ModelRetry

        agent = Agent('test', deps_type=str)

        @agent.result_validator
        def result_validator_simple(data: str) -> str:
            if 'wrong' in data:
                raise ModelRetry('wrong response')
            return data

        @agent.result_validator
        async def result_validator_deps(ctx: CallContext[str], data: str) -> str:
            if ctx.deps in data:
                raise ModelRetry('wrong response')
            return data

        result = agent.run_sync('foobar', deps='spam')
        print(result.data)
        #> success (no retriever calls)
        ```
        """
        self._result_validators.append(_result.ResultValidator(func))
        return func

    @overload
    def retriever(
        self, func: RetrieverContextFunc[AgentDeps, RetrieverParams], /
    ) -> RetrieverContextFunc[AgentDeps, RetrieverParams]: ...

    @overload
    def retriever(
        self, /, *, retries: int | None = None
    ) -> Callable[
        [RetrieverContextFunc[AgentDeps, RetrieverParams]], RetrieverContextFunc[AgentDeps, RetrieverParams]
    ]: ...

    def retriever(
        self,
        func: RetrieverContextFunc[AgentDeps, RetrieverParams] | None = None,
        /,
        *,
        retries: int | None = None,
    ) -> Any:
        """Decorator to register a retriever function which takes
        [`CallContext`][pydantic_ai.dependencies.CallContext] as its first argument.

        Can decorate a sync or async functions.

        The docstring is inspected to extract both the tool description and description of each parameter,
        [learn more](../agents.md#retrievers-tools-and-schema).

        We can't add overloads for every possible signature of retriever, since the return type is a recursive union
        so the signature of functions decorated with `@agent.retriever` is obscured.

        Example:
        ```py
        from pydantic_ai import Agent, CallContext

        agent = Agent('test', deps_type=int)

        @agent.retriever
        def foobar(ctx: CallContext[int], x: int) -> int:
            return ctx.deps + x

        @agent.retriever(retries=2)
        async def spam(ctx: CallContext[str], y: float) -> float:
            return ctx.deps + y

        result = agent.run_sync('foobar', deps=1)
        print(result.data)
        #> {"foobar":1,"spam":1.0}
        ```

        Args:
            func: The retriever function to register.
            retries: The number of retries to allow for this retriever, defaults to the agent's default retries,
                which defaults to 1.
        """  # noqa: D205
        if func is None:

            def retriever_decorator(
                func_: RetrieverContextFunc[AgentDeps, RetrieverParams],
            ) -> RetrieverContextFunc[AgentDeps, RetrieverParams]:
                # noinspection PyTypeChecker
                self._register_retriever(_utils.Either(left=func_), retries)
                return func_

            return retriever_decorator
        else:
            # noinspection PyTypeChecker
            self._register_retriever(_utils.Either(left=func), retries)
            return func

    @overload
    def retriever_plain(self, func: RetrieverPlainFunc[RetrieverParams], /) -> RetrieverPlainFunc[RetrieverParams]: ...

    @overload
    def retriever_plain(
        self, /, *, retries: int | None = None
    ) -> Callable[[RetrieverPlainFunc[RetrieverParams]], RetrieverPlainFunc[RetrieverParams]]: ...

    def retriever_plain(
        self, func: RetrieverPlainFunc[RetrieverParams] | None = None, /, *, retries: int | None = None
    ) -> Any:
        """Decorator to register a retriever function which DOES NOT take `CallContext` as an argument.

        Can decorate a sync or async functions.

        The docstring is inspected to extract both the tool description and description of each parameter,
        [learn more](../agents.md#retrievers-tools-and-schema).

        We can't add overloads for every possible signature of retriever, since the return type is a recursive union
        so the signature of functions decorated with `@agent.retriever` is obscured.

        Example:
        ```py
        from pydantic_ai import Agent, CallContext

        agent = Agent('test')

        @agent.retriever
        def foobar(ctx: CallContext[int]) -> int:
            return 123

        @agent.retriever(retries=2)
        async def spam(ctx: CallContext[str]) -> float:
            return 3.14

        result = agent.run_sync('foobar', deps=1)
        print(result.data)
        #> {"foobar":123,"spam":3.14}
        ```

        Args:
            func: The retriever function to register.
            retries: The number of retries to allow for this retriever, defaults to the agent's default retries,
                which defaults to 1.
        """
        if func is None:

            def retriever_decorator(
                func_: RetrieverPlainFunc[RetrieverParams],
            ) -> RetrieverPlainFunc[RetrieverParams]:
                # noinspection PyTypeChecker
                self._register_retriever(_utils.Either(right=func_), retries)
                return func_

            return retriever_decorator
        else:
            self._register_retriever(_utils.Either(right=func), retries)
            return func

    def _register_retriever(
        self, func: _r.RetrieverEitherFunc[AgentDeps, RetrieverParams], retries: int | None
    ) -> None:
        """Private utility to register a retriever function."""
        retries_ = retries if retries is not None else self._default_retries
        retriever = _r.Retriever[AgentDeps, RetrieverParams](func, retries_)

        if self._result_schema and retriever.name in self._result_schema.tools:
            raise ValueError(f'Retriever name conflicts with result schema name: {retriever.name!r}')

        if retriever.name in self._retrievers:
            raise ValueError(f'Retriever name conflicts with existing retriever: {retriever.name!r}')

        self._retrievers[retriever.name] = retriever

    async def _get_agent_model(
        self, model: models.Model | models.KnownModelName | None
    ) -> tuple[models.Model, models.Model | None, models.AgentModel]:
        """Create a model configured for this agent.

        Args:
            model: model to use for this run, required if `model` was not set when creating the agent.

        Returns:
            a tuple of `(model used, custom_model if any, agent_model)`
        """
        model_: models.Model
        if some_model := self._override_model:
            # we don't want `override_model()` to cover up errors from the model not being defined, hence this check
            if model is None and self.model is None:
                raise exceptions.UserError(
                    '`model` must be set either when creating the agent or when calling it. '
                    '(Even when `override_model()` is customizing the model that will actually be called)'
                )
            model_ = some_model.value
            custom_model = None
        elif model is not None:
            custom_model = model_ = models.infer_model(model)
        elif self.model is not None:
            # noinspection PyTypeChecker
            model_ = self.model = models.infer_model(self.model)
            custom_model = None
        else:
            raise exceptions.UserError('`model` must be set either when creating the agent or when calling it.')

        result_tools = list(self._result_schema.tools.values()) if self._result_schema else None
        return model_, custom_model, model_.agent_model(self._retrievers, self._allow_text_result, result_tools)

    async def _prepare_messages(
        self, deps: AgentDeps, user_prompt: str, message_history: list[_messages.Message] | None
    ) -> tuple[int, list[_messages.Message]]:
        # if message history includes system prompts, we don't want to regenerate them
        if message_history and any(m.role == 'system' for m in message_history):
            # shallow copy messages
            messages = message_history.copy()
        else:
            messages = await self._init_messages(deps)
            if message_history:
                messages += message_history

        new_message_index = len(messages)
        messages.append(_messages.UserPrompt(user_prompt))
        return new_message_index, messages

    async def _handle_model_response(
        self, model_response: _messages.ModelAnyResponse, deps: AgentDeps
    ) -> _MarkFinalResult[ResultData] | list[_messages.Message]:
        """Process a non-streamed response from the model.

        Returns:
            Return `Either` — left: final result data, right: list of messages to send back to the model.
        """
        if model_response.role == 'model-text-response':
            # plain string response
            if self._allow_text_result:
                result_data_input = cast(ResultData, model_response.content)
                try:
                    result_data = await self._validate_result(result_data_input, deps, None)
                except _result.ToolRetryError as e:
                    self._incr_result_retry()
                    return [e.tool_retry]
                else:
                    return _MarkFinalResult(result_data)
            else:
                self._incr_result_retry()
                response = _messages.RetryPrompt(
                    content='Plain text responses are not permitted, please call one of the functions instead.',
                )
                return [response]
        elif model_response.role == 'model-structured-response':
            if self._result_schema is not None:
                # if there's a result schema, and any of the calls match one of its tools, return the result
                # NOTE: this means we ignore any other tools called here
                if match := self._result_schema.find_tool(model_response):
                    call, result_tool = match
                    try:
                        result_data = result_tool.validate(call)
                        result_data = await self._validate_result(result_data, deps, call)
                    except _result.ToolRetryError as e:
                        self._incr_result_retry()
                        return [e.tool_retry]
                    else:
                        return _MarkFinalResult(result_data)

            if not model_response.calls:
                raise exceptions.UnexpectedModelBehavior('Received empty tool call message')

            # otherwise we run all retriever functions in parallel
            messages: list[_messages.Message] = []
            tasks: list[asyncio.Task[_messages.Message]] = []
            for call in model_response.calls:
                if retriever := self._retrievers.get(call.tool_name):
                    tasks.append(asyncio.create_task(retriever.run(deps, call), name=call.tool_name))
                else:
                    messages.append(self._unknown_tool(call.tool_name))

            with _logfire.span('running {tools=}', tools=[t.get_name() for t in tasks]):
                messages += await asyncio.gather(*tasks)
            return messages
        else:
            assert_never(model_response)

    async def _handle_streamed_model_response(
        self, model_response: models.EitherStreamedResponse, deps: AgentDeps
    ) -> _MarkFinalResult[models.EitherStreamedResponse] | list[_messages.Message]:
        """Process a streamed response from the model.

        TODO: change the response type to `models.EitherStreamedResponse | list[_messages.Message]` once we drop 3.9
        (with 3.9 we get `TypeError: Subscripted generics cannot be used with class and instance checks`)

        Returns:
            Return `Either` — left: final result data, right: list of messages to send back to the model.
        """
        if isinstance(model_response, models.StreamTextResponse):
            # plain string response
            if self._allow_text_result:
                return _MarkFinalResult(model_response)
            else:
                self._incr_result_retry()
                response = _messages.RetryPrompt(
                    content='Plain text responses are not permitted, please call one of the functions instead.',
                )
                # stream the response, so cost is correct
                async for _ in model_response:
                    pass

                return [response]
        else:
            assert isinstance(model_response, models.StreamStructuredResponse), f'Unexpected response: {model_response}'
            if self._result_schema is not None:
                # if there's a result schema, iterate over the stream until we find at least one tool
                # NOTE: this means we ignore any other tools called here
                structured_msg = model_response.get()
                while not structured_msg.calls:
                    try:
                        await model_response.__anext__()
                    except StopAsyncIteration:
                        break
                    structured_msg = model_response.get()

                if self._result_schema.find_tool(structured_msg):
                    return _MarkFinalResult(model_response)

            # the model is calling a retriever function, consume the response to get the next message
            async for _ in model_response:
                pass
            structured_msg = model_response.get()
            if not structured_msg.calls:
                raise exceptions.UnexpectedModelBehavior('Received empty tool call message')
            messages: list[_messages.Message] = [structured_msg]

            # we now run all retriever functions in parallel
            tasks: list[asyncio.Task[_messages.Message]] = []
            for call in structured_msg.calls:
                if retriever := self._retrievers.get(call.tool_name):
                    tasks.append(asyncio.create_task(retriever.run(deps, call), name=call.tool_name))
                else:
                    messages.append(self._unknown_tool(call.tool_name))

            with _logfire.span('running {tools=}', tools=[t.get_name() for t in tasks]):
                messages += await asyncio.gather(*tasks)
            return messages

    async def _validate_result(
        self, result_data: ResultData, deps: AgentDeps, tool_call: _messages.ToolCall | None
    ) -> ResultData:
        for validator in self._result_validators:
            result_data = await validator.validate(result_data, deps, self._current_result_retry, tool_call)
        return result_data

    def _incr_result_retry(self) -> None:
        self._current_result_retry += 1
        if self._current_result_retry > self._max_result_retries:
            raise exceptions.UnexpectedModelBehavior(
                f'Exceeded maximum retries ({self._max_result_retries}) for result validation'
            )

    async def _init_messages(self, deps: AgentDeps) -> list[_messages.Message]:
        """Build the initial messages for the conversation."""
        messages: list[_messages.Message] = [_messages.SystemPrompt(p) for p in self._system_prompts]
        for sys_prompt_runner in self._system_prompt_functions:
            prompt = await sys_prompt_runner.run(deps)
            messages.append(_messages.SystemPrompt(prompt))
        return messages

    def _unknown_tool(self, tool_name: str) -> _messages.RetryPrompt:
        self._incr_result_retry()
        names = list(self._retrievers.keys())
        if self._result_schema:
            names.extend(self._result_schema.tool_names())
        if names:
            msg = f'Available tools: {", ".join(names)}'
        else:
            msg = 'No tools available.'
        return _messages.RetryPrompt(content=f'Unknown tool name: {tool_name!r}. {msg}')

    def _get_deps(self, deps: AgentDeps) -> AgentDeps:
        """Get deps for a run.

        If we've overridden deps via `_override_deps_stack`, use that, otherwise use the deps passed to the call.

        We could do runtime type checking of deps against `self._deps_type`, but that's a slippery slope.
        """
        if some_deps := self._override_deps:
            return some_deps.value
        else:
            return deps


@dataclass
class _MarkFinalResult(Generic[ResultData]):
    """Marker class to indicate that the result is the final result.

    This allows us to use `isinstance`, which wouldn't be possible if we were returning `ResultData` directly.
    """

    data: ResultData
