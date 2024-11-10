from __future__ import annotations as _annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable, Generic, Literal, cast, final, overload

import logfire_api
from pydantic import ValidationError
from typing_extensions import assert_never

from . import _result, _retriever as _r, _system_prompt, _utils, exceptions, messages as _messages, models, result
from .call_typing import AgentDeps
from .result import ResponseData

__all__ = 'Agent', 'KnownModelName'
KnownModelName = Literal[
    'openai:gpt-4o',
    'openai:gpt-4o-mini',
    'openai:gpt-4-turbo',
    'openai:gpt-4',
    'openai:gpt-3.5-turbo',
    'gemini-1.5-flash',
    'gemini-1.5-pro',
]
_logfire = logfire_api.Logfire(otel_scope='pydantic-ai')


@final
@dataclass(init=False)
class Agent(Generic[AgentDeps, ResponseData]):
    """Main class for creating "agents" - a way to have a specific type of "conversation" with an LLM."""

    # dataclass fields mostly for my sanity — knowing what attributes are available
    model: models.Model | None
    _result_schema: _result.ResultSchema[ResponseData] | None
    _result_validators: list[_result.ResultValidator[AgentDeps, ResponseData]]
    _allow_text_result: bool
    _system_prompts: tuple[str, ...]
    _retrievers: dict[str, _r.Retriever[AgentDeps, Any]]
    _default_retries: int
    _system_prompt_functions: list[_system_prompt.SystemPromptRunner[AgentDeps]]
    _default_deps: AgentDeps
    _max_result_retries: int
    _current_result_retry: int
    last_run_messages: list[_messages.Message] | None = None
    """The messages from the last run, useful when a run raised an exception.

    Note: these are not used by the agent, e.g. in future runs, they are just stored for developers' convenience.
    """

    def __init__(
        self,
        model: models.Model | KnownModelName | None = None,
        result_type: type[ResponseData] = str,
        *,
        system_prompt: str | Sequence[str] = (),
        # type here looks odd, but it's required os you can avoid "partially unknown" type errors with `deps=None`
        deps: AgentDeps | tuple[()] = (),
        retries: int = 1,
        result_tool_name: str = 'final_result',
        result_tool_description: str | None = None,
        result_retries: int | None = None,
    ):
        self.model = models.infer_model(model) if model is not None else None

        self._result_schema = _result.ResultSchema[result_type].build(
            result_type, result_tool_name, result_tool_description
        )
        # if the result tool is None, or its schema allows `str`, we allow plain text results
        self._allow_text_result = self._result_schema is None or self._result_schema.allow_text_result

        self._system_prompts = (system_prompt,) if isinstance(system_prompt, str) else tuple(system_prompt)
        self._retrievers: dict[str, _r.Retriever[AgentDeps, Any]] = {}
        self._default_deps = cast(AgentDeps, None if deps == () else deps)
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
        model: models.Model | KnownModelName | None = None,
        deps: AgentDeps | None = None,
    ) -> result.RunResult[ResponseData]:
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

        if deps is None:
            deps = self._default_deps

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
            try:
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

                        if left := either.left:
                            # left means return a streamed result
                            run_span.set_attribute('all_messages', messages)
                            run_span.set_attribute('cost', cost)
                            handle_span.set_attribute('result', left.value)
                            handle_span.message = 'handle model response -> final result'
                            return result.RunResult(messages, new_message_index, left.value, cost)
                        else:
                            # right means continue the conversation
                            tool_responses = either.right
                            handle_span.set_attribute('tool_responses', tool_responses)
                            response_msgs = ' '.join(m.role for m in tool_responses)
                            handle_span.message = f'handle model response -> {response_msgs}'
                            messages.extend(tool_responses)
            except (ValidationError, exceptions.UnexpectedModelBehaviour) as e:
                run_span.set_attribute('messages', messages)
                # noinspection PyTypeChecker
                raise exceptions.AgentError(messages, model_used) from e

    def run_sync(
        self,
        user_prompt: str,
        *,
        message_history: list[_messages.Message] | None = None,
        model: models.Model | KnownModelName | None = None,
        deps: AgentDeps | None = None,
    ) -> result.RunResult[ResponseData]:
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
        model: models.Model | KnownModelName | None = None,
        deps: AgentDeps | None = None,
    ) -> AsyncIterator[result.StreamedRunResult[AgentDeps, ResponseData]]:
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

        if deps is None:
            deps = self._default_deps

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
            try:
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

                                if left := either.left:
                                    # left means return a streamed result
                                    result_stream = left.value
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
                                    # right means continue the conversation
                                    tool_responses = either.right
                                    handle_span.set_attribute('tool_responses', tool_responses)
                                    response_msgs = ' '.join(m.role for m in tool_responses)
                                    handle_span.message = f'handle model response -> {response_msgs}'
                                    messages.extend(tool_responses)
                                    # the model_response should have been fully streamed by now, we can add it's cost
                                    cost += model_response.cost()

            except exceptions.UnexpectedModelBehaviour as e:
                run_span.set_attribute('messages', messages)
                # noinspection PyTypeChecker
                raise exceptions.AgentError(messages, model_used) from e

    def system_prompt(
        self, func: _system_prompt.SystemPromptFunc[AgentDeps]
    ) -> _system_prompt.SystemPromptFunc[AgentDeps]:
        """Decorator to register a system prompt function that takes `CallContext` as it's only argument."""
        self._system_prompt_functions.append(_system_prompt.SystemPromptRunner(func))
        return func

    def result_validator(
        self, func: _result.ResultValidatorFunc[AgentDeps, ResponseData]
    ) -> _result.ResultValidatorFunc[AgentDeps, ResponseData]:
        """Decorator to register a result validator function."""
        self._result_validators.append(_result.ResultValidator(func))
        return func

    @overload
    def retriever_context(self, func: _r.RetrieverContextFunc[AgentDeps, _r.P], /) -> _r.Retriever[AgentDeps, _r.P]: ...

    @overload
    def retriever_context(
        self, /, *, retries: int | None = None
    ) -> Callable[[_r.RetrieverContextFunc[AgentDeps, _r.P]], _r.Retriever[AgentDeps, _r.P]]: ...

    def retriever_context(
        self, func: _r.RetrieverContextFunc[AgentDeps, _r.P] | None = None, /, *, retries: int | None = None
    ) -> Any:
        """Decorator to register a retriever function."""
        if func is None:

            def retriever_decorator(
                func_: _r.RetrieverContextFunc[AgentDeps, _r.P],
            ) -> _r.Retriever[AgentDeps, _r.P]:
                # noinspection PyTypeChecker
                return self._register_retriever(_utils.Either(left=func_), retries)

            return retriever_decorator
        else:
            # noinspection PyTypeChecker
            return self._register_retriever(_utils.Either(left=func), retries)

    @overload
    def retriever_plain(self, func: _r.RetrieverPlainFunc[_r.P], /) -> _r.Retriever[AgentDeps, _r.P]: ...

    @overload
    def retriever_plain(
        self, /, *, retries: int | None = None
    ) -> Callable[[_r.RetrieverPlainFunc[_r.P]], _r.Retriever[AgentDeps, _r.P]]: ...

    def retriever_plain(self, func: _r.RetrieverPlainFunc[_r.P] | None = None, /, *, retries: int | None = None) -> Any:
        """Decorator to register a retriever function."""
        if func is None:

            def retriever_decorator(func_: _r.RetrieverPlainFunc[_r.P]) -> _r.Retriever[AgentDeps, _r.P]:
                # noinspection PyTypeChecker
                return self._register_retriever(_utils.Either(right=func_), retries)

            return retriever_decorator
        else:
            return self._register_retriever(_utils.Either(right=func), retries)

    def _register_retriever(
        self, func: _r.RetrieverEitherFunc[AgentDeps, _r.P], retries: int | None
    ) -> _r.Retriever[AgentDeps, _r.P]:
        """Private utility to register a retriever function."""
        retries_ = retries if retries is not None else self._default_retries
        retriever = _r.Retriever[AgentDeps, _r.P](func, retries_)

        if self._result_schema and retriever.name in self._result_schema.tools:
            raise ValueError(f'Retriever name conflicts with result schema name: {retriever.name!r}')

        if retriever.name in self._retrievers:
            raise ValueError(f'Retriever name conflicts with existing retriever: {retriever.name!r}')

        self._retrievers[retriever.name] = retriever
        return retriever

    async def _get_agent_model(
        self, model: models.Model | KnownModelName | None
    ) -> tuple[models.Model, models.Model | None, models.AgentModel]:
        """Create a model configured for this agent.

        Args:
            model: model to use for this run, required if `model` was not set when creating the agent.

        Returns:
            a tuple of `(model used, custom_model if any, agent_model)`
        """
        model_: models.Model
        if model is not None:
            custom_model = model_ = models.infer_model(model)
        elif self.model is not None:
            model_ = self.model
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
    ) -> _utils.Either[ResponseData, list[_messages.Message]]:
        """Process a non-streamed response from the model.

        Returns:
            Return `Either` — left: final result data, right: list of messages to send back to the model.
        """
        if model_response.role == 'model-text-response':
            # plain string response
            if self._allow_text_result:
                result_data_input = cast(ResponseData, model_response.content)
                try:
                    result_data = await self._validate_result(result_data_input, deps, None)
                except _result.ToolRetryError as e:
                    self._incr_result_retry()
                    return _utils.Either(right=[e.tool_retry])
                else:
                    return _utils.Either(left=result_data)
            else:
                self._incr_result_retry()
                response = _messages.RetryPrompt(
                    content='Plain text responses are not permitted, please call one of the functions instead.',
                )
                return _utils.Either(right=[response])
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
                        return _utils.Either(right=[e.tool_retry])
                    else:
                        return _utils.Either(left=result_data)

            if not model_response.calls:
                raise exceptions.UnexpectedModelBehaviour('Received empty tool call message')

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
            return _utils.Either(right=messages)
        else:
            assert_never(model_response)

    async def _handle_streamed_model_response(
        self, model_response: models.EitherStreamedResponse, deps: AgentDeps
    ) -> _utils.Either[models.EitherStreamedResponse, list[_messages.Message]]:
        """Process a streamed response from the model.

        Returns:
            Return `Either` — left: final result data, right: list of messages to send back to the model.
        """
        if isinstance(model_response, models.StreamTextResponse):
            # plain string response
            if self._allow_text_result:
                return _utils.Either(left=model_response)
            else:
                self._incr_result_retry()
                response = _messages.RetryPrompt(
                    content='Plain text responses are not permitted, please call one of the functions instead.',
                )
                # stream the response, so cost is correct
                async for _ in model_response:
                    pass

                return _utils.Either(right=[response])
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
                    return _utils.Either(left=model_response)

            # the model is calling a retriever function, consume the response to get the next message
            async for _ in model_response:
                pass
            structured_msg = model_response.get()
            if not structured_msg.calls:
                raise exceptions.UnexpectedModelBehaviour('Received empty tool call message')
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
            return _utils.Either(right=messages)

    async def _validate_result(
        self, result_data: ResponseData, deps: AgentDeps, tool_call: _messages.ToolCall | None
    ) -> ResponseData:
        for validator in self._result_validators:
            result_data = await validator.validate(result_data, deps, self._current_result_retry, tool_call)
        return result_data

    def _incr_result_retry(self) -> None:
        self._current_result_retry += 1
        if self._current_result_retry > self._max_result_retries:
            raise exceptions.UnexpectedModelBehaviour(
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
