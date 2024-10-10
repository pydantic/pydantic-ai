from __future__ import annotations as _annotations

import asyncio
from collections.abc import Awaitable, Sequence
from copy import copy
from typing import Any, Callable, Generic, Literal, Union, cast, overload

from typing_extensions import assert_never

from . import _system_prompt, _utils, messages as _messages, models as _models, result as _result, retrievers as _r
from .result import ResultData
from .retrievers import AgentDependencies

__all__ = ('Agent',)


KnownModelName = Literal['openai:gpt-4o', 'openai:gpt-4-turbo', 'openai:gpt-4', 'openai:gpt-3.5-turbo']

SysPromptContext = Callable[[_r.CallContext[AgentDependencies]], Union[str, Awaitable[str]]]
SysPromptPlain = Callable[[], Union[str, Awaitable[str]]]


class Agent(Generic[AgentDependencies, ResultData]):
    """Main class for creating "agents" - a way to have a specific type of "conversation" with an LLM."""

    # slots mostly for my sanity — knowing what attributes are available
    __slots__ = (
        '_model',
        'result_schema',
        '_allow_plain_message',
        '_system_prompts',
        '_retrievers',
        '_default_retries',
        '_system_prompt_functions',
        '_default_deps',
    )

    def __init__(
        self,
        model: _models.Model | KnownModelName | None = None,
        response_type: type[_result.ResultData] = str,
        *,
        system_prompt: str | Sequence[str] = (),
        retrievers: Sequence[_r.Retriever[AgentDependencies, Any]] = (),
        deps: AgentDependencies | tuple[None] = (None,),
        retries: int = 1,
        response_schema_name: str = 'final_response',
        response_schema_description: str = 'The final response',
        response_retries: int | None = None,
    ):
        self._model = _models.infer_model(model) if model is not None else None

        self.result_schema = _result.ResultSchema[response_type].build(
            response_type,
            response_schema_name,
            response_schema_description,
            response_retries if response_retries is not None else retries,
        )
        self._allow_plain_message = self.result_schema is None or self.result_schema.allow_plain_message

        self._system_prompts = (system_prompt,) if isinstance(system_prompt, str) else tuple(system_prompt)
        self._retrievers: dict[str, _r.Retriever[AgentDependencies, Any]] = {r_.name: r_ for r_ in retrievers}
        if self.result_schema and self.result_schema.name in self._retrievers:
            raise ValueError(f'Retriever name conflicts with response schema: {self.result_schema.name!r}')
        self._default_deps: AgentDependencies = None if deps == (None,) else deps  # type: ignore
        self._default_retries = retries
        self._system_prompt_functions: list[_system_prompt.SystemPromptRunner[AgentDependencies]] = []

    def with_deps(self, deps: AgentDependencies) -> Agent[AgentDependencies, ResultData]:
        """Return a new agent with the given dependencies."""
        agent: Agent[AgentDependencies, ResultData] = Agent.__new__(Agent)
        for attr in self.__slots__:
            setattr(agent, attr, copy(getattr(self, attr)))
        agent._default_deps = deps
        return agent

    async def run(
        self,
        user_prompt: str,
        *,
        message_history: list[_messages.Message] | None = None,
        model: _models.Model | KnownModelName | None = None,
        deps: AgentDependencies | None = None,
    ) -> _result.RunResult[_result.ResultData]:
        """Run the agent with a user prompt in async mode.

        Args:
            user_prompt: User input to start/continue the conversation.
            message_history: History of the conversation so far.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.

        Returns:
            The result of the run.
        """
        if model is not None:
            model_ = _models.infer_model(model)
        elif self._model is not None:
            model_ = self._model
        else:
            raise RuntimeError('`model` must be set either when creating the agent or when calling it.')

        if deps is None:
            deps = self._default_deps

        if message_history is not None:
            # shallow copy messages
            messages = message_history.copy()
        else:
            messages = await self._init_messages(deps)

        messages.append(_messages.UserPrompt(user_prompt))

        functions: list[_models.AbstractToolDefinition] = list(self._retrievers.values())
        if self.result_schema is not None:
            functions.append(self.result_schema)
        agent_model = model_.agent_model(self._allow_plain_message, functions)

        for retriever in self._retrievers.values():
            retriever.reset()

        while True:
            llm_message = await agent_model.request(messages)
            opt_result = await self._handle_model_response(messages, llm_message, deps)
            if opt_result is not None:
                return _result.RunResult(opt_result.value, messages, cost=_result.Cost(0))

    def run_sync(
        self,
        user_prompt: str,
        *,
        message_history: list[_messages.Message] | None = None,
        model: _models.Model | KnownModelName | None = None,
        deps: AgentDependencies | None = None,
    ) -> _result.RunResult[_result.ResultData]:
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

    async def stream(self, user_prompt: str) -> _result.RunStreamResult[_result.ResultData]:
        """Run the agent with a user prompt asynchronously and stream the results."""
        raise NotImplementedError()

    def system_prompt(
        self, func: _system_prompt.SystemPromptFunc[AgentDependencies]
    ) -> _system_prompt.SystemPromptFunc[AgentDependencies]:
        """Decorator to register a system prompt function that takes `CallContext` as it's only argument."""
        self._system_prompt_functions.append(_system_prompt.SystemPromptRunner(func))
        return func

    @overload
    def retriever_context(
        self, func: _r.RetrieverContextFunc[AgentDependencies, _r.P], /
    ) -> _r.Retriever[AgentDependencies, _r.P]: ...

    @overload
    def retriever_context(
        self, /, *, retries: int | None = None
    ) -> Callable[[_r.RetrieverContextFunc[AgentDependencies, _r.P]], _r.Retriever[AgentDependencies, _r.P]]: ...

    def retriever_context(
        self, func: _r.RetrieverContextFunc[AgentDependencies, _r.P] | None = None, /, *, retries: int | None = None
    ) -> Any:
        """Decorator to register a retriever function."""
        if func is None:

            def retriever_decorator(
                func_: _r.RetrieverContextFunc[AgentDependencies, _r.P],
            ) -> _r.Retriever[AgentDependencies, _r.P]:
                # noinspection PyTypeChecker
                return self._register_retriever(func_, True, retries)

            return retriever_decorator
        else:
            return self._register_retriever(func, True, retries)

    @overload
    def retriever_plain(self, func: _r.RetrieverPlainFunc[_r.P], /) -> _r.Retriever[AgentDependencies, _r.P]: ...

    @overload
    def retriever_plain(
        self, /, *, retries: int | None = None
    ) -> Callable[[_r.RetrieverPlainFunc[_r.P]], _r.Retriever[AgentDependencies, _r.P]]: ...

    def retriever_plain(self, func: _r.RetrieverPlainFunc[_r.P] | None = None, /, *, retries: int | None = None) -> Any:
        """Decorator to register a retriever function."""
        if func is None:

            def retriever_decorator(func_: _r.RetrieverPlainFunc[_r.P]) -> _r.Retriever[AgentDependencies, _r.P]:
                # noinspection PyTypeChecker
                return self._register_retriever(func_, False, retries)

            return retriever_decorator
        else:
            return self._register_retriever(func, False, retries)

    def _register_retriever(
        self, func: _r.RetrieverEitherFunc[AgentDependencies, _r.P], takes_ctx: bool, retries: int | None
    ) -> _r.Retriever[AgentDependencies, _r.P]:
        """Private utility to register a retriever function."""
        retries_ = retries if retries is not None else self._default_retries
        retriever = _r.Retriever[AgentDependencies, _r.P](func, takes_ctx, retries_)

        if self.result_schema and self.result_schema.name == retriever.name:
            raise ValueError(f'Retriever name conflicts with response schema name: {retriever.name!r}')

        if retriever.name in self._retrievers:
            raise ValueError(f'Retriever name conflicts with existing retriever: {retriever.name!r}')

        self._retrievers[retriever.name] = retriever
        return retriever

    async def _handle_model_response(
        self, messages: list[_messages.Message], llm_message: _messages.LLMMessage, deps: AgentDependencies
    ) -> _utils.Option[ResultData]:
        """Process a single response from the model.

        Returns:
            Return `None` to continue the conversation, or a result to end it.
        """
        messages.append(llm_message)
        if llm_message.role == 'llm-response':
            # plain string response
            if self._allow_plain_message:
                return _utils.Some(cast(ResultData, llm_message.content))
            else:
                messages.append(_messages.PlainResponseForbidden())
        elif llm_message.role == 'llm-function-calls':
            if self.result_schema is not None:
                # if there's a result schema, and any of the calls match that name, return the result
                call = next((c for c in llm_message.calls if c.function_name == self.result_schema.name), None)
                if call is not None:
                    either = self.result_schema.validate(call)
                    if result_data := either.left:
                        return _utils.Some(result_data)
                    else:
                        messages.append(either.right)
                        return None

            # otherwise we run all functions in parallel
            coros: list[Awaitable[_messages.Message]] = []
            for call in llm_message.calls:
                retriever = self._retrievers.get(call.function_name)
                if retriever is None:
                    # TODO return message?
                    raise ValueError(f'Unknown function name: {call.function_name!r}')
                coros.append(retriever.run(deps, call))
            messages += await asyncio.gather(*coros)
        else:
            assert_never(llm_message)

    async def _init_messages(self, deps: AgentDependencies) -> list[_messages.Message]:
        """Build the initial messages for the conversation."""
        messages: list[_messages.Message] = [_messages.SystemPrompt(p) for p in self._system_prompts]
        for sys_prompt_runner in self._system_prompt_functions:
            prompt = await sys_prompt_runner.run(deps)
            messages.append(_messages.SystemPrompt(prompt))
        return messages
