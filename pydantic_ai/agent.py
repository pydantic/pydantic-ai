from __future__ import annotations as _annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable, Generic, Literal, Sequence, assert_never, cast, overload

from . import _utils, messages as _messages, models as _models, result as _result, retrievers as _r
from .result import ResultData
from .retrievers import AgentContext

__all__ = ('Agent',)


KnownModelName = Literal['openai:gpt-4o', 'openai:gpt-4-turbo', 'openai:gpt-4', 'openai:gpt-3.5-turbo']


class Agent(Generic[ResultData, AgentContext]):
    """Main class for creating "agents" - a way to have a specific type of "conversation" with an LLM."""

    def __init__(
        self,
        model: _models.Model | KnownModelName | None = None,
        response_type: type[_result.ResultData] = str,
        *,
        system_prompt: str | Sequence[str] = (),
        retrievers: Sequence[_r.Retriever[AgentContext, Any]] = (),
        context: AgentContext = None,
        retries: int = 1,
        response_schema_name: str = 'final_response',
        response_schema_description: str = 'The final response',
        response_retries: int | None = None,
    ):
        self._model = _models.infer_model(model) if model is not None else None

        self._result_schema = _result.ResultSchema[response_type].build(
            response_type,
            response_schema_name,
            response_schema_description,
            response_retries if response_retries is not None else retries,
        )
        self._allow_plain_message = self._result_schema is None or self._result_schema.allow_plain_message

        self._system_prompts = (system_prompt,) if isinstance(system_prompt, str) else tuple(system_prompt)
        self._retrievers: dict[str, _r.Retriever[AgentContext, Any]] = {r_.name: r_ for r_ in retrievers}
        if self._result_schema and self._result_schema.name in self._retrievers:
            raise ValueError(f'Retriever name conflicts with response schema: {self._result_schema.name!r}')
        self._context = context
        self._default_retries = retries
        self._system_prompt_functions: list[Any] = []

    async def run(
        self,
        user_prompt: str,
        *,
        message_history: list[_messages.Message] | None = None,
        model: _models.Model | KnownModelName | None = None,
    ) -> _result.RunResult[_result.ResultData]:
        """Run the agent with a user prompt in async mode.

        Args:
            user_prompt: User input to start/continue the conversation.
            message_history: History of the conversation so far.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.

        Returns:
            The result of the run.
        """
        if model is not None:
            model_ = _models.infer_model(model)
        elif self._model is not None:
            model_ = self._model
        else:
            raise RuntimeError('`model` must be set either when creating the agent or when calling it.')

        if message_history is not None:
            # shallow copy messages
            messages = message_history.copy()
        else:
            messages = await self._init_messages()

        messages.append(_messages.UserPrompt(user_prompt))

        functions: list[_models.AbstractToolDefinition] = list(self._retrievers.values())
        if self._result_schema is not None:
            functions.append(self._result_schema)
        agent_model = model_.agent_model(self._allow_plain_message, functions)

        for retriever in self._retrievers.values():
            retriever.reset()

        while True:
            llm_message = await agent_model.request(messages)
            opt_result = await self._handle_model_response(messages, llm_message)
            if opt_result is not None:
                return _result.RunResult(opt_result.value, messages, cost=_result.Cost(0))

    def run_sync(
        self,
        user_prompt: str,
        *,
        message_history: list[_messages.Message] | None = None,
        model: _models.Model | KnownModelName | None = None,
    ) -> _result.RunResult[_result.ResultData]:
        """Run the agent with a user prompt synchronously.

        This is a convenience method that wraps `self.run` with `asyncio.run()`.

        Args:
            user_prompt: User input to start/continue the conversation.
            message_history: History of the conversation so far.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.

        Returns:
            The result of the run.
        """
        return asyncio.run(self.run(user_prompt, message_history=message_history, model=model))

    async def stream(self, user_prompt: str) -> _result.RunStreamResult[_result.ResultData]:
        """Run the agent with a user prompt asynchronously and stream the results."""
        raise NotImplementedError()

    def system_prompt(self, func: _SystemPromptFunction[AgentContext]) -> _SystemPromptFunction[AgentContext]:
        """Decorator to register a system prompt function."""
        self._system_prompt_functions.append(func)
        return func

    @overload
    def retriever(self, func: _r.RetrieverFunc[AgentContext, _r.P], /) -> _r.Retriever[AgentContext, _r.P]: ...

    @overload
    def retriever(
        self, /, *, retries: int | None = None
    ) -> Callable[
        [_r.RetrieverFunc[AgentContext, _r.P]],
        _r.Retriever[AgentContext, _r.P],
    ]: ...

    def retriever(
        self, func: _r.RetrieverFunc[AgentContext, _r.P] | None = None, /, *, retries: int | None = None
    ) -> Any:
        """Decorator to register a retriever function."""
        if func is None:

            def retriever_decorator(func_: _r.RetrieverFunc[AgentContext, _r.P]) -> _r.Retriever[AgentContext, _r.P]:
                # noinspection PyTypeChecker
                return self._register_retriever(func_, retries)

            return retriever_decorator
        else:
            return self._register_retriever(func, retries)

    def _register_retriever(
        self, func: _r.RetrieverFunc[AgentContext, _r.P], retries: int | None
    ) -> _r.Retriever[AgentContext, _r.P]:
        retries_ = retries if retries is not None else self._default_retries
        retriever = _r.Retriever[AgentContext, _r.P].build(func, retries_)

        if self._result_schema and self._result_schema.name == retriever.name:
            raise ValueError(f'Retriever name conflicts with response schema name: {retriever.name!r}')

        if retriever.name in self._retrievers:
            raise ValueError(f'Retriever name conflicts with existing retriever: {retriever.name!r}')

        self._retrievers[retriever.name] = retriever
        return retriever

    async def _handle_model_response(
        self, messages: list[_messages.Message], llm_message: _messages.LLMMessage
    ) -> _utils.Option[ResultData]:
        messages.append(llm_message)
        if llm_message.role == 'llm-response':
            # plain string response
            if self._allow_plain_message:
                return _utils.Some(cast(ResultData, llm_message.content))
            else:
                messages.append(_messages.PlainResponseForbidden())
        elif llm_message.role == 'llm-function-calls':
            if self._result_schema is not None:
                # if there's a result schema, and any of the calls match that name, return the result
                call = next((c for c in llm_message.calls if c.function_name == self._result_schema.name), None)
                if call is not None:
                    either = self._result_schema.validate(call)
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
                coros.append(retriever.run(self._context, call))
            messages += await asyncio.gather(*coros)
        else:
            assert_never(llm_message)

    async def _init_messages(self) -> list[_messages.Message]:
        messages: list[_messages.Message] = [_messages.SystemPrompt(p) for p in self._system_prompts]
        for func in self._system_prompt_functions:
            prompt = await self._run_system_prompt_function(func)
            messages.append(_messages.SystemPrompt(prompt))
        return messages

    async def _run_system_prompt_function(self, func: _SystemPromptFunction[AgentContext]) -> str:
        takes_call_info = len(inspect.signature(func).parameters) == 1
        if asyncio.iscoroutinefunction(func):
            if takes_call_info:
                return await func(_r.CallInfo(self._context, 0))
            else:
                return await func()
        else:
            if takes_call_info:
                f = cast(Callable[[_r.CallInfo[AgentContext]], str], func)
                return await _utils.run_in_executor(f, _r.CallInfo(self._context, 0))
            else:
                f = cast(Callable[[], str], func)
                return await _utils.run_in_executor(f)


# This is basically a function that may or maybe not take `CallInfo` as an argument, and may or may not be async.
# Usage `SystemPrompt[AgentContext]`
_SystemPromptFunction = (
    Callable[[_r.CallInfo[AgentContext]], str]
    | Callable[[_r.CallInfo[AgentContext]], Awaitable[str]]
    | Callable[[], str]
    | Callable[[], Awaitable[str]]
)
