from __future__ import annotations as _annotations

import asyncio
import inspect
from datetime import datetime
from typing import Any, Awaitable, Callable, Generic, Literal, Sequence, cast, overload

from . import _utils, llm, result, retrievers as r
from .result import ResponseType
from .retrievers import AgentContext


class Agent(Generic[ResponseType, AgentContext]):
    """Main class for creating "agents" - a way to have a specific type of "conversation" with an LLM."""

    def __init__(
        self,
        model: llm.Model | Literal['openai-gpt-4o', 'openai-gpt-4-turbo', 'openai-gpt-4', 'openai-gpt-3.5-turbo'],
        response_type: type[result.ResponseType] = str,
        *,
        system_prompt: str | Sequence[str] = '',
        retrievers: Sequence[r.Retriever[AgentContext, Any]] = (),
        context: AgentContext = None,
    ):
        self._model = llm.infer_model(model)
        self._response_type = response_type
        self._system_prompts = (system_prompt,) if isinstance(system_prompt, str) else tuple(system_prompt)
        self._retrievers: dict[str, r.Retriever[AgentContext, Any]] = {r_.name: r_ for r_ in retrievers}
        self._system_prompt_functions: list[Any] = []
        self._context = context

    async def run(
        self, user_prompt: str, message_history: list[result.Message] | None = None
    ) -> result.RunResult[result.ResponseType]:
        """Run the agent with a user prompt in async mode."""
        if message_history is not None:
            # shallow copy messages
            messages = message_history.copy()
        else:
            messages = await self._init_messages()

        messages.append(result.UserPrompt(role='user', timestamp=datetime.now(), content=user_prompt))
        while True:
            _response = await self._model.request(messages)
            # TODO handle response

    def run_sync(self, user_prompt: str) -> result.RunResult[result.ResponseType]:
        """Run the agent with a user prompt synchronously.

        This is a convenience method that wraps `self.run` with `asyncio.run()`.
        """
        return asyncio.run(self.run(user_prompt))

    async def stream(self, user_prompt: str) -> result.RunStreamResult[result.ResponseType]:
        """Run the agent with a user prompt asynchronously and stream the results."""
        raise NotImplementedError()

    def system_prompt(self, func: SystemPromptFunction[AgentContext]) -> SystemPromptFunction[AgentContext]:
        """Decorator to register a system prompt function."""
        self._system_prompt_functions.append(func)
        return func

    @overload
    def retriever(self, func: r.RetrieverFunc[AgentContext, r.P], /) -> r.Retriever[AgentContext, r.P]: ...

    @overload
    def retriever(
        self, /, *, retries: int = 0
    ) -> Callable[
        [r.RetrieverFunc[AgentContext, r.P]],
        r.Retriever[AgentContext, r.P],
    ]: ...

    def retriever(self, func: r.RetrieverFunc[AgentContext, r.P] | None = None, /, *, retries: int = 0) -> Any:
        """Decorator to register a retriever function."""
        if func is None:

            def retriever_decorator(
                func_: r.RetrieverFunc[AgentContext, r.P],
            ) -> r.Retriever[AgentContext, r.P]:
                return self._register_retriever(func_, retries)

            return retriever_decorator
        else:
            return self._register_retriever(func, retries)

    def _register_retriever(
        self, func: r.RetrieverFunc[AgentContext, r.P], retries: int
    ) -> r.Retriever[AgentContext, r.P]:
        retriever = r.Retriever[AgentContext, r.P].build(func, retries)
        self._retrievers[retriever.name] = retriever
        return retriever

    async def _init_messages(self) -> list[result.Message]:
        messages: list[result.Message] = [result.SystemPrompt(role='system', content=p) for p in self._system_prompts]
        for func in self._system_prompt_functions:
            prompt = await self._run_system_prompt_function(func)
            messages.append(result.SystemPrompt(role='system', content=prompt))
        return messages

    async def _run_system_prompt_function(self, func: SystemPromptFunction[AgentContext]) -> str:
        takes_call_info = len(inspect.signature(func).parameters) == 1
        if asyncio.iscoroutinefunction(func):
            if takes_call_info:
                return await func(r.CallInfo(self._context, 0))
            else:
                return await func()
        else:
            if takes_call_info:
                f = cast(Callable[[r.CallInfo[AgentContext]], str], func)
                return await _utils.run_in_executor(f, r.CallInfo(self._context, 0))
            else:
                f = cast(Callable[[], str], func)
                return await _utils.run_in_executor(f)


# This is basically a function that may or maybe not take `CallInfo` as an argument, and may or may not be async.
# Usage `SystemPrompt[AgentContext]`
SystemPromptFunction = (
    Callable[[r.CallInfo[AgentContext]], str]
    | Callable[[r.CallInfo[AgentContext]], Awaitable[str]]
    | Callable[[], str]
    | Callable[[], Awaitable[str]]
)
