from __future__ import annotations as _annotations

from typing import Any, Awaitable, Callable, Generic, Sequence, overload

from . import retrievers as r
from .result import ResponseType, RunResult, RunStreamResult
from .retrievers import AgentContext

# This is basically a function that may or maybe not take `CallInfo` as an argument, and may or may not be async.
# Usage `SystemPrompt[AgentContext]`
SystemPromptFunction = (
    Callable[[r.CallInfo[AgentContext]], str]
    | Callable[[r.CallInfo[AgentContext]], Awaitable[str]]
    | Callable[[], str]
    | Callable[[], Awaitable[str]]
)


class Agent(Generic[ResponseType, AgentContext]):
    """Main class for creating "agents" - a way to have a specific type of "conversation" with an LLM."""

    def __init__(
        self,
        system_prompt: str | Sequence[str] = '',
        retrievers: Sequence[r.Retriever[AgentContext, Any]] = (),
        response_type: type[ResponseType] = str,
        context: AgentContext = None,
    ):
        self._system_prompt = system_prompt
        self._retrievers: dict[str, r.Retriever[AgentContext, Any]] = {r_.name: r_ for r_ in retrievers}
        self._system_prompt_functions: list[Any] = []
        self._response_type = response_type
        self._context = context

    def run(self, user_prompt: str) -> RunResult[ResponseType]:
        """Run the agent with a user prompt."""
        raise NotImplementedError()

    async def async_run(self, user_prompt: str) -> RunResult[ResponseType]:
        """Run the agent with a user prompt asynchronously."""
        raise NotImplementedError()

    def stream(self, user_prompt: str) -> RunStreamResult[ResponseType]:
        """Run the agent with a user prompt and stream the results."""
        raise NotImplementedError()

    async def async_stream(self, user_prompt: str) -> RunStreamResult[ResponseType]:
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
