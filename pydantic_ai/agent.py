from __future__ import annotations as _annotations

from typing import Generic, Any, Callable, Sequence, overload, Concatenate

from .result import RunResult, RunStreamResult, ResponseType
from . import function_calls as fc
from .function_calls import AgentContext


class Agent(Generic[ResponseType, AgentContext]):
    def __init__(
        self,
        system_prompt: str | Sequence[str] = '',
        response_type: type[ResponseType] = str,
        context: AgentContext = None,
    ):
        self._system_prompt = system_prompt
        self._system_prompt_functions: list[Any] = []
        self._response_type = response_type
        self._context = context

    def run(self, user_prompt: str) -> RunResult[ResponseType]:
        raise NotImplementedError()

    async def async_run(self, user_prompt: str) -> RunResult[ResponseType]:
        raise NotImplementedError()

    def stream(self, user_prompt: str) -> RunStreamResult[ResponseType]:
        raise NotImplementedError()

    async def async_stream(self, user_prompt: str) -> RunStreamResult[ResponseType]:
        raise NotImplementedError()

    def system_prompt(self, func: fc.SystemPrompt) -> fc.SystemPrompt:
        self._system_prompt_functions.append(func)
        return func

    @overload
    def retriever(
        self, func: Callable[Concatenate[fc.CallInfo[AgentContext], fc.P], fc.R], /
    ) -> Callable[Concatenate[fc.CallInfo[AgentContext], fc.P], fc.R]: ...

    @overload
    def retriever(
        self, *, retries: int = 0
    ) -> Callable[
        [Callable[Concatenate[fc.CallInfo[AgentContext], fc.P], fc.R]],
        Callable[Concatenate[fc.CallInfo[AgentContext], fc.P], fc.R],
    ]: ...

    def retriever(self, __func: Any | None = None, /, retries: int = 0) -> Any:
        if __func is None:

            def retriever_decorator(
                func: Callable[Concatenate[fc.CallInfo[AgentContext], fc.P], fc.R],
            ) -> Callable[Concatenate[fc.CallInfo[AgentContext], fc.P], fc.R]:
                self._register_retriever(func, retries)
                return func

            return retriever_decorator
        else:
            self._register_retriever(__func, retries)
            return __func

    def _register_retriever(
        self, func: Callable[Concatenate[fc.CallInfo[AgentContext], fc.P], fc.R], retries: int
    ) -> None:
        pass
