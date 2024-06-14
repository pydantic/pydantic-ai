from dataclasses import dataclass
from typing import Protocol, TypeVar, Any, Callable, Generic, ParamSpec, Concatenate

AgentContext = TypeVar('AgentContext')
# retrieval function parameters
P = ParamSpec('P')
# retrieval function return type
R = TypeVar('R')


@dataclass
class CallInfo(Generic[AgentContext]):
    context: AgentContext
    retry: int


RetrieverFunction = Callable[Concatenate[CallInfo[AgentContext], P], R]


class SystemPromptWithArg(Protocol):
    def __call__(self, info: CallInfo) -> str: ...


class SystemPromptWithoutArg(Protocol):
    def __call__(self) -> str: ...


class AsyncSystemPromptWithArg(Protocol):
    async def __call__(self, info: CallInfo) -> str: ...


class AsyncSystemPromptWithoutArg(Protocol):
    async def __call__(self) -> str: ...


SystemPrompt = TypeVar(
    'SystemPrompt',
    SystemPromptWithArg,
    SystemPromptWithoutArg,
    AsyncSystemPromptWithArg,
    AsyncSystemPromptWithoutArg,
)


@dataclass
class FallbackArguments:
    arguments: dict[str, Any]
