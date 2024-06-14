from dataclasses import dataclass
from typing import TypeVar, Any, Callable, Generic, ParamSpec, Concatenate, Awaitable

AgentContext = TypeVar('AgentContext')
# retrieval function parameters
P = ParamSpec('P')
# retrieval function return type
R = TypeVar('R')


@dataclass
class CallInfo(Generic[AgentContext]):
    context: AgentContext
    retry: int


# Usage `Retriever[AgentContext, P, R]`
Retriever = Callable[Concatenate[CallInfo[AgentContext], P], R]

# Usage `SystemPrompt[AgentContext]`
SystemPrompt = (
    Callable[[CallInfo[AgentContext]], None]
    | Callable[[CallInfo[AgentContext]], Awaitable[None]]
    | Callable[[], None]
    | Callable[[], Awaitable[None]]
)


@dataclass
class FallbackArguments:
    arguments: dict[str, Any]
