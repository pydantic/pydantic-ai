from dataclasses import dataclass
from typing import Awaitable, Callable, Concatenate, Generic, ParamSpec, TypeVar

AgentContext = TypeVar('AgentContext')
# retrieval function parameters
P = ParamSpec('P')
# retrieval function return type
R = TypeVar('R')


@dataclass
class CallInfo(Generic[AgentContext]):
    """Information about the current call."""

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
