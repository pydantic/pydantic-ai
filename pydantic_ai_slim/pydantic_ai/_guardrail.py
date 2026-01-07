"""Internal guardrail wrapper classes.

This module contains the internal implementation of guardrail wrappers.
For the public API, see [`guardrails`][pydantic_ai.guardrails].
"""

from __future__ import annotations as _annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic

from typing_extensions import TypeVar

from . import _utils
from ._run_context import AgentDepsT, RunContext
from .guardrails import GuardrailResult
from .messages import UserContent

if TYPE_CHECKING:
    from .agent import Agent

__all__ = (
    'InputGuardrail',
    'OutputGuardrail',
    'InputGuardrailFunc',
    'OutputGuardrailFunc',
)

T = TypeVar('T', default=None)
OutputDataT = TypeVar('OutputDataT', default=str)


# Input guardrail functions receive the run context, agent, and user prompt,
# and return a GuardrailResult indicating whether the guardrail was triggered.
# The function can be sync or async.
InputGuardrailFunc = (
    Callable[['RunContext[AgentDepsT]', 'Agent[AgentDepsT, Any]', 'str | Sequence[UserContent]'], GuardrailResult[T]]
    | Callable[
        ['RunContext[AgentDepsT]', 'Agent[AgentDepsT, Any]', 'str | Sequence[UserContent]'],
        Awaitable[GuardrailResult[T]],
    ]
)

# Output guardrail functions receive the run context, agent, and agent output,
# and return a GuardrailResult indicating whether the guardrail was triggered.
# The function can be sync or async.
OutputGuardrailFunc = (
    Callable[['RunContext[AgentDepsT]', 'Agent[AgentDepsT, OutputDataT]', OutputDataT], GuardrailResult[T]]
    | Callable[['RunContext[AgentDepsT]', 'Agent[AgentDepsT, OutputDataT]', OutputDataT], Awaitable[GuardrailResult[T]]]
)


@dataclass
class InputGuardrail(Generic[AgentDepsT, T]):
    """Wrapper for input guardrail functions.

    Input guardrails validate user prompts before the agent runs.
    They can run in parallel with the agent (default) or block until validation passes.

    See [guardrails docs](./guardrails.md) for more information.

    Attributes:
        function: The guardrail function to execute.
        run_in_parallel: If True (default), guardrail runs concurrently with agent.
            If False, guardrail must pass before agent starts.
        name: Name of the guardrail (defaults to function name).
    """

    function: InputGuardrailFunc[AgentDepsT, T]
    """The guardrail function to execute."""

    run_in_parallel: bool = True
    """If True (default), guardrail runs concurrently with agent."""

    name: str | None = None
    """Name of the guardrail (defaults to function name)."""

    _is_async: bool = field(init=False, repr=False)

    def __post_init__(self):
        self._is_async = _utils.is_async_callable(self.function)
        if self.name is None:
            self.name = getattr(self.function, '__name__', 'input_guardrail')

    async def run(
        self,
        agent: Agent[AgentDepsT, Any],
        prompt: str | Sequence[UserContent],
        run_context: RunContext[AgentDepsT],
    ) -> GuardrailResult[T]:
        """Execute the guardrail function.

        Args:
            agent: The agent being run.
            prompt: The user prompt being validated.
            run_context: The current run context.

        Returns:
            The guardrail result indicating pass/fail.
        """
        args = (run_context, agent, prompt)

        if self._is_async:
            return await self.function(*args)  # type: ignore
        else:
            return await _utils.run_in_executor(self.function, *args)  # type: ignore


@dataclass
class OutputGuardrail(Generic[AgentDepsT, OutputDataT, T]):
    """Wrapper for output guardrail functions.

    Output guardrails validate agent responses before returning to the user.
    They always run sequentially after the agent completes.

    See [guardrails docs](./guardrails.md) for more information.

    Attributes:
        function: The guardrail function to execute.
        name: Name of the guardrail (defaults to function name).
    """

    function: OutputGuardrailFunc[AgentDepsT, OutputDataT, T]
    """The guardrail function to execute."""

    name: str | None = None
    """Name of the guardrail (defaults to function name)."""

    _is_async: bool = field(init=False, repr=False)

    def __post_init__(self):
        self._is_async = _utils.is_async_callable(self.function)
        if self.name is None:
            self.name = getattr(self.function, '__name__', 'output_guardrail')

    async def run(
        self,
        agent: Agent[AgentDepsT, OutputDataT],
        output: OutputDataT,
        run_context: RunContext[AgentDepsT],
    ) -> GuardrailResult[T]:
        """Execute the guardrail function.

        Args:
            agent: The agent that produced the output.
            output: The agent output being validated.
            run_context: The current run context.

        Returns:
            The guardrail result indicating pass/fail.
        """
        args = (run_context, agent, output)

        if self._is_async:
            return await self.function(*args)  # type: ignore
        else:
            return await _utils.run_in_executor(self.function, *args)  # type: ignore
