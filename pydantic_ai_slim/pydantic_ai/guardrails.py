"""Guardrails for validating agent inputs and outputs.

Guardrails provide a way to validate, filter, and control agent inputs and outputs.
They can be used to enforce safety policies, compliance requirements, or custom business rules.
"""

from __future__ import annotations as _annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic

from typing_extensions import TypeVar

from . import _utils
from .exceptions import AgentRunError
from .messages import UserContent

if TYPE_CHECKING:
    from ._run_context import RunContext

__all__ = (
    'GuardrailResult',
    'InputGuardrailTripwireTriggered',
    'OutputGuardrailTripwireTriggered',
    'InputGuardrail',
    'OutputGuardrail',
)

T = TypeVar('T', default=None)
AgentDepsT = TypeVar('AgentDepsT', default=None)
OutputDataT = TypeVar('OutputDataT', default=str)


@dataclass
class GuardrailResult(Generic[T]):
    """Result from a guardrail function.

    Guardrail functions return this type to indicate whether the guardrail was triggered
    and provide optional structured output and metadata.

    Example:
        ```python
        from pydantic_ai import Agent, GuardrailResult, RunContext

        agent = Agent('openai:gpt-4o')

        @agent.input_guardrail
        async def check_content(ctx: RunContext[None], prompt: str) -> GuardrailResult[None]:
            if 'blocked_word' in prompt.lower():
                return GuardrailResult.blocked(message='Content contains blocked word')
            return GuardrailResult.passed()
        ```

    Attributes:
        tripwire_triggered: Whether the guardrail was triggered (True = blocked/failed).
        output: Structured output from the guardrail (e.g., classification result).
        message: Human-readable explanation of the guardrail decision.
        metadata: Additional context for logging/audit.
    """

    tripwire_triggered: bool
    """Whether the guardrail was triggered (True = blocked/failed)."""

    output: T | None = None
    """Structured output from the guardrail (e.g., classification result)."""

    message: str | None = None
    """Human-readable explanation of the guardrail decision."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional context for logging/audit."""

    @classmethod
    def passed(cls, output: T | None = None, message: str | None = None) -> GuardrailResult[T]:
        """Factory for a passing guardrail result.

        Args:
            output: Optional structured output from the guardrail.
            message: Optional human-readable message.

        Returns:
            A GuardrailResult with tripwire_triggered=False.

        Example:
            ```python {test="skip" lint="skip"}
            result = GuardrailResult.passed(message='Content is safe')
            ```
        """
        return cls(tripwire_triggered=False, output=output, message=message)

    @classmethod
    def blocked(cls, output: T | None = None, message: str | None = None, **metadata: Any) -> GuardrailResult[T]:
        """Factory for a triggered (blocked) guardrail result.

        Args:
            output: Optional structured output from the guardrail.
            message: Optional human-readable explanation of why the guardrail was triggered.
            **metadata: Additional key-value pairs for logging/audit.

        Returns:
            A GuardrailResult with tripwire_triggered=True.

        Example:
            ```python {test="skip" lint="skip"}
            result = GuardrailResult.blocked(
                message='PII detected in input',
                detected_pii_types=['email', 'phone'],
            )
            ```
        """
        return cls(tripwire_triggered=True, output=output, message=message, metadata=metadata)


class InputGuardrailTripwireTriggered(AgentRunError):
    """Exception raised when an input guardrail is triggered.

    This exception is raised when an input guardrail returns a GuardrailResult
    with tripwire_triggered=True. Application code should catch this exception
    to handle blocked inputs appropriately.

    Attributes:
        guardrail_name: Name of the guardrail function that was triggered.
        result: The GuardrailResult that triggered the exception.

    Example:
        ```python {test="skip" lint="skip"}
        from pydantic_ai.guardrails import InputGuardrailTripwireTriggered


        async def main():
            try:
                await agent.run('blocked content')
            except InputGuardrailTripwireTriggered as e:
                print(f'Input blocked by {e.guardrail_name}: {e.result.message}')
        ```
    """

    guardrail_name: str
    """Name of the guardrail function that was triggered."""

    result: GuardrailResult[Any]
    """The GuardrailResult that triggered the exception."""

    def __init__(self, guardrail_name: str, result: GuardrailResult[Any]):
        self.guardrail_name = guardrail_name
        self.result = result
        message = result.message or f'Input guardrail "{guardrail_name}" triggered'
        super().__init__(message)


class OutputGuardrailTripwireTriggered(AgentRunError):
    """Exception raised when an output guardrail is triggered.

    This exception is raised when an output guardrail returns a GuardrailResult
    with tripwire_triggered=True. Application code should catch this exception
    to handle blocked outputs appropriately.

    Attributes:
        guardrail_name: Name of the guardrail function that was triggered.
        result: The GuardrailResult that triggered the exception.

    Example:
        ```python {test="skip" lint="skip"}
        from pydantic_ai.guardrails import OutputGuardrailTripwireTriggered


        async def main():
            try:
                await agent.run('prompt')
            except OutputGuardrailTripwireTriggered as e:
                print(f'Output blocked by {e.guardrail_name}: {e.result.message}')
        ```
    """

    guardrail_name: str
    """Name of the guardrail function that was triggered."""

    result: GuardrailResult[Any]
    """The GuardrailResult that triggered the exception."""

    def __init__(self, guardrail_name: str, result: GuardrailResult[Any]):
        self.guardrail_name = guardrail_name
        self.result = result
        message = result.message or f'Output guardrail "{guardrail_name}" triggered'
        super().__init__(message)


# Input guardrail function signature: (ctx, prompt) -> GuardrailResult
InputGuardrailFunc = (
    Callable[['RunContext[AgentDepsT]', 'str | Sequence[UserContent]'], GuardrailResult[T]]
    | Callable[['RunContext[AgentDepsT]', 'str | Sequence[UserContent]'], Awaitable[GuardrailResult[T]]]
)

# Output guardrail function signature: (ctx, output) -> GuardrailResult
OutputGuardrailFunc = (
    Callable[['RunContext[AgentDepsT]', OutputDataT], GuardrailResult[T]]
    | Callable[['RunContext[AgentDepsT]', OutputDataT], Awaitable[GuardrailResult[T]]]
)


@dataclass
class InputGuardrail(Generic[AgentDepsT, T]):
    """Wrapper for input guardrail functions.

    Input guardrails validate user prompts before the agent runs.

    Attributes:
        function: The guardrail function to execute.
        blocking: If True, guardrail must pass before other guardrails run.
            If False (default), guardrail runs concurrently with other non-blocking guardrails.
        name: Name of the guardrail (defaults to function name).
    """

    function: InputGuardrailFunc[AgentDepsT, T]
    """The guardrail function to execute."""

    blocking: bool = False
    """If True, guardrail must pass before other guardrails run."""

    name: str | None = None
    """Name of the guardrail (defaults to function name)."""

    _is_async: bool = field(init=False, repr=False)

    def __post_init__(self):
        self._is_async = _utils.is_async_callable(self.function)
        if self.name is None:
            self.name = getattr(self.function, '__name__', 'input_guardrail')

    async def run(
        self,
        prompt: str | Sequence[UserContent],
        run_context: RunContext[AgentDepsT],
    ) -> GuardrailResult[T]:
        """Execute the guardrail function."""
        if self._is_async:
            return await self.function(run_context, prompt)  # type: ignore
        else:
            return await _utils.run_in_executor(self.function, run_context, prompt)  # type: ignore


@dataclass
class OutputGuardrail(Generic[AgentDepsT, OutputDataT, T]):
    """Wrapper for output guardrail functions.

    Output guardrails validate agent responses before returning to the user.
    They always run sequentially after the agent completes.

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
        output: OutputDataT,
        run_context: RunContext[AgentDepsT],
    ) -> GuardrailResult[T]:
        """Execute the guardrail function."""
        if self._is_async:
            return await self.function(run_context, output)  # type: ignore
        else:
            return await _utils.run_in_executor(self.function, run_context, output)  # type: ignore
