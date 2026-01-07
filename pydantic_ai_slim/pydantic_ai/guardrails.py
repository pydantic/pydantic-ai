"""Guardrails for validating agent inputs and outputs.

Guardrails provide a way to validate, filter, and control agent inputs and outputs.
They can be used to enforce safety policies, compliance requirements, or custom business rules.

See [guardrails docs](./guardrails.md) for more information.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from typing import Any, Generic

from typing_extensions import TypeVar

__all__ = (
    'GuardrailResult',
    'InputGuardrailTripwireTriggered',
    'OutputGuardrailTripwireTriggered',
)

T = TypeVar('T', default=None)


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
        async def check_content(
            ctx: RunContext[None], agent: Agent[None, str], prompt: str
        ) -> GuardrailResult[None]:
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
            ```python
            return GuardrailResult.passed(message="Content is safe")
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
            ```python
            return GuardrailResult.blocked(
                message="PII detected in input",
                detected_pii_types=["email", "phone"]
            )
            ```
        """
        return cls(tripwire_triggered=True, output=output, message=message, metadata=metadata)


@dataclass
class InputGuardrailTripwireTriggered(Exception):
    """Exception raised when an input guardrail is triggered.

    This exception is raised when an input guardrail returns a GuardrailResult
    with tripwire_triggered=True. Application code should catch this exception
    to handle blocked inputs appropriately.

    Attributes:
        guardrail_name: Name of the guardrail function that was triggered.
        result: The GuardrailResult that triggered the exception.

    Example:
        ```python
        from pydantic_ai.guardrails import InputGuardrailTripwireTriggered

        try:
            result = await agent.run("blocked content")
        except InputGuardrailTripwireTriggered as e:
            print(f"Input blocked by {e.guardrail_name}: {e.result.message}")
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


@dataclass
class OutputGuardrailTripwireTriggered(Exception):
    """Exception raised when an output guardrail is triggered.

    This exception is raised when an output guardrail returns a GuardrailResult
    with tripwire_triggered=True. Application code should catch this exception
    to handle blocked outputs appropriately.

    Attributes:
        guardrail_name: Name of the guardrail function that was triggered.
        result: The GuardrailResult that triggered the exception.

    Example:
        ```python
        from pydantic_ai.guardrails import OutputGuardrailTripwireTriggered

        try:
            result = await agent.run("prompt")
        except OutputGuardrailTripwireTriggered as e:
            print(f"Output blocked by {e.guardrail_name}: {e.result.message}")
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
