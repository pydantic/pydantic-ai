"""Exceptions raised by the guardrail capabilities."""

from __future__ import annotations


class GuardrailError(Exception):
    """Base exception for guardrail violations."""


class InputBlocked(GuardrailError):
    """Raised by a user-supplied input guardrail to hard-fail a run.

    Prefer returning `False` from the guard callable to trigger a graceful
    refusal via [`SkipModelRequest`][pydantic_ai.exceptions.SkipModelRequest].
    Raise this explicitly when the caller should have to handle the failure.
    """


class OutputBlocked(GuardrailError):
    """Raised by [`OutputGuardrail`][pydantic_ai_harness.OutputGuardrail] when the final output fails validation."""


class ToolBlocked(GuardrailError):
    """Raised by a user-supplied tool guard to hard-fail a run.

    Prefer returning `False` (or `GuardrailResult.block()`) from the guard callable:
    the refusal message becomes the tool result, so the agent sees why the call
    was denied and can choose another approach. Raise this explicitly when the
    caller, not the agent, should have to handle the failure.

    Args:
        tool_name: The tool whose call was refused.
        reason: Why it was refused. Optional.
    """

    tool_name: str
    reason: str | None

    def __init__(self, tool_name: str, reason: str | None = None):
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f'Tool {tool_name!r} blocked: {reason}' if reason else f'Tool {tool_name!r} blocked')
