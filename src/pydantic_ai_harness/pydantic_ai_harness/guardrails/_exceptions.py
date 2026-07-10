"""Exceptions raised by the guardrail capabilities."""

from __future__ import annotations


class GuardrailError(Exception):
    """Base exception for guardrail violations."""


class InputBlocked(GuardrailError):
    """Raised by a user-supplied input guard to hard-fail a run.

    Prefer returning `False` from the guard callable to trigger a graceful
    refusal via [`SkipModelRequest`][pydantic_ai.exceptions.SkipModelRequest].
    Raise this explicitly when the caller should have to handle the failure.
    """


class OutputBlocked(GuardrailError):
    """Raised by [`OutputGuard`][pydantic_ai_harness.OutputGuard] when the final output fails validation."""
