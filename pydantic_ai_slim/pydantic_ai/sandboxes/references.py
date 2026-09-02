"""Serializable sandbox identity."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ('SandboxRef',)


@dataclass(frozen=True, kw_only=True)
class SandboxRef:
    """Serializable identity of an existing sandbox environment.

    A capability's [`get_sandbox`][pydantic_ai.capabilities.AbstractCapability.get_sandbox] turns it
    back into a live backend; credentials and clients live on the capability, never here.
    """

    sandbox_id: str
    """The backend's stable identifier for the environment."""
