"""Serializable sandbox identity."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ('SandboxRef',)


@dataclass(frozen=True, kw_only=True)
class SandboxRef:
    """Serializable identity for an existing sandbox environment.

    Turned back into a live [`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] by the
    capability chain's [`get_sandbox`][pydantic_ai.capabilities.AbstractCapability.get_sandbox]
    hook; credentials and clients live on capabilities, never in the identity.
    """

    provider: str
    """Short identifier of the backing implementation (e.g. `'e2b'`, `'docker'`).

    Must equal the `provider` the identified backend reports; capabilities use it to recognize
    their own refs.
    """

    sandbox_id: str
    """The implementation's stable identifier for the environment, unique per provider."""

    capability_id: str | None = None
    """Effective ID of the capability responsible for reconnecting and destroying the sandbox.

    Framework-created refs are stamped with the winning supplier's run-local ID. Give a supplier
    an explicit stable `id` when its refs cross runs or processes. A ref supplied by a caller may
    omit this for backward compatibility, in which case the capability chain is searched using
    normal precedence.
    """
