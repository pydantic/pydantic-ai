"""Package-internal policy for the framework-provided sandbox default."""

from .protocol import SandboxBackend
from .unavailable import UnavailableSandbox

DEFAULT_SANDBOX_UNAVAILABLE_REASON = (
    'No sandbox is attached to this run. Pass `sandbox=LocalSandbox()` to the run method to use the '
    'local machine (unsafe: commands and file operations run with the full permissions of this process), '
    'attach a capability that supplies a sandbox through its `acquire_sandbox` hook, or pass a `SandboxRef` '
    'to connect to an existing environment. See https://ai.pydantic.dev/sandbox/ for details.'
)


class _DefaultUnavailableSandbox(UnavailableSandbox):
    """Nominal marker for framework policy, never application configuration."""


def default_sandbox_backend(*, reason: str = DEFAULT_SANDBOX_UNAVAILABLE_REASON) -> SandboxBackend:
    """Create the framework-provided sandbox backend for a new run.

    Deliberately unable to execute anything: host access is never implied, so every operation
    raises with the attachment instructions.
    """
    return _DefaultUnavailableSandbox(reason=reason)


def is_default_sandbox_backend(backend: object) -> bool:
    """Whether `backend` is the framework's implicit unavailable placeholder."""
    return isinstance(backend, _DefaultUnavailableSandbox)
