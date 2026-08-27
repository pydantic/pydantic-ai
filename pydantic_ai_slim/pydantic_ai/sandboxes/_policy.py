"""Package-internal policy for the framework-provided sandbox default."""

from .protocol import SandboxBackend
from .unavailable import UnavailableSandbox

DEFAULT_SANDBOX_UNAVAILABLE_REASON = (
    'No sandbox is attached to this run. Pass `sandbox=LocalSandbox()` to the run method to use the '
    'local machine (unsafe: commands and file operations run with the full permissions of this process), '
    'attach a capability that supplies a sandbox through its `create_sandbox` hook, or pass a `SandboxRef` '
    'to connect to an existing environment. See https://ai.pydantic.dev/sandbox/ for details.'
)


def default_sandbox_backend() -> SandboxBackend:
    """Create the framework-provided sandbox backend for a new run.

    Deliberately unable to execute anything: host access is never implied, so every operation
    raises with the attachment instructions.
    """
    return UnavailableSandbox(reason=DEFAULT_SANDBOX_UNAVAILABLE_REASON)
