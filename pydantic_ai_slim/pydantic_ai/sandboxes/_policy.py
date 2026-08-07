"""Package-internal policy for the framework-provided sandbox default."""

import os

from .local import LocalSandbox
from .protocol import SandboxBackend
from .unavailable import UnavailableSandbox

DEFAULT_SANDBOX_UNAVAILABLE_REASON = (
    'The default local sandbox requires a POSIX platform because its timeout contract must kill the whole process '
    'group. Attach a container- or VM-based sandbox instead.'
)


class DefaultLocalSandbox(LocalSandbox):
    """Marker subclass for the framework-owned, per-run default local sandbox."""


def default_sandbox_backend() -> SandboxBackend:
    """Create the framework-provided sandbox backend for a new run."""
    if os.name == 'posix':
        return DefaultLocalSandbox()
    return UnavailableSandbox(reason=DEFAULT_SANDBOX_UNAVAILABLE_REASON)
