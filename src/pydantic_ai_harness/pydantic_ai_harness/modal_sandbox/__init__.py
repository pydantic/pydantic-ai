"""Modal sandbox capability: gives agents an isolated cloud sandbox to work in.

`ModalSandbox` is the supported entry point; build an agent with it and use its
tools. `ModalSandboxSession` exposes lower-level lifecycle, command, and file access
for applications that need to share a caller-owned sandbox across runs. The model-facing
toolset remains an implementation detail of the capability.
"""

from pydantic_ai_harness.modal_sandbox._capability import ModalSandbox
from pydantic_ai_harness.modal_sandbox._session import (
    ModalSandboxAuthError,
    ModalSandboxError,
    ModalSandboxExecResult,
    ModalSandboxSession,
    ModalSandboxTerminalError,
    ModalSandboxUnavailableError,
)

__all__ = [
    'ModalSandbox',
    'ModalSandboxAuthError',
    'ModalSandboxError',
    'ModalSandboxExecResult',
    'ModalSandboxSession',
    'ModalSandboxTerminalError',
    'ModalSandboxUnavailableError',
]
