"""Render sandbox workspace provider for standard Harness tools.

External assumptions verified 2026-10-09 against Render SDK 1.2.0 and
https://render.com/docs/sandboxes: asynchronous exec streams remain running after
client disconnect, and copy_to/copy_from transfer file contents. The remote image
provides Python 3 on Linux. Recheck these contracts with the live workspace suite
when upgrading the experimental SDK API.
"""

try:
    from ._backend import RenderSandboxBackend, RenderSandboxConfig
except ImportError as exc:
    raise ImportError('Install `pydantic-ai-harness[render-sandbox]` to use RenderSandbox.') from exc

from ._capability import RenderSandbox
from ._types import RenderSandboxCleanupError, RenderSandboxEvent

__all__ = [
    'RenderSandbox',
    'RenderSandboxBackend',
    'RenderSandboxConfig',
    'RenderSandboxCleanupError',
    'RenderSandboxEvent',
]
