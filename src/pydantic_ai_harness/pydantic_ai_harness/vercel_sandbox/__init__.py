"""Vercel Sandbox capability: gives agents an isolated cloud computer to work in.

`VercelSandbox` is the supported entry point; build an agent with it and add tools that use the
workspace, such as `Coder`. `VercelSandboxBackend` is the Vercel implementation of Pydantic AI's
workspace backend protocol, public for applications that want to create or attach to a sandbox
themselves and pass it to a run as `workspace=`.
"""

from pydantic_ai_harness.vercel_sandbox._backend import VercelSandboxBackend
from pydantic_ai_harness.vercel_sandbox._capability import VercelSandbox

__all__ = [
    'VercelSandbox',
    'VercelSandboxBackend',
]
