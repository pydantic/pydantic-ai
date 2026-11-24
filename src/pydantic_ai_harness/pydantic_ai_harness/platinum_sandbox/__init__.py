"""Platinum sandbox capability: gives agents an isolated cloud computer to work in.

`PlatinumSandbox` is the supported entry point; build an agent with it and add tools that use the
workspace, such as `Coder`. `PlatinumSandboxBackend` is the Platinum implementation of Pydantic AI's
workspace backend protocol, public for applications that want to create or attach to a sandbox
themselves and pass it to a run as `workspace=`.
"""

from pydantic_ai_harness.platinum_sandbox._backend import PlatinumSandboxBackend
from pydantic_ai_harness.platinum_sandbox._capability import PlatinumSandbox

__all__ = [
    'PlatinumSandbox',
    'PlatinumSandboxBackend',
]
