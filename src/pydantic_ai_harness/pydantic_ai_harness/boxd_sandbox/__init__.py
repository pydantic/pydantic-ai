"""boxd sandbox capability: gives agents an isolated cloud computer to work in.

`BoxdSandbox` is the supported entry point; build an agent with it and add tools that use the
workspace, such as `Coder`. `BoxdSandboxBackend` is the boxd implementation of Pydantic AI's
workspace backend protocol, public for applications that want to create or attach to a machine
themselves and pass it to a run as `workspace=`.
"""

from pydantic_ai_harness.boxd_sandbox._backend import BoxdSandboxBackend
from pydantic_ai_harness.boxd_sandbox._capability import BoxdSandbox

__all__ = [
    'BoxdSandbox',
    'BoxdSandboxBackend',
]
