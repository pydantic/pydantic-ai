"""smolvm sandbox capability: gives agents an isolated microVM on this machine to work in.

`SmolSandbox` is the supported entry point; build an agent with it and add tools
that use the workspace, such as `Coder`.
`SmolSandboxBackend` is the smolvm implementation of Pydantic AI's workspace backend protocol,
public for applications that want to boot or attach to a VM themselves and pass it to a run
as `workspace=`.
"""

from pydantic_ai_harness.smol_sandbox._backend import SmolSandboxBackend
from pydantic_ai_harness.smol_sandbox._capability import SmolSandbox

__all__ = [
    'SmolSandbox',
    'SmolSandboxBackend',
]
