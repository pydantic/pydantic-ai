"""BoxLite sandbox capability: gives agents an isolated micro-VM on this machine to work in.

`BoxLiteSandbox` is the supported entry point; build an agent with it and add tools
that use the workspace, such as `Coder`.
`BoxLiteSandboxBackend` is the BoxLite implementation of Pydantic AI's workspace backend protocol,
public for applications that want to create or attach to a box themselves and pass it to a run
as `workspace=`.
"""

from pydantic_ai_harness.boxlite_sandbox._backend import BoxLiteSandboxBackend
from pydantic_ai_harness.boxlite_sandbox._capability import BoxLiteSandbox

__all__ = [
    'BoxLiteSandbox',
    'BoxLiteSandboxBackend',
]
