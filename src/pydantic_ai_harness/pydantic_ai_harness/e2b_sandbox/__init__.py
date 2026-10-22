"""E2B sandbox capability: gives agents an isolated cloud computer to work in.

`E2BSandbox` is the supported entry point; build an agent with it and add tools
that use the workspace, such as `Coder`.
`E2BSandboxBackend` is the E2B implementation of Pydantic AI's workspace backend protocol,
public for applications that want to create or attach to a sandbox themselves and pass it to
a run as `workspace=`.
"""

from pydantic_ai_harness.e2b_sandbox._backend import E2BSandboxBackend
from pydantic_ai_harness.e2b_sandbox._capability import E2BSandbox

__all__ = [
    'E2BSandbox',
    'E2BSandboxBackend',
]
