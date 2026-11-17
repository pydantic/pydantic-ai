"""OpenSandbox sandbox capability: gives agents a sandbox on your own OpenSandbox server to work in.

`OpenSandbox` is the supported entry point; build an agent with it and add tools that use the
workspace, such as `Coder`. `OpenSandboxBackend` is the OpenSandbox implementation of Pydantic AI's
workspace backend protocol, public for applications that want to create or attach to a sandbox themselves
and pass it to a run as `workspace=`.
"""

from pydantic_ai_harness.opensandbox_sandbox._backend import OpenSandboxBackend
from pydantic_ai_harness.opensandbox_sandbox._capability import OpenSandbox

__all__ = [
    'OpenSandbox',
    'OpenSandboxBackend',
]
