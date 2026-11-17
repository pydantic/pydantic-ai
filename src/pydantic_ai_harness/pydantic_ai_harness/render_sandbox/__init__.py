"""Render sandbox capability: runs an agent's commands and file edits in a Render sandbox.

`RenderSandbox` is the supported entry point; build an agent with it and add tools
that use the workspace, such as `Coder`.
`RenderSandboxBackend` is the Render implementation of Pydantic AI's workspace backend protocol,
public for applications that want to create or attach to a sandbox themselves and pass it to
a run as `workspace=`.
"""

from pydantic_ai_harness.render_sandbox._backend import RenderSandboxBackend
from pydantic_ai_harness.render_sandbox._capability import RenderSandbox

__all__ = [
    'RenderSandbox',
    'RenderSandboxBackend',
]
