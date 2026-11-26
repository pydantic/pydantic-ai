"""Daytona sandbox capability: gives agents an isolated cloud computer to work in.

`DaytonaSandbox` is the supported entry point; build an agent with it and add tools that use the
workspace, such as `Coder`. `DaytonaSandboxBackend` is the Daytona implementation of Pydantic AI's
workspace backend protocol, public for applications that want to create or attach to a sandbox
themselves and pass it to a run as `workspace=`.
"""

from pydantic_ai_harness.daytona_sandbox._backend import DaytonaSandboxBackend
from pydantic_ai_harness.daytona_sandbox._capability import DaytonaSandbox

__all__ = [
    'DaytonaSandbox',
    'DaytonaSandboxBackend',
]
