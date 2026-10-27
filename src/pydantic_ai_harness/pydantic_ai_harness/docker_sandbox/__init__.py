"""Docker sandbox capability: runs the agent's commands and file edits in a local Docker (or Podman) container.

`DockerSandbox` is the supported entry point; build an agent with it and add tools that use the
workspace, such as `Coder`. `DockerSandboxBackend` is the workspace backend itself, public for
applications that want to build one and pass it to a run as `workspace=`.
"""

from pydantic_ai_harness.docker_sandbox._backend import DockerSandboxBackend
from pydantic_ai_harness.docker_sandbox._capability import DockerSandbox

__all__ = [
    'DockerSandbox',
    'DockerSandboxBackend',
]
