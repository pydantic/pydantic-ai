"""Docker workspace capability: runs the agent's commands and file edits in a local Docker (or Podman) container.

`DockerWorkspace` is the supported entry point; build an agent with it and add tools that use the
workspace, such as `Coder`. `DockerWorkspaceBackend` is the workspace backend itself, public for
applications that want to build one and pass it to a run as `workspace=`.
"""

from pydantic_ai_harness.docker_workspace._backend import DockerWorkspaceBackend
from pydantic_ai_harness.docker_workspace._capability import DockerWorkspace

__all__ = [
    'DockerWorkspace',
    'DockerWorkspaceBackend',
]
