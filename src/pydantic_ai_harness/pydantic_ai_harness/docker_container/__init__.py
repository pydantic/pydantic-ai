"""Docker container capability: runs the agent's commands and file edits in a local Docker (or Podman) container.

`DockerContainer` is the supported entry point; build an agent with it and add tools that use the
workspace, such as `Coder`. `DockerContainerBackend` is the workspace backend itself, public for
applications that want to build one and pass it to a run as `workspace=`.
"""

from pydantic_ai_harness.docker_container._backend import DESKTOP_DOCKERFILE, DockerContainerBackend
from pydantic_ai_harness.docker_container._capability import DockerContainer

__all__ = [
    'DESKTOP_DOCKERFILE',
    'DockerContainer',
    'DockerContainerBackend',
]
