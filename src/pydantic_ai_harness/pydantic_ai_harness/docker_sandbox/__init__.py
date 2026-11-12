"""Deprecated import location for `pydantic_ai_harness.docker_container`.

This capability was renamed because "sandbox" implied a security boundary that a plain
container, which shares the host's kernel, doesn't give. Importing from here still works
but emits a `HarnessDeprecationWarning`. Import from `pydantic_ai_harness.docker_container` instead.
"""

from pydantic_ai_harness._warn import warn_module_renamed
from pydantic_ai_harness.docker_container import DockerContainerBackend
from pydantic_ai_harness.docker_container._deprecated import DockerSandbox

DockerSandboxBackend = DockerContainerBackend

warn_module_renamed(
    'docker_sandbox',
    'docker_container',
    reason=(
        '`DockerSandbox` is now `DockerContainer` and `DockerSandboxBackend` is now `DockerContainerBackend`: '
        'the old names implied a security sandbox, but a container shares the host kernel and its commands run as '
        "the image's user, usually root."
    ),
)

__all__ = [
    'DockerSandbox',
    'DockerSandboxBackend',
]
