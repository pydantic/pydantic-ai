"""Deprecated pre-rename name for `DockerContainer`, kept for agent-spec compatibility."""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import deprecated

from pydantic_ai.tools import AgentDepsT
from pydantic_ai_harness._warn import HarnessDeprecationWarning
from pydantic_ai_harness.docker_container._capability import DockerContainer


@deprecated(
    '`DockerSandbox` is now `DockerContainer`: the old name implied a security sandbox, but a container shares the '
    'host kernel. In a spec, rename the `DockerSandbox` block to `DockerContainer`; the options are unchanged.',
    category=HarnessDeprecationWarning,
)
@dataclass(init=False)
class DockerSandbox(DockerContainer[AgentDepsT]):
    """Deprecated name for `DockerContainer`.

    Its class name is its agent-spec block name, so specs saved before the rename, with a
    `DockerSandbox` block, still load when this class is passed via `custom_capability_types`.
    Specs saved through this class keep that block name; rename the block to `DockerContainer`
    in the spec to migrate. Subclasses keep their own class name, as before the rename.

    `init=False` keeps the inherited `__init__`: one regenerated here would resolve its
    annotations against this module, hiding fields from the `AgentSpec` schema.
    """
