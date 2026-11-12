"""Deprecated pre-rename name for `DockerContainer`, kept for agent-spec compatibility."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.tools import AgentDepsT
from pydantic_ai_harness.docker_container._capability import DockerContainer


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
