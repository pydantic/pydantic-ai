"""Deprecated pre-rename name for `DockerContainer`, kept for agent-spec compatibility."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.tools import AgentDepsT
from pydantic_ai_harness.docker_container._capability import DockerContainer


@dataclass(init=False)
class DockerSandbox(DockerContainer[AgentDepsT]):
    """Deprecated name for `DockerContainer`.

    Keeps the pre-rename `'DockerSandbox'` agent-spec block name, so specs saved before the
    rename still load when this class is passed via `custom_capability_types`. Specs
    saved through this class keep the old block name; migrate to `DockerContainer` to
    save under the new one.

    `init=False` keeps the inherited `__init__`: one regenerated here would resolve its
    annotations against this module, hiding fields from the `AgentSpec` schema.
    """

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'DockerSandbox'
