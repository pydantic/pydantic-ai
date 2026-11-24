"""Deprecated pre-rename name for `AnnounceCurrentTime`, kept for agent-spec compatibility."""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import deprecated

from pydantic_ai.tools import AgentDepsT
from pydantic_ai_harness._warn import HarnessDeprecationWarning
from pydantic_ai_harness.announce_current_time._capability import AnnounceCurrentTime


@deprecated(
    '`CurrentTime` is now `AnnounceCurrentTime`: capabilities that act on the run are named with a verb phrase. '
    'In a spec, rename the `CurrentTime` block to `AnnounceCurrentTime`; the options are unchanged.',
    category=HarnessDeprecationWarning,
)
@dataclass(init=False)
class CurrentTime(AnnounceCurrentTime[AgentDepsT]):
    """Deprecated name for `AnnounceCurrentTime`.

    Its class name is its agent-spec block name, so specs saved before the rename, with a
    `CurrentTime` block, still load when this class is passed via `custom_capability_types`.
    Specs saved through this class keep that block name; rename the block to `AnnounceCurrentTime`
    in the spec to migrate. Subclasses keep their own class name, as before the rename.

    `init=False` keeps the inherited `__init__`: one regenerated here would resolve its
    annotations against this module, hiding fields from the `AgentSpec` schema.
    """
