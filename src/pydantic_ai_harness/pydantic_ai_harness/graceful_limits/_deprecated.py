"""Deprecated pre-rename name for `GracefulLimits`, kept for agent-spec compatibility."""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import deprecated

from pydantic_ai.tools import AgentDepsT
from pydantic_ai_harness._warn import HarnessDeprecationWarning
from pydantic_ai_harness.graceful_limits._capability import GracefulLimits


@deprecated(
    '`WrapUpAtLimits` is now `GracefulLimits`, as it now also ends a run gracefully at its `timeout=`. '
    'In a spec, rename the `WrapUpAtLimits` block to `GracefulLimits`; the options are unchanged.',
    category=HarnessDeprecationWarning,
)
@dataclass(init=False)
class WrapUpAtLimits(GracefulLimits[AgentDepsT]):
    """Deprecated name for `GracefulLimits`.

    Its class name is its agent-spec block name, so specs saved before the rename, with a
    `WrapUpAtLimits` block, still load when this class is passed via `custom_capability_types`.
    Specs saved through this class keep that block name; rename the block to `GracefulLimits`
    in the spec to migrate. Subclasses keep their own class name, as before the rename.

    `init=False` keeps the inherited `__init__`: one regenerated here would resolve its
    annotations against this module, hiding fields from the `AgentSpec` schema.
    """
