"""The compatibility matrix: plugins another plugin already includes, so the two never run together."""

from collections.abc import Mapping

from pydantic_ai_harness.subagents import SubAgents

INCLUDED: Mapping[str, Mapping[str, type[object] | None]] = {
    'pydantic_clai2.builtin_plugins.coder': {
        'pydantic_clai2.builtin_plugins.compaction': None,
        'pydantic_ai_harness.subagents:SubAgents': SubAgents,
    },
}
"""Factory of a plugin, to the factories of the plugins it already provides.

Each included factory names the capability class the plugin must bind for that to hold, or `None`
when it always does: `coder` binds `SubAgents` only with `sub_agents` on, so with it off a separate
`SubAgents` stays available. While a plugin is on, the plugins it includes stay off and `/plugins`
greys them out. Keyed by factory, not id, so a user's own id for the same plugin, or a row saved
from the former harness catalog, matches too.
"""


def included_by(factory: str, includes: Mapping[str, frozenset[str]]) -> str | None:
    """The name of a loaded plugin that includes `factory`, given loaded plugin names to the factories each includes."""
    return next((name for name, included in includes.items() if factory in included), None)
