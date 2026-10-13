"""The compatibility matrix: plugins another plugin already includes, so the two never run together."""

from collections.abc import Mapping

from pydantic_ai_harness.compaction import (
    FallbackCompaction,
    SlidingWindowCompaction,
    SummarizingCompaction,
    TieredCompaction,
)
from pydantic_ai_harness.subagents import SubAgents

Binds = type[object] | tuple[type[object], ...]
"""The capability class, or any of several, a plugin must bind to include another; an `isinstance` target."""

HISTORY_COMPACTION: tuple[type[object], ...] = (
    FallbackCompaction,
    SummarizingCompaction,
    SlidingWindowCompaction,
    TieredCompaction,
)
"""Harness strategies that rewrite history the way the `compaction` plugin's chain does.

`ClearToolResults`, which `Coder` binds, is not one: it only empties old tool results and works
before the chain, so `coder` and `compaction` run together.
"""

INCLUDED: Mapping[str, Mapping[str, Binds]] = {
    'pydantic_clai2.builtin_plugins.coder': {
        'pydantic_clai2.builtin_plugins.compaction': HISTORY_COMPACTION,
        'pydantic_ai_harness.subagents:SubAgents': SubAgents,
    },
}
"""Factory of a plugin, to the factories of the plugins it already provides.

Each included factory names the capability classes the plugin must bind for that to hold:
`coder` binds `SubAgents` only with `sub_agents` on, so with it off a separate `SubAgents` stays
available, and it binds no history compaction, so `compaction` runs alongside it. While a plugin
includes another, the included one stays off and `/plugins` greys it out, so a second chain is
never registered. Keyed by factory, not id, so a user's own id for the same plugin, or a row saved
from the former harness catalog, matches too.
"""


def included_by(factory: str, includes: Mapping[str, frozenset[str]]) -> str | None:
    """The name of a loaded plugin that includes `factory`, given loaded plugin names to the factories each includes."""
    return next((name for name, included in includes.items() if factory in included), None)
