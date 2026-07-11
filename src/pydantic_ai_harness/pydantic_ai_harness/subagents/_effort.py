"""Minimum thinking-effort floor shared by the sub-agent capability and its orchestrator."""

from __future__ import annotations

from pydantic_ai.settings import ThinkingEffort, ThinkingLevel

MINIMUM_EFFORT_FLOOR: ThinkingEffort = 'low'
"""Lowest thinking effort any agent this capability builds is allowed to run at.

`clamp_effort` raises anything below this to the floor. The constant is exported
so an orchestrator that builds its own agents can apply the same floor to them
(the orchestrator-side application is the caller's responsibility)."""

_EFFORT_RANK: dict[ThinkingEffort, int] = {'minimal': 0, 'low': 1, 'medium': 2, 'high': 3, 'xhigh': 4}
"""Ordering of the concrete effort levels, low to high."""


def clamp_effort(level: ThinkingLevel | None, floor: ThinkingEffort = MINIMUM_EFFORT_FLOOR) -> ThinkingLevel:
    """Raise a thinking level to at least `floor`.

    - `None` or `False` (thinking unset or disabled, both below any floor) become `floor`.
    - `True` (thinking on at the provider's default effort) is left as `True`: its
      magnitude is provider-defined and not comparable to a named level, so it is
      neither downgraded nor bumped.
    - A concrete effort below `floor` becomes `floor`; one at or above `floor` is unchanged.
    """
    if level is None or level is False:
        return floor
    if level is True:
        return True
    if _EFFORT_RANK[level] < _EFFORT_RANK[floor]:
        return floor
    return level
