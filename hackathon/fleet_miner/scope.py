"""Who a proposal should apply to, measured from where its evidence came from rather than guessed from its text."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Literal

from .models import AppliesTo

Scope = Literal['organization', 'team', 'repo']
SHARE = 0.8
"""A pattern is local to one repo (or team) when at least this share of its tagged evidence comes from it."""
MIN_TAGGED = 2
MIN_COVERAGE = 0.5
"""Measure only when at least this many, and this share, of the evidence carries a team or repo."""


def measure_scope(
    evidence: Sequence[tuple[str | None, str | None]], *, window_teams: Iterable[str], window_repos: Iterable[str]
) -> tuple[Scope, str, AppliesTo] | None:
    """Scope from `(team, repo_slug)` per piece of evidence, or `None` when too little of it carries either.

    `repo` when >= 80% of the tagged evidence comes from one repo, or when it all comes from exactly one repo while
    the window holds two or more. Otherwise `team`, by the same rule over teams. Otherwise `organization`.
    """
    repos = Counter(r for _, r in evidence if r)
    teams = Counter(t for t, _ in evidence if t)
    tagged_any = sum(1 for t, r in evidence if t or r)
    # Too little tagged evidence to measure (e.g. 1 of 20 prompts came from an updated clai2): don't let it decide.
    if tagged_any < max(MIN_TAGGED, len(evidence) * MIN_COVERAGE):
        return None
    for kind, counts, everywhere in (('repo', repos, set(window_repos)), ('team', teams, set(window_teams))):
        if not counts:
            continue
        top, n = counts.most_common(1)[0]
        tagged = sum(counts.values())
        only_one = len(counts) == 1 and len(everywhere) >= 2
        if n / tagged >= SHARE or only_one:
            reason = (
                f'Measured: {n} of {tagged} tagged prompts/calls come from {kind} `{top}` '
                f'({len(everywhere)} {kind}s active in the window).'
            )
            applies = AppliesTo(repos=[top]) if kind == 'repo' else AppliesTo(teams=[top])
            return kind, reason, applies  # pyright: ignore[reportReturnType]
    reason = (
        f'Measured: spread over {len(repos)} repo(s) and {len(teams)} team(s), none with {SHARE:.0%} of the '
        f'tagged evidence.'
    )
    return 'organization', reason, AppliesTo()
