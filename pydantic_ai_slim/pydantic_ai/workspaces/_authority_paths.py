"""Internal helpers that work out the authority of workspace paths."""

from __future__ import annotations as _annotations

from collections.abc import Iterable, Mapping
from typing import TypeAlias

from pydantic_ai.messages import Authority

AUTHORITY_RANK: dict[Authority, int] = {'none': 0, 'user': 1, 'operator': 2}

AuthorityLayer: TypeAlias = tuple[Authority, Mapping[str, Authority]]
"""One wrapper's authority: its level for the workspace's files, and the levels it gives paths, keyed by path."""


def lowest_authority(levels: Iterable[Authority]) -> Authority:
    """The lowest of `levels`; `'operator'`, the highest, when there are none."""
    lowest: Authority = 'operator'
    for level in levels:
        if AUTHORITY_RANK[level] < AUTHORITY_RANK[lowest]:
            lowest = level
    return lowest


def merge_path_authority(paths: Iterable[tuple[str, Authority]]) -> dict[str, Authority]:
    """Map each path to its level, keeping the lowest for a path given more than once."""
    merged: dict[str, Authority] = {}
    for path, level in paths:
        merged[path] = lowest_authority((level, merged.get(path, level)))
    return merged


def is_within(path: str, directory: str) -> bool:
    return path == directory or path.startswith(directory.rstrip('/') + '/')


def longest_match(location: str, path_authority: Mapping[str, Authority], default: Authority) -> Authority:
    matches = [path for path in path_authority if is_within(location, path)]
    return path_authority[max(matches, key=len)] if matches else default


def layer_authority(location: str, layer: AuthorityLayer) -> Authority:
    """The level one wrapper gives a location: its longest matching path's, else its own.

    Paths also match case-folded, keeping the lower level, so on a case-insensitive filesystem (macOS by
    default) another spelling of a lower path can't escape it. On a case-sensitive one, this can only lower
    a level, never raise one.
    """
    level, path_authority = layer
    folded = merge_path_authority((path.casefold(), path_level) for path, path_level in path_authority.items())
    return lowest_authority(
        (longest_match(location, path_authority, level), longest_match(location.casefold(), folded, level))
    )
