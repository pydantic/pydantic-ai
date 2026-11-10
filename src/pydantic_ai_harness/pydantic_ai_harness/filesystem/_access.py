"""The allow, deny, and read-only glob patterns that gate every filesystem tool path."""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from dataclasses import dataclass


def matches(path: str, pattern: str, *, ignore_case: bool = False) -> bool:
    """Glob-match a relative path, treating a leading `**/` as 'any directory, including the root'.

    `fnmatch` has no recursive `**`, so a bare `**/secrets*` would miss a
    root-level `secrets.yaml` -- there's no leading directory to match.
    Retrying with the `**/` prefix stripped covers the zero-directory case.
    """
    spellings = [(path, pattern, fnmatch.fnmatch)]
    if ignore_case:
        # Folding can change a string's length (`ß` becomes `ss`), so a `?` that matches the
        # original spelling may not match the folded one: folding adds a way to match.
        spellings.append((path.casefold(), pattern.casefold(), fnmatch.fnmatchcase))
    return any(
        match(name, glob) or (glob.startswith('**/') and match(name, glob[3:])) for name, glob, match in spellings
    )


def first_matching_pattern(path: str, patterns: Sequence[str]) -> str | None:
    """Return the first denied or read-only pattern matching this path or one of its ancestors.

    A directory pattern protects its descendants even without a `/**` suffix. Matching ignores
    case: on a case-insensitive filesystem (the macOS and Windows default) `.GIT/config` is
    `.git/config`, and the workspace reports a path in the case it was written, so a
    case-sensitive match would let a differently cased spelling past. On a case-sensitive
    filesystem this only protects more. It is exact for literal text; a `?` or `[...]` can still miss a
    name whose case folding changes its length, such as `SS` for `ß` on APFS.
    """
    parts = path.split('/')
    paths = ['/'.join(parts[:end]) for end in range(1, len(parts) + 1)]
    return next((p for p in patterns if any(matches(c, p, ignore_case=True) for c in paths)), None)


@dataclass(frozen=True)
class AccessPatterns:
    """The glob patterns a root-relative path is checked against."""

    allowed: tuple[str, ...]
    """If non-empty, only matching paths are accessible."""
    denied: tuple[str, ...]
    """Matching paths, and paths below them, are not accessible at all."""
    read_only: tuple[str, ...]
    """Matching paths, and paths below them, are readable but not writable."""

    def __bool__(self) -> bool:
        return bool(self.allowed or self.denied or self.read_only)

    def check(self, path: str, *, write: bool = False, check_allowed: bool = True) -> None:
        """Raise `PermissionError` unless the patterns permit this access to `path`.

        `check_allowed=False` skips the `allowed` gate. Walkers (`list_directory`,
        `search_files`, `find_files`) pass it so their root directory isn't required to
        match `allowed` itself -- `.` or `src` would never match a file pattern like
        `src/*.py`. The walk's entries are still filtered per entry via `readable`.
        Denied patterns continue to gate the root.
        """
        if write and (matched := first_matching_pattern(path, self.read_only)):
            raise PermissionError(f'Path {path!r} is protected (matches {matched!r}).')
        if matched := self.denied_by(path):
            raise PermissionError(f'Path {path!r} is denied by pattern {matched!r}.')
        if check_allowed and self.allowed and not any(matches(path, p) for p in self.allowed):
            raise PermissionError(f'Path {path!r} does not match any allowed pattern.')

    def readable(self, path: str) -> bool:
        """Predicate form of the read-level `check`.

        Read-only patterns are not consulted: they gate writes, and the walkers only read.
        """
        if self.denied_by(path) is not None:
            return False
        return not self.allowed or any(matches(path, p) for p in self.allowed)

    def denied_by(self, path: str) -> str | None:
        """The first denied pattern matching `path` or one of its ancestors."""
        return first_matching_pattern(path, self.denied)
