"""Fail when a change adds an entry to a baseline a test gate keeps of the violations it already found.

A gate that finds violations on `main` lists them in a baseline and fails on any other. The baseline may only shrink:
fixing a violation removes its entry, and a change may not add one. Each baseline is a set or dict literal assigned
to a module-level name, so this reads it without importing the module.

Usage:
    uv run scripts/check_test_baselines.py --staged                     # the index against HEAD, for pre-commit
    uv run scripts/check_test_baselines.py --base REF                   # the working tree against REF
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Sequence

from _ratchet import Change, escape, git, merge_head, parse_args, read_changes, read_objects

BASELINES: dict[str, tuple[str, ...]] = {
    'tests/leak_guards.py': ('GC_FINALIZED_BASELINE',),
}
"""Each file that holds a baseline, with the names its baselines are assigned to."""


def entries(source: str | None, name: str) -> set[str] | None:
    """The string entries of the set or dict literal assigned to `name` at module level, or `None` without one."""
    if source is None:
        return None
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if not (isinstance(target, ast.Name) and target.id == name):
            continue
        if isinstance(value, ast.Call) and value.args:  # `frozenset({...})`
            value = value.args[0]
        keys = value.keys if isinstance(value, ast.Dict) else value.elts if isinstance(value, ast.Set) else []
        return {key.value for key in keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}
    return None


def added(changes: list[Change], theirs: str | None = None) -> list[tuple[str, str, str]]:
    """Each `(path, name, entry)` a change added to a baseline; during a merge, only one that `theirs` lacks too.

    A change that creates a baseline adds nothing: there was none to shrink.
    """
    found: list[tuple[str, str, str]] = []
    for change in changes:
        path = change.new_path
        if path not in BASELINES:
            continue
        # `git cat-file --batch` reports a path the merged commit lacks as missing, which `read_objects` can't parse.
        merged = read_objects([f'{theirs}:{path}'])[0] if theirs and git('ls-tree', theirs, '--', path) else None
        for name in BASELINES[path]:
            ours, theirs_entries = entries(change.old_source, name), entries(merged, name)
            if ours is None and theirs_entries is None:
                continue
            before = (ours or set()) | (theirs_entries or set())
            found.extend((path, name, entry) for entry in sorted((entries(change.new_source, name) or set()) - before))
    return found


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check and return the exit code."""
    options = parse_args(argv, 'Check that no change adds to a test gate baseline.')
    found = added(read_changes(options, list(BASELINES)), merge_head() if options.staged else None)
    for path, name, entry in found:
        message = f'`{name}` gained `{entry}`: fix the violation instead; a baseline only shrinks'
        print(f'::error file={escape(path, prop=True)}::{escape(message)}' if options.github else f'{path}: {message}')
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main())
