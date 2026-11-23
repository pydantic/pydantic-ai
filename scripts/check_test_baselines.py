"""Fail when a change adds an entry to a baseline a test gate keeps of the violations it already found.

A gate that finds violations on `main` lists them in a baseline and fails on any other. The baseline may only shrink:
fixing a violation removes its entry, and a change may not add one. Each baseline is a set or dict literal of strings,
or `frozenset()` of one, assigned once to a module-level name, so this reads it without importing the module.

Only the change that introduces `BASELINES` creates baselines. Once the base registers one, every entry of a baseline
the base lacks counts as added, so a change can't add a second baseline or carry an existing one's entries to a new
name, and a baseline leaves `BASELINES` only once the base holds it empty.

Usage:
    uv run scripts/check_test_baselines.py --staged                     # the index against HEAD, for pre-commit
    uv run scripts/check_test_baselines.py --base REF                   # the working tree against REF
"""

from __future__ import annotations

import ast
import subprocess
import sys
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

from _ratchet import escape, merge_head, parse_args, read_objects

CHECKER = 'scripts/check_test_baselines.py'

BASELINES: dict[str, str] = {
    'GC_FINALIZED_BASELINE': 'tests/leak_guards.py',
}
"""Each baseline's name, with the file that assigns it."""


class InvalidBaseline(Exception):
    """A baseline or registry this check can't read as a literal, so it can't tell what a change adds."""


def assigned(source: str | None, name: str) -> ast.expr | None:
    """The value `source` assigns to `name` at module level, or `None` without one."""
    bindings = [
        node
        for node in ast.parse(source or '').body
        if isinstance(node, ast.Assign | ast.AugAssign | ast.AnnAssign)
        and node.value is not None
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
    ]
    if len(bindings) > 1 or any(isinstance(node, ast.AugAssign) for node in bindings):
        raise InvalidBaseline(f'`{name}` is assigned more than once: assign it once')
    return bindings[0].value if bindings else None


def entries(source: str | None, name: str) -> set[str] | None:
    """The entries of the baseline `source` assigns to `name`, or `None` without one."""
    value = assigned(source, name)
    if value is None:
        return None
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'frozenset':
        if not value.args and not value.keywords:
            return set()
        if len(value.args) == 1 and not value.keywords:
            value = value.args[0]
    keys = value.keys if isinstance(value, ast.Dict) else value.elts if isinstance(value, ast.Set) else None
    found = {key.value for key in keys or () if isinstance(key, ast.Constant) and isinstance(key.value, str)}
    if keys is None or len(found) < len(keys):
        raise InvalidBaseline(f'`{name}` is not a set or dict literal of strings, or `frozenset()` of one')
    return found


def registry(source: str | None) -> dict[str, str]:
    """The `BASELINES` a copy of this script registers, or none when there is no copy."""
    value = assigned(source, 'BASELINES')
    if value is None:
        return {}
    pairs = list(zip(value.keys, value.values)) if isinstance(value, ast.Dict) else None
    found = {
        key.value: path.value
        for key, path in pairs or ()
        if isinstance(key, ast.Constant)
        and isinstance(key.value, str)
        and isinstance(path, ast.Constant)
        and isinstance(path.value, str)
    }
    if pairs is None or len(found) < len(pairs):
        raise InvalidBaseline('`BASELINES` is not a dict literal of strings')
    return found


def show(revision: str, path: str) -> str | None:
    """`path` at `revision`, `''` meaning the index, or `None` when it has no such file."""
    spec = f'{revision}:{path}'
    exists = subprocess.run(('git', 'cat-file', '-e', spec), capture_output=True).returncode == 0
    return read_objects([spec])[0] if exists else None


def held(read: Callable[[str], str | None], baselines: dict[str, str]) -> dict[str, set[str] | None]:
    """The entries each baseline holds, read from the file `baselines` maps it to; `None` where it assigns none."""
    found: dict[str, set[str] | None] = {}
    for name, path in baselines.items():
        try:
            found[name] = entries(read(path), name)
        except InvalidBaseline as error:
            raise InvalidBaseline(f'{path}: {error}') from None
    return found


def added(bases: Sequence[str], read: Callable[[str], str | None]) -> list[tuple[str, str, str]]:
    """Each `(path, name, entry)` that `read` adds to the baselines all of `bases` hold together."""
    try:
        registries = [registry(show(base, CHECKER)) for base in bases]
        baselines = registry(read(CHECKER))
    except InvalidBaseline as error:
        raise InvalidBaseline(f'{CHECKER}: {error}') from None
    after = held(read, baselines)
    if unassigned := sorted(name for name, found in after.items() if found is None):
        names = ', '.join(f'`{name}` from `{baselines[name]}`' for name in unassigned)
        raise InvalidBaseline(f'{CHECKER}: `BASELINES` reads {names}, which assigns no such baseline')
    if not any(registries):
        return []
    before: dict[str, set[str]] = {}
    for base, registered in zip(bases, registries):
        for name, found in held(partial(show, base), registered).items():
            before.setdefault(name, set()).update(found or ())
    # Otherwise a change could move a baseline out of the check's sight and grow it there.
    if dropped := sorted(name for name, found in before.items() if found and name not in baselines):
        names = ', '.join(f'`{name}`' for name in dropped)
        raise InvalidBaseline(f'{CHECKER}: `BASELINES` drops {names}, which the base holds entries in: empty it first')
    return [
        (baselines[name], name, entry)
        for name, found in sorted(after.items())
        for entry in sorted((found or set()) - before.get(name, set()))
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check and return the exit code."""
    options = parse_args(argv, 'Check that no change adds to a test gate baseline.')
    # During a merge, an entry either parent holds is not one the merge adds.
    theirs = merge_head() if options.staged else None
    bases = [options.base, *([theirs] if theirs else [])]
    read: Callable[[str], str | None] = (
        partial(show, '')
        if options.staged
        else lambda path: Path(path).read_text(encoding='utf-8') if Path(path).exists() else None
    )
    try:
        found = added(bases, read)
    except InvalidBaseline as error:
        print(f'::error::{escape(str(error))}' if options.github else str(error))
        return 1
    for path, name, entry in found:
        message = f'`{name}` gained `{entry}`: fix the violation instead; a baseline only shrinks'
        print(f'::error file={escape(path, prop=True)}::{escape(message)}' if options.github else f'{path}: {message}')
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main())
