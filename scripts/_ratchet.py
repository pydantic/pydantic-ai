"""The base-ref comparison that `scripts/check_complexity.py` and `scripts/check_test_waits.py` share.

A check reads each changed `.py` file under its roots on both sides of a change and compares what it measures, so
a rule ratchets with no committed baseline: whatever the base already has is allowed, and the change may not add to
it. `--staged` compares the index with `HEAD`, for pre-commit; `--base REF` compares the working tree, untracked
files included, with `REF`, for CI.
"""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

T = TypeVar('T')


@dataclass
class Options:
    base: str
    staged: bool
    github: bool


@dataclass
class Change:
    old_path: str | None
    new_path: str | None
    old_source: str | None = None
    new_source: str | None = None


def parse_args(argv: Sequence[str] | None, description: str) -> Options:
    parser = argparse.ArgumentParser(description=description)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--staged', action='store_true', help='compare the index with HEAD')
    source.add_argument('--base', help='compare the working tree with this revision')
    parser.add_argument('--format', choices=('text', 'github'), default='text')
    args = parser.parse_args(argv)
    staged: bool = args.staged
    base: str | None = args.base
    output_format: str = args.format
    # The group is required, so `base` is set whenever `--staged` is not.
    return Options(base or 'HEAD', staged, output_format == 'github')


def git(*args: str, stdin: bytes | None = None) -> bytes:
    return subprocess.run(('git', *args), input=stdin, capture_output=True, check=True).stdout


def read_objects(specs: Sequence[str]) -> list[str]:
    """Read `rev:path` blobs through one `git cat-file --batch`."""
    if not specs:
        return []
    out = git('cat-file', '--batch', stdin=''.join(f'{spec}\n' for spec in specs).encode())
    blobs: list[str] = []
    pos = 0
    for _ in specs:
        end = out.index(b'\n', pos)
        size = int(out[pos:end].split()[2])
        blobs.append(out[end + 1 : end + 1 + size].decode('utf-8', errors='replace'))
        pos = end + 1 + size + 1
    return blobs


def changed_files(revisions: Sequence[str], roots: Sequence[str]) -> list[Change]:
    """Every changed `.py` file under `roots`, with renames paired."""
    fields = git('diff', '--name-status', '-z', '-M', *revisions, '--', *roots).decode().split('\0')
    changes: list[Change] = []
    i = 0
    while i < len(fields) - 1:
        status, path = fields[i], fields[i + 1]
        if status.startswith('R'):
            changes.append(Change(path, fields[i + 2]))
            i += 3
            continue
        changes.append(Change(None if status == 'A' else path, None if status == 'D' else path))
        i += 2
    return [c for c in changes if (c.new_path or c.old_path or '').endswith('.py')]


def untracked_files(roots: Sequence[str]) -> list[Change]:
    paths = git('ls-files', '--others', '--exclude-standard', '-z', '--', *roots).decode().split('\0')
    return [Change(None, path) for path in paths if path.endswith('.py')]


def load(changes: list[Change], base: str, staged: bool) -> None:
    """Read both sides of each change, the new side from the index when `staged`, else from the working tree."""
    olds = iter(read_objects([f'{base}:{c.old_path}' for c in changes if c.old_path]))
    news = iter(read_objects([f':{c.new_path}' for c in changes if c.new_path] if staged else []))
    for c in changes:
        c.old_source = next(olds) if c.old_path else None
        if c.new_path:
            c.new_source = next(news) if staged else Path(c.new_path).read_text(encoding='utf-8')


def read_changes(options: Options, roots: Sequence[str]) -> list[Change]:
    """Every changed `.py` file under `roots`, read on both sides."""
    if options.staged:
        found = changed_files(['--cached', options.base], roots)
    else:
        found = changed_files([options.base], roots) + untracked_files(roots)
    load(found, options.base, options.staged)
    return found


def merge_head() -> str | None:
    """The commit a merge in progress brings in, or `None` outside a merge."""
    found = subprocess.run(('git', 'rev-parse', '-q', '--verify', 'MERGE_HEAD'), capture_output=True, text=True)
    return found.stdout.strip() or None


def worse_than_both(
    ours: list[T], theirs: list[T], key: Callable[[T], Hashable], before: Callable[[T], int | None]
) -> list[T]:
    """The findings against `HEAD` that `MERGE_HEAD` also has, each reported against the parent it was worse in.

    Compared with `HEAD` alone, a merge would be blamed for everything the merged branch made worse.
    """
    by_key = {key(finding): finding for finding in theirs}
    return [
        max(finding, by_key[key(finding)], key=lambda f: before(f) or 0) for finding in ours if key(finding) in by_key
    ]


def escape(value: str, prop: bool = False) -> str:
    """Escape a GitHub workflow command's message, or with `prop` one of its properties."""
    value = value.replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')
    return value.replace(':', '%3A').replace(',', '%2C') if prop else value
