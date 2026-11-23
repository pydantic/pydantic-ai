"""The base-ref comparison that the `scripts/check_*.py` ratchets share.

A check reads each changed `.py` file under its roots on both sides of a change and compares what it measures, so
a rule ratchets with no committed baseline: whatever the base already has is allowed, and the change may not add to
it. `--staged` compares the index with `HEAD`, for pre-commit; `--base REF` compares the working tree, untracked
files included, with `REF`, for CI. `count_growth` and `ratchet_growth` serve a check that counts findings per file.
"""

from __future__ import annotations

import argparse
import subprocess
from collections import Counter
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
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


@dataclass
class Finding:
    line: int
    text: str
    """The source line it is on, stripped, which pairs it across the two sides of a change and between files."""


@dataclass
class Growth:
    path: str
    rule: str
    old: int
    new: int
    added: list[Finding]
    """The findings the change added to this file, without those that moved here from another changed file."""


Scan = Callable[[str | None], Mapping[str, list[Finding]]]


def _texts(findings: list[Finding]) -> Counter[str]:
    return Counter(f.text for f in findings)


def count_growth(changes: Sequence[Change], scan: Scan) -> list[Growth]:
    """Each rule whose count in a file a change raised, less the findings that moved in from another changed file.

    A finding moved when the change removed a line with the same text from another changed file.
    """
    scans = [(c, scan(c.old_source), scan(c.new_source)) for c in changes]
    moves: dict[str, Counter[str]] = {}
    for _, old, new in scans:
        for rule, found in old.items():
            moves.setdefault(rule, Counter()).update(_texts(found) - _texts(new.get(rule, [])))
    growth: list[Growth] = []
    for change, old, new in scans:
        for rule, found in new.items():
            was = old.get(rule, [])
            fresh = _texts(found) - _texts(was)
            moved_in = moves.get(rule, Counter[str]())
            added: list[Finding] = []
            moved = 0
            for f in found:
                if not fresh[f.text]:
                    continue
                fresh[f.text] -= 1
                if moved_in[f.text]:
                    moved_in[f.text] -= 1
                    moved += 1
                else:
                    added.append(f)
            if len(found) - len(was) - moved > 0:
                growth.append(Growth(change.new_path or '', rule, len(was), len(found), added))
    return growth


def ratchet_growth(options: Options, roots: Sequence[str], scan: Scan) -> list[Growth]:
    """What `count_growth` finds for this run's change; during a merge, only what is worse than in both parents."""
    growth = count_growth(read_changes(options, roots), scan)
    theirs = merge_head() if options.staged else None
    if theirs:
        their_growth = count_growth(read_changes(replace(options, base=theirs), roots), scan)
        growth = worse_than_both(growth, their_growth, key=lambda g: (g.path, g.rule), before=lambda g: g.old)
    return growth


def print_growth(growth: Sequence[Growth], advice: Mapping[str, str], title: str, guide: str, github: bool) -> None:
    """List each file whose count grew with the findings the change added, then what to do about each rule."""
    for g in growth:
        print(f'{g.path}: {g.rule} {g.old} -> {g.new}')
        for f in g.added:
            print(f'    {g.path}:{f.line}: {f.text}')
            if github:
                heading = escape(f'{title}: {g.rule} {g.old} -> {g.new}', prop=True)
                print(
                    f'::error file={escape(g.path, prop=True)},line={f.line},title={heading}::{escape(advice[g.rule])}'
                )
    if growth:
        print()
        for rule in sorted({g.rule for g in growth}):
            print(f'{rule}: {advice[rule]}')
        print(f'See {guide}.')
