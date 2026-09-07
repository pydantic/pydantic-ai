"""Reject asyncio usage outside the reviewed migration inventory."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

from pydantic import TypeAdapter
from typing_extensions import TypedDict

from asyncio_policy import asyncio_sites, config_suppressions


class ExceptionEntry(TypedDict):
    """A reviewed reason and exact inventory of permitted asyncio sites."""

    reason: str
    sites: dict[str, int]


def main() -> int:
    """Check tracked and unignored Python files against the exception inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    root: Path = args.root
    inventory = TypeAdapter(dict[str, ExceptionEntry]).validate_json(
        (root / 'scripts/asyncio_exceptions.json').read_text(encoding='utf-8')
    )
    result = subprocess.run(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard', '--', '*.py', '*.pyi', '*.toml'],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    failures: list[str] = []
    remaining = set(inventory)
    for name in sorted(set(result.stdout.rstrip('\0').split('\0')) - {''}):
        path = root / name
        if not path.is_file():
            continue
        if path.suffix == '.toml':
            if path.name not in ('pyproject.toml', 'ruff.toml', '.ruff.toml'):
                continue
            actual = config_suppressions(path)
        else:
            actual = asyncio_sites(path.read_text(encoding='utf-8'))
        expected = inventory.get(name)
        if expected is None:
            if actual:
                failures.append(f'{name}: unapproved asyncio usage or lint suppression; use AnyIO')
            continue
        remaining.remove(name)
        if not expected['reason'].strip():
            failures.append(f'{name}: the asyncio exception needs a reason')
        approved = Counter(expected['sites'])
        for site, count in sorted((actual - approved).items()):
            failures.append(f'{name}: {count} unapproved occurrence(s) of {site}')
        for site, count in sorted((approved - actual).items()):
            failures.append(f'{name}: remove {count} stale occurrence(s) of {site} from the inventory')
        if not actual:
            failures.append(f'{name}: remove the empty asyncio exception')
    for name in sorted(remaining):
        failures.append(f'{name}: remove the exception for a missing file')
    for failure in failures:
        sys.stderr.write(f'{failure}\n')
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
