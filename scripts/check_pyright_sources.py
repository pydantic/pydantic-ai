"""Fail when Pyright read fewer source files than the root `[tool.pyright]` configuration names.

Pyright drops files from its configured scope without a diagnostic: 1.1.414 started applying its default
excludes (`**/.*` among them) over explicit `include` entries, and the twelve `.github/scripts` files the root
configuration listed went unchecked with CI green (https://github.com/pydantic/pydantic-ai/pull/8732).

Usage: `check_pyright_sources.py <log>`, where `<log>` is the output of `make typecheck-pyright` run with
`PYRIGHT_STATS=1`, which makes the root run print `Found <n> source files`. The expected count is every tracked
`.py` and `.pyi` file under an `include` entry and under no `exclude` entry, with none of Pyright's own excludes
applied, so a file Pyright stops reading on its own lowers the count it reports below that.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

_FOUND = re.compile(r'^Found (\d+) source files?$', re.MULTILINE)


def configured_sources(pyproject: Path) -> list[str]:
    """The tracked source files the root Pyright configuration names."""
    pyright = tomllib.loads(pyproject.read_text(encoding='utf-8'))['tool']['pyright']
    include: list[str] = [entry.rstrip('/') for entry in pyright['include']]
    exclude: list[str] = [entry.rstrip('/') for entry in pyright.get('exclude', [])]
    tracked = subprocess.run(
        ['git', 'ls-files', '-z', '--', '*.py', '*.pyi'],
        capture_output=True,
        text=True,
        check=True,
        cwd=pyproject.parent,
    ).stdout.split('\0')

    def covered(path: str, entries: list[str]) -> bool:
        return any(path == entry or path.startswith(f'{entry}/') for entry in entries)

    return [path for path in tracked if path and covered(path, include) and not covered(path, exclude)]


def main(log: str) -> int:
    """Compare the count the root run printed to `log` with the configured sources, and return the exit code."""
    found = _FOUND.search(Path(log).read_text(encoding='utf-8'))
    if found is None:
        print(f'No `Found <n> source files` line in {log}: run `make typecheck-pyright` with `PYRIGHT_STATS=1`.')
        return 1
    expected = len(configured_sources(Path('pyproject.toml')))
    read = int(found.group(1))
    print(f'Pyright read {read} source files; the root `[tool.pyright]` configuration names {expected}.')
    if read < expected:
        print(
            f'Pyright skipped {expected - read} configured files without a diagnostic. Find them with '
            '`pyright --verbose`, then move them into a project Pyright reads or out of `include`.'
        )
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
