"""List or remove the `[tool.uv.exclude-newer-package]` overrides that the global `exclude-newer` window has overtaken.

An override dates the newest release uv may lock for one package. Once that date is older than now minus the
relative `[tool.uv] exclude-newer` span, the override no longer lets a newer release in: it holds the package below
the window every other package gets. Only the value counts, never the comment beside it, and a `false` override never
expires.

- `list` prints the expired package names, one per line.
- `remove` also deletes each expired entry from the file, together with the comment block directly above it once no
  entry under that block remains. Every other byte of the file stays as it was.
"""

from __future__ import annotations

import argparse
import re
import tomllib
from collections.abc import Collection, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

TABLE = '[tool.uv.exclude-newer-package]'
SPAN = re.compile(r'(\d+) ?(hour|day|week)s?')
SPAN_UNITS = {'hour': timedelta(hours=1), 'day': timedelta(days=1), 'week': timedelta(weeks=1)}
HEADER = re.compile(rf'^{re.escape(TABLE)}.*\n', re.MULTILINE)
NEXT_TABLE = re.compile(r'^[ \t]*\[', re.MULTILINE)
# A comment block and the entry lines directly under it, up to the next comment or blank line.
GROUP = re.compile(r'^(?P<comments>(?:[ \t]*#.*\n)*)(?P<entries>(?:[ \t]*[^\s#\[].*(?:\n|\Z))+)', re.MULTILINE)
KEY = re.compile(r'["\']?([A-Za-z0-9._-]+)["\']?\s*=')


def timestamp(value: str) -> datetime:
    """Parse an ISO 8601 timestamp, reading one without an offset as UTC."""
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def expired(text: str, now: datetime) -> list[str]:
    """Return the packages in the `pyproject.toml` text whose override is dated before `now` minus the global span."""
    uv = tomllib.loads(text)['tool']['uv']
    span: object = uv['exclude-newer']
    overrides: dict[str, object] = uv.get('exclude-newer-package', {})
    if not isinstance(span, str) or (match := SPAN.fullmatch(span)) is None:
        raise ValueError(f'`[tool.uv] exclude-newer` is {span!r}, not a relative span such as "7 days".')
    cutoff = now - int(match[1]) * SPAN_UNITS[match[2]]
    names: list[str] = []
    for name, value in overrides.items():
        if not isinstance(value, str):
            continue
        try:
            dated = timestamp(value)
        except ValueError:
            # A relative span moves with the clock, so it never falls behind; `uv lock` rejects any other value.
            continue
        if dated < cutoff:
            names.append(name)
    return names


def remove(text: str, names: Collection[str]) -> str:
    """Delete the named entries from the override table, and each comment block left with no entry under it."""
    header = HEADER.search(text)
    start = header.end() if header else len(text)
    end = next_table.start() if (next_table := NEXT_TABLE.search(text, start)) else len(text)
    pieces = [text[:start]]
    position = start
    found: set[str] = set()
    for group in GROUP.finditer(text, start, end):
        kept: list[str] = []
        for line in group['entries'].splitlines(keepends=True):
            key = KEY.match(line.strip())
            if key and key[1] in names:
                found.add(key[1])
            else:
                kept.append(line)
        pieces += [text[position : group.start()], group['comments'] if kept else '', *kept]
        position = group.end()
    if missing := set(names) - found:
        raise ValueError(f'No `{TABLE}` entry line for {sorted(missing)}.')
    return ''.join([*pieces, text[position:]])


def main(argv: Sequence[str] | None = None) -> None:
    """Print the expired overrides, and with `remove`, delete them from the file."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['list', 'remove'])
    parser.add_argument('pyproject', nargs='?', type=Path, default=Path('pyproject.toml'))
    parser.add_argument('--now', type=timestamp, default=datetime.now(UTC), help='ISO 8601 timestamp (default: now)')
    args = parser.parse_args(argv)
    command: str = args.command
    path: Path = args.pyproject
    now: datetime = args.now
    # Bytes, so that line endings round-trip unchanged.
    text = path.read_bytes().decode('utf-8')
    names = expired(text, now)
    if command == 'remove' and names:
        path.write_bytes(remove(text, names).encode('utf-8'))
    for name in names:
        print(name)


if __name__ == '__main__':
    main()
