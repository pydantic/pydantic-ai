#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
from typing import TypeAlias

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


TomlValue: TypeAlias = bool | int | float | str | list['TomlValue'] | dict[str, 'TomlValue']

_ALLOWED_EXCLUDE_NEWER_PACKAGES = frozenset({'pydantic-harness'})


def _load_lockfile(path: Path) -> dict[str, TomlValue] | None:
    if tomllib is None:
        return None

    with path.open('rb') as lockfile:
        return tomllib.load(lockfile)


def _strip_toml_key_quotes(key: str) -> str:
    key = key.strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in {'"', "'"}:
        return key[1:-1]
    return key


def _find_exclude_newer_packages(path: Path) -> set[str]:
    packages: set[str] = set()
    in_exclude_newer_package = False

    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue

        if line.startswith('['):
            in_exclude_newer_package = line == '[options.exclude-newer-package]'
            continue

        if in_exclude_newer_package and '=' in line:
            package, _, _value = line.partition('=')
            packages.add(_strip_toml_key_quotes(package))

    return packages


def _get_exclude_newer_packages(lockfile_data: dict[str, TomlValue] | None) -> set[str]:
    if lockfile_data is None:
        return set()

    options = lockfile_data.get('options')
    if not isinstance(options, dict):
        return set()

    exclude_newer_package = options.get('exclude-newer-package')
    if exclude_newer_package is None:
        return set()

    if isinstance(exclude_newer_package, dict):
        return set(exclude_newer_package)

    print('Expected `uv.lock` `options.exclude-newer-package` to be a table.', file=sys.stderr)
    sys.exit(1)


def main() -> int:
    """Check that `uv.lock` does not contain unexpected package-specific cooldown exclusions."""
    lockfile_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('uv.lock')
    exclude_newer_packages = _get_exclude_newer_packages(_load_lockfile(lockfile_path))
    exclude_newer_packages.update(_find_exclude_newer_packages(lockfile_path))
    unexpected_packages = sorted(exclude_newer_packages - _ALLOWED_EXCLUDE_NEWER_PACKAGES)

    if not unexpected_packages:
        return 0

    print(f'{lockfile_path} contains unexpected `exclude-newer-package` entries:', file=sys.stderr)
    for package in unexpected_packages:
        print(f'  - {package}', file=sys.stderr)
    print(file=sys.stderr)
    print('Package-specific `exclude-newer` entries can bypass dependency cooldown checks.', file=sys.stderr)
    print(
        'Remove these entries from `uv.lock`, or update the allowlist in '
        '`scripts/check_uv_lock_exclusions.py` if the exception is intentional.',
        file=sys.stderr,
    )
    return 1


if __name__ == '__main__':
    sys.exit(main())
