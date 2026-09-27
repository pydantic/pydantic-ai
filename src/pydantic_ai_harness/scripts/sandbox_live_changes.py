"""Decide which sandbox providers' live tests a change needs, and print them as a JSON list.

The live tests start real, billed sandboxes, so CI runs a provider's live tier only when a
change could break it:

- the provider's own package (its README included), docs page, or tests changed; the live
  tier runs every example on the docs page;
- the provider's SDK moved in `uv.lock`;
- code every provider runs under changed (`SHARED`), or the `pydantic-ai-slim` pin moved,
  which is how a change to the workspace protocol reaches the harness. Either one runs every
  provider.

Nothing else in `pyproject.toml` or `uv.lock` counts: an unrelated dependency bump runs no
live test. CI installs with `--locked`, so a `pyproject.toml` change that matters to a
provider always shows up as a moved lock entry.

A provider named in `PROVIDERS` follows one layout: package `pydantic_ai_harness/<name>_sandbox`,
docs page `docs/<name>-sandbox.md`, tests `tests/<name>_sandbox`, extra `<name>`, marker
`<name>_live`, and the environment variables `PYDANTIC_AI_HARNESS_<NAME>_LIVE` and
`<NAME>_REQUIRE_LIVE`. Adding a provider is one entry here. A provider whose package is not in the tree yet is never selected.

Usage:

    python scripts/sandbox_live_changes.py changed BASE_SHA HEAD_SHA
    python scripts/sandbox_live_changes.py all

Standard library only, so CI runs it without syncing the project.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

# Code every provider's live tests run through.
SHARED = (
    'pydantic_ai_harness/_workspace.py',
    'pydantic_ai_harness/_workspace_provider.py',
    'pydantic_ai_harness/_warn.py',
    'pydantic_ai_harness/shell/',
    'pydantic_ai_harness/filesystem/',
    'pydantic_ai_harness/coder/',
    'tests/conftest.py',
    'tests/_tool_calls.py',
    'tests/_docs_examples.py',
    '.github/workflows/sandbox-live.yml',
    'scripts/sandbox_live_changes.py',
)

# Lock entries every provider depends on.
SHARED_LOCK_PACKAGES = ('pydantic-ai-slim',)

# Provider name to the lock entries of its SDK.
PROVIDERS: dict[str, tuple[str, ...]] = {
    'modal': ('modal', 'grpclib', 'synchronicity'),
    'e2b': ('e2b', 'connectrpc'),
    'sprites': ('sprites-py', 'websockets'),
}

_ROOT = Path(__file__).resolve().parents[1]


def present_providers(root: Path = _ROOT) -> list[str]:
    """The providers whose package exists in the tree at `root`."""
    return [name for name in PROVIDERS if (root / 'pydantic_ai_harness' / f'{name}_sandbox').is_dir()]


def moved_lock_packages(base_lock: str, head_lock: str) -> set[str]:
    """The packages whose version or source differs between two `uv.lock` texts."""
    base, head = _lock_entries(base_lock), _lock_entries(head_lock)
    return {name for name in base.keys() | head.keys() if base.get(name) != head.get(name)}


def _lock_entries(text: str) -> dict[str, tuple[str, str]]:
    # uv writes `name`, `version` and `source` as the first lines of every `[[package]]`
    # table, one key per line, so reading those lines is enough and keeps this script
    # free of `tomllib`, which Python 3.10 lacks.
    entries: dict[str, tuple[str, str]] = {}
    for table in text.split('[[package]]\n')[1:]:
        keys = dict(
            line.split(' = ', 1)
            for line in table.splitlines()
            if line.startswith(('name = ', 'version = ', 'source = '))
        )
        if 'name' in keys:
            entries[keys['name'].strip('"')] = (keys.get('version', ''), keys.get('source', ''))
    return entries


def providers_for(
    changed_files: Iterable[str], base_lock: str, head_lock: str, present: Iterable[str] | None = None
) -> list[str]:
    """The providers a change needs live tests for, in `PROVIDERS` order.

    `present` limits the answer to providers in the tree; it defaults to all of `PROVIDERS`.
    """
    files = list(changed_files)
    moved = moved_lock_packages(base_lock, head_lock) if 'uv.lock' in files else set[str]()
    shared = any(_touches(path, SHARED) for path in files) or bool(moved & set(SHARED_LOCK_PACKAGES))

    selected: list[str] = []
    for name, packages in PROVIDERS.items():
        own = (f'pydantic_ai_harness/{name}_sandbox/', f'docs/{name}-sandbox.md', f'tests/{name}_sandbox/')
        # The E2B search smoke test lives with FileSystem, outside the provider test directory.
        if (
            shared
            or any(_touches(path, own) for path in files)
            or (name == 'e2b' and 'tests/filesystem/test_search_live.py' in files)
            or moved & set(packages)
        ):
            selected.append(name)
    allowed = set(PROVIDERS if present is None else present)
    return [name for name in selected if name in allowed]


def _touches(path: str, prefixes: Iterable[str]) -> bool:
    return any(path == prefix or (prefix.endswith('/') and path.startswith(prefix)) for prefix in prefixes)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(['git', *args], cwd=_ROOT, capture_output=True, text=True)


def _changed(base: str, head: str) -> list[str]:
    # A push that creates a branch reports an all-zero `before`, and a force-push can leave
    # the old head unreachable; neither gives a base to compare, so every provider runs.
    if set(base) <= {'0'} or _git('cat-file', '-e', f'{base}^{{commit}}').returncode != 0:
        return present_providers()
    diff = _git('diff', '--name-only', '--no-renames', base, head)
    if diff.returncode != 0:
        raise SystemExit(diff.stderr)
    base_lock = _git('show', f'{base}:uv.lock').stdout
    head_lock = _git('show', f'{head}:uv.lock').stdout
    return providers_for(diff.stdout.splitlines(), base_lock, head_lock, present_providers())


def main(argv: list[str]) -> None:
    """Print the providers for `changed BASE HEAD` or `all` as a JSON list."""
    if argv[:1] == ['all'] and len(argv) == 1:
        print(json.dumps(present_providers()))
    elif argv[:1] == ['changed'] and len(argv) == 3:
        print(json.dumps(_changed(argv[1], argv[2])))
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main(sys.argv[1:])
