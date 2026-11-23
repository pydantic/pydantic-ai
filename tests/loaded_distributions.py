"""Check the third-party distributions an entry point loads in a fresh interpreter against a checked-in allowlist.

A guard that names the modules an entry point must not load passes when a change adds a new heavy dependency, so
each startup check lists what its entry point may load instead. A distribution the entry point newly loads fails
until its import is deferred or it is added with a reason, and an entry the entry point no longer loads fails as
stale. The lists name distributions rather than modules, so a dependency that reorganizes its modules does not
churn them.

The child interpreter runs `SNAPSHOT_PRELUDE` before the code under test and calls `snapshot(entry_point)` where
the check applies; `snapshots` reads what it printed, and `assert_loads_only` compares one snapshot with its list.
"""

from __future__ import annotations

import functools
import importlib.metadata
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter

from tests.manifests import assert_classified

_SNAPSHOT_MARKER = 'LOADED_MODULES '

SNAPSHOT_PRELUDE = f"""
import sys as _sys
_startup_modules = set(_sys.modules)
def snapshot(entry_point):
    import json
    loaded = {{
        name: getattr(module, '__file__', None)
        for name, module in list(_sys.modules.items())
        if name not in _startup_modules and name.partition('.')[0] not in _sys.stdlib_module_names
    }}
    print({_SNAPSHOT_MARKER!r} + json.dumps([entry_point, loaded]), flush=True)
"""
"""Child code defining `snapshot(entry_point)`, which prints the non-standard-library modules loaded since startup.

Modules loaded before the code under test, by `.pth` files or coverage, belong to the environment and are left out.
"""

_SNAPSHOT = TypeAdapter(tuple[str, dict[str, str | None]])


@dataclass(frozen=True)
class Loads:
    """A distribution the entry point loads wherever it is installed. Installed but not loaded, it is stale."""

    reason: str


@dataclass(frozen=True)
class MayLoad:
    """A distribution only some installs load, such as one an older version of a dependency imports. Never stale."""

    reason: str


Allowlist = Mapping[str, Loads | MayLoad]
"""Distributions an entry point may load, keyed by normalized name (`pydantic-core`, not `pydantic_core`)."""


def child_env(**variables: str) -> dict[str, str]:
    """This process's environment plus `variables`, without `COVERAGE_*`, so coverage does not instrument the child."""
    env = {key: value for key, value in os.environ.items() if not key.startswith('COVERAGE_')}
    return env | variables


def snapshots(stdout: str) -> dict[str, dict[str, str | None]]:
    """The modules each `snapshot(entry_point)` call printed, by entry point: module name to file."""
    return dict(
        _SNAPSHOT.validate_json(line.removeprefix(_SNAPSHOT_MARKER))
        for line in stdout.splitlines()
        if line.startswith(_SNAPSHOT_MARKER)
    )


def assert_loads_only(entry_point: str, modules: Mapping[str, str | None], allowlist: Allowlist, *, table_name: str):
    """Fail if the snapshot's distributions are not those `allowlist` names for the installed environment."""
    loaded = loaded_distributions(modules)
    if new := sorted(loaded - allowlist.keys()):
        raise AssertionError(
            f'{entry_point} now loads {new}. Defer the import that pulls them in to where it is first used, '
            f'or add each to `{table_name}` with the reason it has to load here.'
        )
    installed = _installed_distributions()
    expected = {
        name: entry
        for name, entry in allowlist.items()
        if name in loaded or (isinstance(entry, Loads) and name in installed)
    }
    assert_classified(loaded, expected, complete_walk=True, table_name=table_name)


def loaded_distributions(modules: Mapping[str, str | None]) -> set[str]:
    """The third-party distributions that installed `modules`, a snapshot's module names and files.

    Modules no distribution lists, such as the workspace's editable packages, are skipped.
    """
    owners = _module_owners()
    loaded: set[str] = set()
    for name, file in modules.items():
        if file is None:
            # A namespace package runs no code: the distributions that share it are credited through their modules.
            continue
        candidates = owners.get(name.partition('.')[0], ())
        if len(candidates) > 1:
            # Several distributions share the namespace (`opentelemetry`, `google`): credit the one that installed
            # this module's file.
            candidates = [c for c in candidates if os.path.realpath(file) in _files(c)]
        loaded.update(candidates)
    return loaded - _workspace_distributions()


def _normalize(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name).lower()


@functools.cache
def _module_owners() -> dict[str, list[str]]:
    return {
        module: sorted({_normalize(name) for name in names})
        for module, names in importlib.metadata.packages_distributions().items()
    }


@functools.cache
def _installed_distributions() -> frozenset[str]:
    # Only distributions that install a module: one that installs none, like a metapackage, is never loaded.
    return frozenset(name for names in _module_owners().values() for name in names)


@functools.cache
def _files(distribution: str) -> frozenset[str]:
    installed = importlib.metadata.distribution(distribution)
    return frozenset(os.path.realpath(str(installed.locate_file(file))) for file in installed.files or ())


@functools.cache
def _workspace_distributions() -> frozenset[str]:
    root = Path(__file__).parents[1]
    pyproject = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))
    members = [root / member for member in pyproject['tool']['uv']['workspace']['members']]
    names = [
        tomllib.loads((path / 'pyproject.toml').read_text(encoding='utf-8'))['project']['name'] for path in members
    ]
    return frozenset(_normalize(name) for name in [pyproject['project']['name'], *names])
