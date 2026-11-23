"""Every tool and scan that names its own package roots covers every package the workspace ships, or says why not.

A workspace member added without updating a tool's scope is outside that tool's gate while CI stays green: the
coverage gate passes on a package it never measures (https://github.com/pydantic/pydantic-ai/pull/9654), and
blocking calls go undetected in a package BlockBuster doesn't scan (https://github.com/pydantic/pydantic-ai/pull/12740).
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable, Iterable
from fnmatch import fnmatchcase
from pathlib import PurePosixPath

from ._source_roots import REPO_ROOT, ShippedPackage, shipped_packages
from .conftest import BLOCKBUSTER_PACKAGES
from .manifests import Exempt, assert_classified

_TOOL = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['tool']


def _matches_a_glob(globs: Iterable[str]) -> Callable[[ShippedPackage], bool]:
    # A `**/` in these tools' globs also matches no directory, where `fnmatch` needs at least one.
    patterns = [pattern for glob in globs for pattern in (glob, glob.replace('/**/', '/'))]
    return lambda package: any(fnmatchcase(f'{package.path}/__init__.py', pattern) for pattern in patterns)


def _under_a_root(roots: Iterable[str]) -> Callable[[ShippedPackage], bool]:
    return lambda package: any(package.path.is_relative_to(PurePosixPath(root)) for root in roots)


SCOPES: dict[str, Callable[[ShippedPackage], bool]] = {
    'ruff': _matches_a_glob(_TOOL['ruff']['include']),
    'pyright': _under_a_root(_TOOL['pyright']['include']),
    'coverage': _matches_a_glob(_TOOL['coverage']['run']['include']),
    'complexipy': _under_a_root(_TOOL['complexipy']['paths']),
    'blockbuster': lambda package: package.name in BLOCKBUSTER_PACKAGES,
}
"""Whether each tool's configured scope reaches a shipped package."""

_LAUNCHER = 'a launcher of a few lines over `pydantic_ai._cli`, which this tool covers'
_SHIM = 'the deprecated `pydantic-clai2` alias, which re-exports `pydantic_ai_pax` modules this tool covers'
_EXAMPLES = 'example applications, not library code; `tests/import_examples.py` imports each of them'

EXEMPT: dict[tuple[str, str], Exempt] = {
    ('coverage', 'clai'): Exempt(_LAUNCHER),
    ('coverage', 'pydantic_ai_examples'): Exempt(_EXAMPLES),
    ('coverage', 'pydantic_ai_pax'): Exempt(
        'left outside the 100% gate since https://github.com/pydantic/pydantic-ai/pull/9654'
    ),
    ('coverage', 'pydantic_clai2'): Exempt(_SHIM),
    ('complexipy', 'clai'): Exempt(_LAUNCHER),
    ('complexipy', 'pydantic_ai_examples'): Exempt(_EXAMPLES),
    ('complexipy', 'pydantic_clai2'): Exempt(_SHIM),
    ('blockbuster', 'pydantic_ai_examples'): Exempt(_EXAMPLES),
    ('blockbuster', 'pydantic_clai2'): Exempt(_SHIM),
}
"""The shipped packages a tool leaves out on purpose, keyed by tool and import name."""


def test_every_tool_scope_covers_every_shipped_package():
    uncovered = [
        (tool, package.name) for tool, covers in SCOPES.items() for package in shipped_packages() if not covers(package)
    ]
    assert_classified(uncovered, EXEMPT, complete_walk=True, table_name='EXEMPT')


def test_every_shipped_package_is_a_source_directory():
    assert [package.path for package in shipped_packages() if not (REPO_ROOT / package.path).is_dir()] == []
