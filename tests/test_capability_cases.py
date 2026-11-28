"""Every core capability has a case in `tests/capability_cases.py`."""

from __future__ import annotations

import pytest

from .append_only import discover_capabilities
from .capability_cases import (
    CASES,
    CORE_CAPABILITY_PACKAGES,
    Unbuildable,
    import_package_modules,
    shipped_capability_types,
)
from .manifests import assert_classified


def test_every_capability_has_a_case() -> None:
    """A new capability must say what two of it mean before it can ship, and how to build it if its defaults can't.

    Without a case the answer defaults to whatever the base class does, which is the one outcome nobody chose. The
    required set is every public capability the source declares, which reading source finds without any optional
    dependency, plus every class, public or not, that importing the packages finds. An entry is reported stale only
    when every module imported: a missing optional group makes its capabilities look deleted.
    """
    found, skipped = shipped_capability_types(CORE_CAPABILITY_PACKAGES)
    assert_classified(
        discover_capabilities(CORE_CAPABILITY_PACKAGES).keys() | found.keys(),
        CASES,
        complete_walk=not skipped,
        table_name='CASES',
    )


def test_only_the_listed_cases_are_unbuildable() -> None:
    """An `Unbuildable` case runs in none of the suites that build capabilities, so each one is a reviewed choice."""
    unbuildable = sorted(name for name, case in CASES.items() if isinstance(case.build, Unbuildable))
    assert unbuildable == ['BatchModelRequests', 'NativeOrLocalTool', 'RequestInput', 'WindowsWorkspace']


def test_only_an_optional_package_may_be_missing() -> None:
    """The walk leaves out an optional package that isn't installed, but a required one has to be there."""
    assert list(import_package_modules((), optional=('pydantic_ai_not_installed',))) == []
    with pytest.raises(ModuleNotFoundError, match='pydantic_ai_not_installed'):
        list(import_package_modules(('pydantic_ai_not_installed',)))
