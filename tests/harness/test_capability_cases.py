"""Every Pydantic AI Harness capability has a case in `tests/harness/capability_cases.py`."""

from __future__ import annotations

from .. import capability_cases as core
from ..append_only import discover_capabilities
from ..capability_cases import Unbuildable, shipped_capability_types
from ..manifests import assert_classified
from .capability_cases import CASES, HARNESS_CAPABILITY_PACKAGES


def test_every_capability_has_a_case() -> None:
    """A new capability must say what two of it mean before it can ship, and how to build it if its defaults can't.

    Without a case the answer defaults to whatever `AbstractCapability` does, which is the one outcome nobody chose.
    The required set is every public capability the source declares, which reading source finds without any
    optional dependency, plus every class, public or not, that importing the package finds. An entry is reported
    stale only when every module imported: an optional group that is not installed makes its capabilities look
    deleted, and the slim CI lane installs none of them.
    """
    found, skipped = shipped_capability_types(HARNESS_CAPABILITY_PACKAGES)
    assert_classified(
        discover_capabilities(HARNESS_CAPABILITY_PACKAGES).keys() | found.keys(),
        CASES,
        complete_walk=not skipped,
        table_name='CASES',
    )


def test_only_the_listed_cases_are_unbuildable() -> None:
    """An `Unbuildable` case runs in none of the suites that build capabilities, so each one is a reviewed choice."""
    unbuildable = sorted(name for name, case in CASES.items() if isinstance(case.build, Unbuildable))
    assert unbuildable == [
        'AWSLambdaDurability',
        'AbsurdDurability',
        'DynamicWorkflow',
        'GoogleWorkspace',
        'ModelRouter',
        'PGTaskDurability',
        'RenderWorkflows',
        'SSHWorkspace',
    ]


def test_core_and_harness_cases_have_distinct_names() -> None:
    """Cases are keyed by class name, so a harness capability named like a core one could not be read beside it."""
    assert not (CASES.keys() & core.CASES.keys())
