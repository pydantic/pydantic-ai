"""What the suites that run the `CONTRACTS` share: every case, which ones a contract applies to, and how one holds.

`tests/harness/test_capability_contracts.py` runs most contracts; the durable ones run in
`tests/harness/test_durable_tool_io.py` (DBOS) and `tests/harness/test_durable_temporal_replay.py` (Temporal).
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from _pytest.mark import ParameterSet

from .. import capability_cases as core
from ..append_only import discover_capabilities
from ..capability_cases import CORE_CAPABILITY_PACKAGES, CapabilityCase, Contract
from .capability_cases import CASES as HARNESS_CASES, HARNESS_CAPABILITY_PACKAGES

CASES: dict[str, CapabilityCase] = {**core.CASES, **HARNESS_CASES}
"""Every core and Harness case, keyed by class name."""

PUBLIC = {**discover_capabilities(CORE_CAPABILITY_PACKAGES), **discover_capabilities(HARNESS_CAPABILITY_PACKAGES)}
"""The public capabilities, which the suites build; a private class only has to say what two of it mean."""


def contract_cases(contract: Contract) -> list[ParameterSet]:
    """The cases `contract` applies to, each whose calls rewrite history marked `moves_cache_prefix`."""
    return [
        pytest.param(name, marks=pytest.mark.moves_cache_prefix(reason=reason))
        if case.calls is not None and (reason := case.calls.moves_cache_prefix)
        else pytest.param(name)
        for name, case in sorted(CASES.items())
        if name in PUBLIC and contract.applies(name, case)
    ]


def assert_holds(contract: Contract, name: str, breaks: Sequence[str]) -> None:
    """`breaks` says how the case breaks `contract`: it must say nothing, unless `contract` exempts the case."""
    if exempt := contract.exempt.get(name):
        assert breaks, f'{name} now passes: remove its `{contract.name}` exemption ({exempt.reason})'
    else:
        assert not breaks, '; '.join(breaks)
