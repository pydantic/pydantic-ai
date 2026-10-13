"""Shared fixtures for SmolSandbox tests."""

from __future__ import annotations

import importlib.util
import os
import types
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

# smolmachines only has wheels for macOS on Apple silicon and Linux x86_64/aarch64.
_HAS_SMOL = importlib.util.find_spec('smol') is not None
collect_ignore = [] if _HAS_SMOL else ['test_conformance.py', 'test_smol_sandbox.py']

if TYPE_CHECKING or _HAS_SMOL:  # pragma: no branch - the SDK-installed and other jobs take opposite branches
    from .fake_smol import FakeSmol


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Gate the live tier, which boots real VMs: it needs a host that can run them."""
    if item.get_closest_marker('smol_live') is None:
        return
    if os.getenv('PYDANTIC_AI_HARNESS_SMOL_LIVE') == '1':  # pragma: no cover - live tier
        return
    pytest.skip('requires PYDANTIC_AI_HARNESS_SMOL_LIVE=1 on a host that can run smolvm VMs')


_SMOL_MODULES = ('pydantic_ai_harness.smol_sandbox._backend', 'pydantic_ai_harness.smol_sandbox._capability')
"""The modules that call the smol SDK, each patched to the fake."""


class _PoisonedSmol(types.ModuleType):
    """A `smol` stand-in that fails loudly on any attribute access.

    Real smol is installed in the dev venv for the live tier, so a unit test that forgets the
    `fake_smol` fixture would otherwise boot real VMs, or fail on a host that can't.
    """

    def __getattr__(self, name: str) -> object:  # pragma: no cover - tripwire, hit only by a misbehaving test
        raise AssertionError(
            'A smol_sandbox unit test touched the real `smol` package. '
            'Use the `fake_smol` fixture, or mark the test `smol_live`.'
        )


@pytest.fixture(autouse=True)
def _no_real_smol(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Poison `smol` for every test here except the opt-in live tier."""
    if 'smol_live' in request.keywords:  # pragma: no cover - live tier runs without coverage
        yield
        return
    poisoned = _PoisonedSmol('smol')
    for module in _SMOL_MODULES:
        monkeypatch.setattr(f'{module}.smol', poisoned)
    yield


if _HAS_SMOL:  # pragma: no branch - the fixture cannot be defined without its SDK-backed fake

    @pytest.fixture
    def fake_smol(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[FakeSmol]:
        """Inject a fake `smol` module, whose VMs run in `tmp_path`, and yield its control surface."""
        # Resolved so the directory `pwd -P` reports matches on hosts whose temp directory is a symlink.
        control = FakeSmol(tmp_path.resolve())
        for module in _SMOL_MODULES:
            monkeypatch.setattr(f'{module}.smol', control.module)
        yield control
