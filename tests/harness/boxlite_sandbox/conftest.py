"""Shared fixtures for BoxLiteSandbox tests."""

from __future__ import annotations

import importlib.util
import os
import types
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

# BoxLite only has wheels for macOS on Apple silicon and Linux x86_64/aarch64, up to Python 3.14.
_HAS_BOXLITE = importlib.util.find_spec('boxlite') is not None
collect_ignore = [] if _HAS_BOXLITE else ['test_conformance.py', 'test_boxlite_sandbox.py', 'test_boxlite_live.py']

if TYPE_CHECKING or _HAS_BOXLITE:  # pragma: no branch - the SDK-installed and other jobs take opposite branches
    from .fake_boxlite import FakeBoxLite


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Gate the live tier, which boots real VMs: it needs a host that can run them."""
    if item.get_closest_marker('boxlite_live') is None:
        return
    if os.getenv('PYDANTIC_AI_HARNESS_BOXLITE_LIVE') == '1':  # pragma: no cover - live tier
        return
    pytest.skip('requires PYDANTIC_AI_HARNESS_BOXLITE_LIVE=1 on a host that can run BoxLite VMs')


_BOXLITE_MODULES = ('pydantic_ai_harness.boxlite_sandbox._backend',)
"""The modules that call the BoxLite SDK, each patched to the fake."""


class _PoisonedBoxLite(types.ModuleType):
    """A `boxlite` stand-in that fails loudly on any attribute access.

    Real BoxLite is installed in the dev venv for the live tier, so a unit test that forgets the
    `fake_boxlite` fixture would otherwise boot real VMs, or fail on a host that can't.
    """

    def __getattr__(self, name: str) -> object:  # pragma: no cover - tripwire, hit only by a misbehaving test
        raise AssertionError(
            'A boxlite_sandbox unit test touched the real `boxlite` package. '
            'Use the `fake_boxlite` fixture, or mark the test `boxlite_live`.'
        )


@pytest.fixture(autouse=True)
def _no_real_boxlite(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Poison `boxlite` for every test here except the opt-in live tier."""
    if 'boxlite_live' in request.keywords:  # pragma: no cover - live tier runs without coverage
        yield
        return
    poisoned = _PoisonedBoxLite('boxlite')
    for module in _BOXLITE_MODULES:
        monkeypatch.setattr(f'{module}.boxlite', poisoned)
    yield


if _HAS_BOXLITE:  # pragma: no branch - the fixture cannot be defined without its SDK-backed fake

    @pytest.fixture
    def fake_boxlite(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[FakeBoxLite]:
        """Inject a fake `boxlite` module, whose boxes run commands on the host, and yield its control surface."""
        # Resolved so the directory `pwd -P` reports matches on hosts whose temp directory is a symlink.
        control = FakeBoxLite(tmp_path.resolve())
        for module in _BOXLITE_MODULES:
            monkeypatch.setattr(f'{module}.boxlite', control.module)
        yield control
        control.close()
