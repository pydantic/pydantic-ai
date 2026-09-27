"""Shared fixtures for ModalSandbox tests."""

from __future__ import annotations

import os
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pytest

from .fake_modal import FakeModal

# CI stops the live job after 15 minutes, before a `finally` can terminate what it started.
# Live tests create sandboxes with these limits so one left behind that way ends on its own.
LIVE_SANDBOX_TIMEOUT = 900
LIVE_IDLE_TIMEOUT = 300


class _PoisonedModal(types.ModuleType):
    """A `modal` stand-in that fails loudly on any attribute access.

    Real modal is installed in the dev venv for the live tier, so a unit test that
    forgets the `fake_modal` fixture would otherwise reach the real SDK and, with
    developer credentials configured, create real billed sandboxes. This is the
    `ALLOW_MODEL_REQUESTS = False` of the Modal seam.
    """

    def __getattr__(self, name: str) -> object:  # pragma: no cover - tripwire, hit only by a misbehaving test
        raise AssertionError(
            'A modal_sandbox unit test touched the real `modal` package. '
            'Use the `fake_modal` fixture, or mark the test `modal_live`.'
        )


def skip_or_fail_live_tier() -> None:
    """Skip a live test without `PYDANTIC_AI_HARNESS_MODAL_LIVE=1` and Modal credentials.

    CI sets `MODAL_REQUIRE_LIVE`, which turns the skip into a failure, so a missing or empty
    secret cannot pass the live job by skipping every test. An empty variable counts as unset.
    """
    credentials = (os.getenv('MODAL_TOKEN_ID') and os.getenv('MODAL_TOKEN_SECRET')) or Path(
        '~/.modal.toml'
    ).expanduser().exists()
    if os.getenv('PYDANTIC_AI_HARNESS_MODAL_LIVE') == '1' and credentials:
        return  # pragma: no cover - live tier runs without coverage
    reason = 'requires PYDANTIC_AI_HARNESS_MODAL_LIVE=1 and Modal credentials'
    if os.getenv('MODAL_REQUIRE_LIVE', '').lower() in {'1', 'true', 'yes'}:
        pytest.fail(f'MODAL_REQUIRE_LIVE is set, but the live tier {reason}.')
    pytest.skip(reason)


@pytest.fixture(autouse=True)
def _no_real_modal(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Poison `modal` for every test here except the opt-in live tier, which needs credentials."""
    if 'modal_live' in request.keywords:
        skip_or_fail_live_tier()
        yield  # pragma: no cover - live tier runs without coverage
        return  # pragma: no cover
    monkeypatch.setitem(sys.modules, 'modal', _PoisonedModal('modal'))
    yield


@pytest.fixture(autouse=True)
def _fresh_missing_tools_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-arm the once-per-process warning so each test sees it independently of test order."""
    monkeypatch.setattr('pydantic_ai_harness.modal_sandbox._capability._warned_no_workspace_tools', False)


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def fake_modal(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeModal]:
    """Inject a fake `modal` module and yield its control surface."""
    control = FakeModal()
    monkeypatch.setitem(sys.modules, 'modal', control.module)
    yield control
