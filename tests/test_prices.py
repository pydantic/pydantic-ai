"""Tests for `pydantic_ai.prices`."""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import threading

import pytest
from genai_prices import UpdatePrices, update_prices

from pydantic_ai import prices


@pytest.fixture(autouse=True)
def isolate(monkeypatch: pytest.MonkeyPatch):
    # Both modules hold a process-wide singleton; reset both so tests
    # don't leak threads or trip `RuntimeError` on the next `start()`.
    def noop(self: UpdatePrices) -> None: ...

    monkeypatch.setattr(UpdatePrices, 'fetch', noop)
    prices._updater = None
    update_prices._global_update_prices = None
    yield
    if prices._updater is not None:
        prices._updater.stop()
    prices._updater = None
    update_prices._global_update_prices = None


def _running_threads() -> int:
    return sum(t.name == 'genai_prices:update' and t.is_alive() for t in threading.enumerate())


def test_starts_one_thread():
    prices.update_in_background()
    prices.update_in_background()
    prices.update_in_background()
    assert _running_threads() == 1


def test_swallows_failure(monkeypatch: pytest.MonkeyPatch):
    def boom(self: UpdatePrices, *, wait: bool | float = False) -> None:
        raise RuntimeError

    monkeypatch.setattr(UpdatePrices, 'start', boom)
    prices.update_in_background()
    assert _running_threads() == 0


def test_retries_after_failure(monkeypatch: pytest.MonkeyPatch):
    real_start = UpdatePrices.start

    def boom(self: UpdatePrices, *, wait: bool | float = False) -> None:
        raise RuntimeError

    monkeypatch.setattr(UpdatePrices, 'start', boom)
    prices.update_in_background()

    monkeypatch.setattr(UpdatePrices, 'start', real_start)
    prices.update_in_background()
    assert _running_threads() == 1
