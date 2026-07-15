"""Unit tests for local process ownership in `pydantic_ai.prices`; no model request is involved."""

# pyright: reportPrivateUsage=false
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from pydantic_ai import prices


@pytest.fixture(autouse=True)
def reset_updater(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prices, '_updater', None)


def test_update_in_background_starts_once_across_concurrent_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    start_entered = Event()
    allow_start = Event()
    instances: list[object] = []

    class StubUpdatePrices:
        def __init__(self) -> None:
            instances.append(self)

        def start(self) -> None:
            start_entered.set()
            assert allow_start.wait(timeout=5)

    monkeypatch.setattr(prices, '_UpdatePrices', StubUpdatePrices)

    with ThreadPoolExecutor(max_workers=8) as executor:
        first = executor.submit(prices.update_in_background)
        assert start_entered.wait(timeout=5)
        others = [executor.submit(prices.update_in_background) for _ in range(7)]
        allow_start.set()
        first.result(timeout=5)
        for future in others:
            future.result(timeout=5)

    prices.update_in_background()
    assert len(instances) == 1


def test_update_in_background_propagates_start_failure_and_can_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingUpdatePrices:
        def start(self) -> None:
            raise RuntimeError('start failed')

    monkeypatch.setattr(prices, '_UpdatePrices', FailingUpdatePrices)
    with pytest.raises(RuntimeError, match='start failed'):
        prices.update_in_background()

    started = False

    class SuccessfulUpdatePrices:
        def start(self) -> None:
            nonlocal started
            started = True

    monkeypatch.setattr(prices, '_UpdatePrices', SuccessfulUpdatePrices)
    prices.update_in_background()
    assert started
