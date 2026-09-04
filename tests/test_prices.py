from __future__ import annotations

import threading

import pytest
from genai_prices import UpdatePrices
from genai_prices.data_snapshot import DataSnapshot, get_snapshot, set_custom_snapshot

from pydantic_ai import prices


def test_update_in_background(monkeypatch: pytest.MonkeyPatch):
    """The returned updater is started and can be waited on and stopped; no network is involved."""
    downloaded = threading.Event()

    def fetch(self: UpdatePrices) -> DataSnapshot:
        downloaded.set()
        return get_snapshot()

    monkeypatch.setattr(UpdatePrices, 'fetch', fetch)

    with prices.update_in_background() as updater:
        assert updater.wait(timeout=5)
        assert downloaded.is_set()

    for thread in threading.enumerate():
        if thread.name == 'genai_prices:update':
            thread.join(timeout=5)
    set_custom_snapshot(None)
