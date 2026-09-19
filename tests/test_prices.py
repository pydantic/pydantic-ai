from __future__ import annotations

import threading
from decimal import Decimal

import pytest
from genai_prices import UpdatePrices
from genai_prices.data_snapshot import DataSnapshot, get_snapshot, set_custom_snapshot
from inline_snapshot import snapshot

from pydantic_ai import prices
from pydantic_ai._genai_prices import calculate_price_for_usage
from pydantic_ai.usage import RequestUsage, RunUsage


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


def test_duration_billed_model_prices_from_audio_seconds():
    """A model with no token prices at all must not price its token counts as a confident zero.

    Grok Voice bills per audio hour, so `calc_price` finds every token rate missing and returns `0` —
    indistinguishable from a genuinely free call, which means a `cost_limit` never trips and no
    unavailable-cost warning is emitted. Reporting the duration under the name pricing knows it by is
    what makes the call priceable at all.
    """
    tokens_only = RequestUsage(input_tokens=100, output_tokens=50)
    assert calculate_price_for_usage(
        tokens_only, model_name='grok-voice-latest', provider_name='x-ai'
    ).total_price == snapshot(Decimal('0'))

    with_duration = RequestUsage(input_tokens=100, output_tokens=50)
    with_duration.audio_seconds = 5
    assert calculate_price_for_usage(
        with_duration, model_name='grok-voice-latest', provider_name='x-ai'
    ).total_price == snapshot(Decimal('0.006666666666666666666666666667'))


def test_fractional_audio_seconds_survive_into_pricing():
    """Durations arrive fractional, which is why `details` (typed `dict[str, int]`) cannot carry them."""
    usage = RequestUsage()
    usage.audio_seconds = 2.07
    assert calculate_price_for_usage(
        usage, model_name='grok-voice-latest', provider_name='x-ai'
    ).total_price == snapshot(Decimal('0.00276'))


def test_audio_seconds_accumulates_across_a_run():
    """`RunUsage.incr` must sum the duration, or a session's cost would only ever reflect one response."""
    run = RunUsage()
    for _ in range(3):
        request = RequestUsage()
        request.audio_seconds = 2.0
        run.incr(request)
    assert run.audio_seconds == snapshot(6.0)
