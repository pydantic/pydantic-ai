from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from pydantic_ai import UsageExtractionFailedWarning, usage as usage_module
from pydantic_ai.usage import RequestUsage


class FakeProvider:
    def __init__(self, outcome: SimpleNamespace | Exception):
        self.outcome = outcome

    def extract_usage(self, data: Any, *, api_flavor: str) -> tuple[str, SimpleNamespace]:
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return 'model', self.outcome


class FakeSnapshot:
    def __init__(self, outcomes: dict[tuple[str | None, str | None], FakeProvider | Exception]):
        self.outcomes = outcomes

    def find_provider(self, model_ref: None, provider_id: str | None, provider_api_url: str | None) -> FakeProvider:
        outcome = self.outcomes.get((provider_id, provider_api_url), LookupError())
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_usage_extraction_failure_warns_once(monkeypatch: pytest.MonkeyPatch):
    """Usage extraction is best effort, but unexpected failures must be observable."""
    provider = FakeProvider(RuntimeError('provider extractor raised'))
    snapshot = FakeSnapshot(
        {
            (None, 'https://example.com'): provider,
            ('provider', None): provider,
            ('fallback', None): provider,
        }
    )
    monkeypatch.setattr(usage_module, 'get_snapshot', lambda: snapshot)

    with pytest.warns(
        UsageExtractionFailedWarning, match='Failed to extract usage with `genai-prices`: RuntimeError'
    ) as caught:
        extracted = RequestUsage.extract(
            {}, provider='provider', provider_url='https://example.com', provider_fallback='fallback'
        )

    assert extracted == RequestUsage()
    assert len(caught) == 1


def test_usage_extraction_falls_back_without_warning(monkeypatch: pytest.MonkeyPatch):
    """A fake snapshot isolates fallback behavior that no fixed provider response guarantees."""
    snapshot = FakeSnapshot(
        {
            (None, 'https://example.com'): FakeProvider(RuntimeError('gateway extractor failed')),
            ('provider', None): FakeProvider(SimpleNamespace(input_tokens=42)),
        }
    )
    monkeypatch.setattr(usage_module, 'get_snapshot', lambda: snapshot)

    extracted = RequestUsage.extract(
        {}, provider='provider', provider_url='https://example.com', provider_fallback='fallback'
    )

    assert extracted == RequestUsage(input_tokens=42)


def test_usage_construction_failure_warns(monkeypatch: pytest.MonkeyPatch):
    """A fake usage shape reproduces incompatibility with a future `genai-prices` field."""
    provider = FakeProvider(SimpleNamespace(requests=1))
    snapshot = FakeSnapshot(
        {
            (None, 'https://example.com'): provider,
            ('provider', None): provider,
            ('fallback', None): provider,
        }
    )
    monkeypatch.setattr(usage_module, 'get_snapshot', lambda: snapshot)

    with pytest.warns(UsageExtractionFailedWarning, match='AttributeError'):
        extracted = RequestUsage.extract(
            {}, provider='provider', provider_url='https://example.com', provider_fallback='fallback'
        )

    assert extracted == RequestUsage()


def test_usage_provider_lookup_failure_warns(monkeypatch: pytest.MonkeyPatch):
    """A fake snapshot exercises an unexpected lookup failure without corrupting pricing data."""
    snapshot = FakeSnapshot({(None, 'https://example.com'): RuntimeError('snapshot failed')})
    monkeypatch.setattr(usage_module, 'get_snapshot', lambda: snapshot)

    with pytest.warns(UsageExtractionFailedWarning, match='RuntimeError: snapshot failed'):
        extracted = RequestUsage.extract(
            {}, provider='unknown', provider_url='https://example.com', provider_fallback='fallback'
        )

    assert extracted == RequestUsage()
