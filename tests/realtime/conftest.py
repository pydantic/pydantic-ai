"""Realtime test configuration.

`test_finance_demo.py` imports the `pydantic_ai_examples` package, which isn't installed in the main
test environment (examples are import-checked separately via `tests/import_examples.py`). Skip
collecting it unless that package is available, so the unit-test job doesn't error on import.
"""

import importlib.util

import pytest

collect_ignore: list[str] = []

if importlib.util.find_spec('pydantic_ai_examples') is None:
    collect_ignore.append('test_finance_demo.py')


@pytest.fixture(autouse=True)
def _realtime_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide placeholder API keys so realtime models can resolve their default providers offline.

    The realtime models resolve their provider (and its API client) eagerly at construction, like
    `OpenAIChatModel` / `GoogleModel`. These tests never hit the network, so a placeholder key is
    enough to let `OpenAIRealtimeModel()` / `GoogleRealtimeModel()` build their default providers.
    """
    monkeypatch.setenv('OPENAI_API_KEY', 'mock-api-key')
    monkeypatch.setenv('GOOGLE_API_KEY', 'mock-api-key')
