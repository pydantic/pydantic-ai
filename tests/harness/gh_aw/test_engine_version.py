from __future__ import annotations

import importlib.util
import urllib.error
import urllib.request
from collections.abc import Callable
from email.message import Message
from pathlib import Path
from typing import Protocol, runtime_checkable

import pytest


@runtime_checkable
class _EngineVersionModule(Protocol):
    unpublished_reason: Callable[[str], str | None]


class _Response:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        pass


@pytest.fixture
def engine_version_module() -> _EngineVersionModule:
    script = Path(__file__).parents[3] / 'src' / 'pydantic_ai_harness' / 'scripts' / 'gh_aw_engine_version.py'
    spec = importlib.util.spec_from_file_location('gh_aw_engine_version', script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert isinstance(module, _EngineVersionModule)
    return module


def test_unpublished_reason_checks_both_packages(
    engine_version_module: _EngineVersionModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    urls: list[str] = []

    def urlopen(url: str, *, timeout: float) -> _Response:
        urls.append(url)
        return _Response(200)

    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)

    assert engine_version_module.unpublished_reason('0.21.0') is None
    assert urls == [
        'https://pypi.org/pypi/pydantic-ai-harness/0.21.0/json',
        'https://pypi.org/pypi/pydantic-clai2/0.21.0/json',
    ]


def test_unpublished_reason_reports_missing_clai2_release(
    engine_version_module: _EngineVersionModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    urls: list[str] = []

    def urlopen(url: str, *, timeout: float) -> _Response:
        urls.append(url)
        if 'pydantic-clai2' in url:
            raise urllib.error.HTTPError(url, 404, 'not found', Message(), None)
        return _Response(200)

    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)

    reason = engine_version_module.unpublished_reason('0.21.0')

    assert reason is not None
    assert 'pydantic-clai2' in reason
    assert 'engine.version: 0.21.0' in reason
    assert 'HTTP 404' in reason
    assert urls == [
        'https://pypi.org/pypi/pydantic-ai-harness/0.21.0/json',
        'https://pypi.org/pypi/pydantic-clai2/0.21.0/json',
    ]


def test_unpublished_reason_stops_when_harness_release_is_missing(
    engine_version_module: _EngineVersionModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    urls: list[str] = []

    def urlopen(url: str, *, timeout: float) -> _Response:
        urls.append(url)
        raise urllib.error.HTTPError(url, 404, 'not found', Message(), None)

    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)

    reason = engine_version_module.unpublished_reason('0.21.0')

    assert reason is not None
    assert 'pydantic-ai-harness' in reason
    assert 'engine.version: 0.21.0' in reason
    assert 'HTTP 404' in reason
    assert urls == ['https://pypi.org/pypi/pydantic-ai-harness/0.21.0/json']


def test_unpublished_reason_names_package_on_request_error(
    engine_version_module: _EngineVersionModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    urls: list[str] = []

    def urlopen(url: str, *, timeout: float) -> _Response:
        urls.append(url)
        if 'pydantic-clai2' in url:
            raise OSError('connection failed')
        return _Response(200)

    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)

    reason = engine_version_module.unpublished_reason('0.21.0')

    assert reason == 'Could not ask PyPI whether pydantic-clai2 0.21.0 is published: connection failed'
    assert urls == [
        'https://pypi.org/pypi/pydantic-ai-harness/0.21.0/json',
        'https://pypi.org/pypi/pydantic-clai2/0.21.0/json',
    ]
