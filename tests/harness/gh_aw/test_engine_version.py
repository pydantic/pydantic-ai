from __future__ import annotations

import importlib.util
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from email.message import Message
from io import BytesIO
from pathlib import Path
from typing import Protocol, runtime_checkable

import pytest


@runtime_checkable
class _EngineVersionModule(Protocol):
    unpublished_reason: Callable[[str], str | None]


class _Response:
    def __init__(self, status: int, body: bytes = b'{}') -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        pass


HARNESS_URL = 'https://pypi.org/pypi/pydantic-ai-harness/0.21.0/json'
CLAI2_INDEX = 'https://pypi.org/simple/pydantic-clai2/'


def _index(*files: tuple[str, bool | str]) -> bytes:
    """PyPI's index JSON for `pydantic-clai2`: each file with its yanked state."""
    return json.dumps({'files': [{'filename': name, 'yanked': yanked} for name, yanked in files]}).encode()


ALPHA = _index(('pydantic_clai2-0.21.0a0-py3-none-any.whl', False), ('pydantic_clai2-0.21.0a0.tar.gz', False))


@pytest.fixture
def engine_version_module(monkeypatch: pytest.MonkeyPatch) -> _EngineVersionModule:
    script = Path(__file__).parents[3] / 'src' / 'pydantic_ai_harness' / 'scripts' / 'gh_aw_engine_version.py'
    spec = importlib.util.spec_from_file_location('gh_aw_engine_version', script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Pydantic resolves the script's postponed annotations through `sys.modules`, as when it runs as `__main__`.
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    assert isinstance(module, _EngineVersionModule)
    return module


def test_unpublished_reason_checks_both_packages(
    engine_version_module: _EngineVersionModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    urls: list[str] = []

    def urlopen(request: urllib.request.Request, *, timeout: float) -> _Response:
        url = request.full_url
        urls.append(url)
        # The index's JSON form; its HTML form is the default.
        assert request.get_header('Accept') == 'application/vnd.pypi.simple.v1+json'
        return _Response(200, ALPHA if url == CLAI2_INDEX else b'{}')

    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)

    assert engine_version_module.unpublished_reason('0.21.0') is None
    assert urls == [HARNESS_URL, CLAI2_INDEX]


def test_unpublished_reason_reports_missing_clai2_release(
    engine_version_module: _EngineVersionModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    urls: list[str] = []
    error_response = BytesIO()

    def urlopen(request: urllib.request.Request, *, timeout: float) -> _Response:
        url = request.full_url
        urls.append(url)
        if 'pydantic-clai2' in url:
            raise urllib.error.HTTPError(url, 404, 'not found', Message(), error_response)
        return _Response(200)

    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)

    reason = engine_version_module.unpublished_reason('0.21.0')

    assert reason is not None
    assert 'pydantic-clai2' in reason
    assert 'engine.version: 0.21.0' in reason
    assert 'HTTP 404' in reason
    assert error_response.closed
    assert urls == [HARNESS_URL, CLAI2_INDEX]


def test_unpublished_reason_stops_when_harness_release_is_missing(
    engine_version_module: _EngineVersionModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    urls: list[str] = []
    error_response = BytesIO()

    def urlopen(request: urllib.request.Request, *, timeout: float) -> _Response:
        url = request.full_url
        urls.append(url)
        raise urllib.error.HTTPError(url, 404, 'not found', Message(), error_response)

    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)

    reason = engine_version_module.unpublished_reason('0.21.0')

    assert reason is not None
    assert 'pydantic-ai-harness' in reason
    assert 'engine.version: 0.21.0' in reason
    assert 'HTTP 404' in reason
    assert error_response.closed
    assert urls == [HARNESS_URL]


def test_unpublished_reason_names_package_on_request_error(
    engine_version_module: _EngineVersionModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    urls: list[str] = []

    def urlopen(request: urllib.request.Request, *, timeout: float) -> _Response:
        url = request.full_url
        urls.append(url)
        if 'pydantic-clai2' in url:
            raise OSError('connection failed')
        return _Response(200)

    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)

    reason = engine_version_module.unpublished_reason('0.21.0')

    assert reason == 'Could not ask PyPI whether pydantic-clai2 0.21.0 is published: connection failed'
    assert urls == [HARNESS_URL, CLAI2_INDEX]


@pytest.mark.parametrize(
    ('index', 'published'),
    [
        # CLAI publishes each release as a pre-release until 1.0, then as a final.
        pytest.param(ALPHA, True, id='alpha'),
        pytest.param(_index(('pydantic_clai2-0.21.0b1.tar.gz', False)), True, id='beta'),
        pytest.param(_index(('pydantic_clai2-0.21.0-py3-none-any.whl', False)), True, id='final'),
        # Another release's stages do not satisfy the pin, nor do yanked files, which pip skips.
        pytest.param(
            _index(('pydantic_clai2-0.20.0.tar.gz', False), ('pydantic_clai2-0.21.1a0.tar.gz', False)),
            False,
            id='other-releases',
        ),
        pytest.param(
            _index(('pydantic_clai2-0.21.0a0.tar.gz', True), ('pydantic_clai2-0.21.0a0-py3-none-any.whl', 'broken')),
            False,
            id='yanked',
        ),
    ],
)
def test_unpublished_reason_accepts_any_stage_of_the_clai2_release(
    engine_version_module: _EngineVersionModule, monkeypatch: pytest.MonkeyPatch, index: bytes, published: bool
) -> None:
    def urlopen(request: urllib.request.Request, *, timeout: float) -> _Response:
        return _Response(200, index if request.full_url == CLAI2_INDEX else b'{}')

    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)

    reason = engine_version_module.unpublished_reason('0.21.0')

    if published:
        assert reason is None
    else:
        assert reason is not None
        assert 'no unyanked `pydantic-clai2` release matching `>=0.21.0a0,<=0.21.0`' in reason
