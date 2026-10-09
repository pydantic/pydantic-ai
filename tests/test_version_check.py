from __future__ import annotations

import importlib.util
import json
import os
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import httpx2
import pytest

import pydantic_ai._display as _display
import pydantic_ai._utils as _utils
import pydantic_ai._version_check as _version_check
import pydantic_ai.models as models

from ._inline_snapshot import snapshot

_NOW = 2_000_000_000.0
_CACHE_AGE = 24 * 60 * 60
_FIND_SPEC = importlib.util.find_spec
_DETECT_CODING_AGENT = _display.detect_coding_agent


@pytest.fixture(autouse=True)
def version_check_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path))
    # Where the cache lives on Windows, so that no platform's tests reach the real one.
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    monkeypatch.delenv('PYDANTIC_AI_NO_VERSION_CHECK', raising=False)
    monkeypatch.delenv('DO_NOT_TRACK', raising=False)
    monkeypatch.delenv('AI_AGENT', raising=False)
    monkeypatch.delenv('AGENT', raising=False)
    # The agent running the suite sets some of these itself, so which of them exist depends on who
    # ran `pytest`: gathered first, so that clearing them is the same code path either way.
    signals = [signal for _, signals in _display._CODING_AGENTS for signal in signals]  # pyright: ignore[reportPrivateUsage]
    names = {signal.partition('=')[0] for signal in signals if not signal.endswith('*')}
    names |= {
        name for name in os.environ for signal in signals if signal.endswith('*') and name.startswith(signal[:-1])
    }
    for name in names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_version_check.time, 'time', lambda: _NOW)
    monkeypatch.setattr(_display, 'detect_coding_agent', lambda: None)

    def find_spec_without_harness(name: str) -> object | None:
        return None if name == 'pydantic_ai_harness' else _FIND_SPEC(name)

    monkeypatch.setattr(importlib.util, 'find_spec', find_spec_without_harness)


@pytest.fixture
def cache_file(tmp_path: Path) -> Path:
    return tmp_path / 'pydantic-ai' / 'version-check.json'


def write_cache(cache_file: Path, *, checked_at: float, latest: dict[str, object]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({'checked_at': checked_at, 'latest': latest}), encoding='utf-8')


def install_transport(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx2.Request], httpx2.Response]
) -> list[httpx2.Request]:
    requests: list[httpx2.Request] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return handler(request)

    def get(url: str, **kwargs: Any) -> httpx2.Response:
        with httpx2.Client(transport=httpx2.MockTransport(record)) as client:
            return client.get(url, **kwargs)

    monkeypatch.setattr(_version_check.httpx2, 'get', get)
    return requests


def run_check() -> None:
    thread = _version_check.start_version_check(_version_check.read_cache())
    assert thread is not None
    assert thread.daemon is True
    thread.join()
    assert not thread.is_alive()


def test_fresh_machine_checks_without_a_cached_notice(monkeypatch: pytest.MonkeyPatch, cache_file: Path):
    monkeypatch.setattr(_version_check.platform, 'python_version', lambda: '3.14.1')
    monkeypatch.setattr(_version_check.platform, 'system', lambda: 'TestOS')
    monkeypatch.setattr(_version_check.platform, 'machine', lambda: 'test-cpu')
    monkeypatch.setattr(models, 'get_user_agent', lambda: 'pydantic-ai/2.45.0')

    def distribution_version(_module: str, distribution: str) -> str | None:
        return '0.1.6' if distribution == 'genai-prices' else None

    monkeypatch.setattr(_version_check, '_distribution_version', distribution_version)
    requests = install_transport(
        monkeypatch,
        lambda request: httpx2.Response(
            200,
            json={
                'pypi': {
                    'pydantic-ai': {'latest': '2.46.0', 'future-field': 'ignored'},
                    'unknown-package': {'latest': '99.0.0'},
                },
                'npm': {'unknown-package': {'latest': '99.0.0'}},
                'crates': {'unknown-package': {'latest': '99.0.0'}},
            },
        ),
    )

    assert _version_check.cached_updates(_version_check.read_cache()) == []
    run_check()

    assert len(requests) == 1
    request = requests[0]
    with httpx2.Client() as client:
        default_headers = dict(client.build_request('GET', _version_check.VERSION_CHECK_URL).headers)
    default_headers.pop('user-agent')
    request_headers = dict(request.headers)
    assert request_headers.pop('user-agent') == snapshot(
        'pydantic-ai/2.45.0 (Python 3.14.1; TestOS; test-cpu) genai-prices/0.1.6'
    )
    assert (request.method, str(request.url), request_headers) == (
        'GET',
        _version_check.VERSION_CHECK_URL,
        default_headers,
    )
    assert json.loads(cache_file.read_text(encoding='utf-8')) == snapshot(
        {'checked_at': 2000000000.0, 'latest': {'pydantic-ai': '2.46.0'}}
    )


def test_recent_cache_does_not_check(monkeypatch: pytest.MonkeyPatch, cache_file: Path):
    write_cache(cache_file, checked_at=_NOW - _CACHE_AGE + 1, latest={'pydantic-ai': '2.46.0'})

    get = Mock()
    monkeypatch.setattr(_version_check.httpx2, 'get', get)

    assert _version_check.start_version_check(_version_check.read_cache()) is None
    get.assert_not_called()


def test_stale_cache_checks(monkeypatch: pytest.MonkeyPatch, cache_file: Path):
    write_cache(cache_file, checked_at=_NOW - _CACHE_AGE, latest={'pydantic-ai': '2.46.0'})
    requests = install_transport(monkeypatch, lambda request: httpx2.Response(200, json={'pypi': {}}))

    run_check()

    assert len(requests) == 1


def test_future_cache_timestamp_checks(monkeypatch: pytest.MonkeyPatch, cache_file: Path):
    write_cache(cache_file, checked_at=_NOW + 1, latest={'pydantic-ai': '2.46.0'})
    requests = install_transport(monkeypatch, lambda request: httpx2.Response(200, json={'pypi': {}}))

    run_check()

    assert len(requests) == 1


def test_failed_request_is_already_throttled_and_preserves_latest(monkeypatch: pytest.MonkeyPatch, cache_file: Path):
    write_cache(cache_file, checked_at=0, latest={'pydantic-ai': '9.0.0'})

    def fail(*args: Any, **kwargs: Any) -> httpx2.Response:
        checked = json.loads(cache_file.read_text(encoding='utf-8'))
        assert checked == {'checked_at': _NOW, 'latest': {'pydantic-ai': '9.0.0'}}
        raise httpx2.ConnectTimeout('offline')

    monkeypatch.setattr(_version_check.httpx2, 'get', fail)

    run_check()

    assert json.loads(cache_file.read_text(encoding='utf-8')) == {
        'checked_at': _NOW,
        'latest': {'pydantic-ai': '9.0.0'},
    }
    assert _version_check.start_version_check(_version_check.read_cache()) is None


@pytest.mark.parametrize(
    'response',
    [
        pytest.param(httpx2.Response(404), id='404'),
        pytest.param(httpx2.Response(200, content=b'not json'), id='invalid-json'),
    ],
)
def test_failed_responses_are_silent(response: httpx2.Response, monkeypatch: pytest.MonkeyPatch, cache_file: Path):
    install_transport(monkeypatch, lambda request: response)

    run_check()

    assert json.loads(cache_file.read_text(encoding='utf-8'))['checked_at'] == _NOW


@pytest.mark.parametrize(
    'body',
    [
        pytest.param(['2.46.0'], id='non-object'),
        pytest.param({}, id='no-pypi-registry'),
        pytest.param({'pypi': []}, id='pypi-registry-not-an-object'),
        pytest.param({'pypi': {'pydantic-ai': '2.46.0'}}, id='package-not-an-object'),
        pytest.param({'pypi': {'pydantic-ai': {'other': '2.46.0'}}}, id='package-without-latest'),
        pytest.param({'pypi': {'pydantic-ai': {'latest': 246}}}, id='non-string-version'),
        pytest.param({'pypi': {'pydantic-ai': {'latest': '2.46.0\x1b[31m'}}}, id='unsafe-version'),
    ],
)
def test_malformed_response_body_caches_no_latest(body: object, monkeypatch: pytest.MonkeyPatch, cache_file: Path):
    install_transport(monkeypatch, lambda request: httpx2.Response(200, json=body))

    run_check()

    assert json.loads(cache_file.read_text(encoding='utf-8')) == {'checked_at': _NOW, 'latest': {}}


def test_unwritable_cache_still_checks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cache_root = tmp_path / 'not-a-directory'
    cache_root.write_text('occupied', encoding='utf-8')
    monkeypatch.setenv('XDG_CACHE_HOME', str(cache_root))
    requests = install_transport(
        monkeypatch,
        lambda request: httpx2.Response(200, json={'pypi': {'pydantic-ai': {'latest': '2.46.0'}}}),
    )

    run_check()

    assert len(requests) == 1
    assert cache_root.read_text(encoding='utf-8') == 'occupied'


def test_unavailable_home_still_checks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('XDG_CACHE_HOME')
    monkeypatch.setattr(Path, 'home', lambda: (_ for _ in ()).throw(RuntimeError('no home')))
    requests = install_transport(monkeypatch, lambda request: httpx2.Response(200, json={'pypi': {}}))

    run_check()

    assert len(requests) == 1


def test_corrupt_cache_is_a_miss(monkeypatch: pytest.MonkeyPatch, cache_file: Path):
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text('{', encoding='utf-8')
    requests = install_transport(monkeypatch, lambda request: httpx2.Response(200, json={'pypi': {}}))

    assert _version_check.cached_updates(_version_check.read_cache()) == []
    run_check()

    assert len(requests) == 1


@pytest.mark.parametrize(
    ('variable', 'value'),
    [
        pytest.param('PYDANTIC_AI_NO_VERSION_CHECK', '', id='specific-opt-out'),
        pytest.param('DO_NOT_TRACK', '1', id='do-not-track'),
        pytest.param('DO_NOT_TRACK', 'TRUE', id='do-not-track-case-insensitive'),
    ],
)
def test_opt_out_never_accesses_cache_or_network(variable: str, value: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(variable, value)
    cache_file = Mock()
    get = Mock()
    monkeypatch.setattr(_version_check, '_cache_file', cache_file)
    monkeypatch.setattr(_version_check.httpx2, 'get', get)

    cache = _version_check.read_cache()
    assert cache is None
    assert _version_check.cached_updates(cache) == []
    assert _version_check.start_version_check(cache) is None
    cache_file.assert_not_called()
    get.assert_not_called()


@pytest.mark.parametrize('value', ['', '0', 'false', 'FALSE'])
def test_do_not_track_false_values_allow_the_check(value: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('DO_NOT_TRACK', value)
    requests = install_transport(monkeypatch, lambda request: httpx2.Response(200, json={'pypi': {}}))

    run_check()

    assert len(requests) == 1


@pytest.mark.parametrize(
    ('installed', 'latest', 'expected'),
    [
        pytest.param('2.45.1.dev13+abc', '2.45.2', [('pydantic-ai', '2.45.2')], id='dev-older'),
        pytest.param('2.45.2.dev13+abc', '2.45.2', [('pydantic-ai', '2.45.2')], id='dev-build-of-the-release'),
        pytest.param('2.45.1+local', '2.45.2', [('pydantic-ai', '2.45.2')], id='local-older'),
        pytest.param('2.46.0', '2.45.2', [], id='installed-newer'),
        pytest.param('2.46', '2.46.0', [], id='trailing-zero-is-the-same-release'),
        pytest.param('2.47.0b1', '2.47.0', [('pydantic-ai', '2.47.0')], id='beta-before-release'),
        pytest.param('2.47.0rc2', '2.47.0', [('pydantic-ai', '2.47.0')], id='rc-before-release'),
        pytest.param('2.47.0.dev3', '2.47.0', [('pydantic-ai', '2.47.0')], id='dev-before-release'),
        pytest.param('2.47.0', '2.47.0', [], id='same-release'),
        pytest.param('2.47.0.post1', '2.47.0', [], id='post-release'),
        pytest.param('2.47.0+local', '2.47.0', [], id='local-build-of-release'),
        pytest.param('2.47.1.dev2', '2.47.0', [], id='dev-of-newer-release'),
        pytest.param('2.47.0', '2.48.0', [('pydantic-ai', '2.48.0')], id='newer-release'),
        pytest.param('2.9', '2.10', [('pydantic-ai', '2.10')], id='numeric-components'),
    ],
)
def test_cached_updates_compare_release_tuples(
    installed: str,
    latest: str,
    expected: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    cache_file: Path,
):
    write_cache(cache_file, checked_at=_NOW, latest={'pydantic-ai': latest})
    monkeypatch.setattr(_version_check, '_installed_versions', lambda: {'pydantic-ai': installed})

    assert _version_check.cached_updates(_version_check.read_cache()) == expected


def test_unsafe_versions_are_never_returned(monkeypatch: pytest.MonkeyPatch, cache_file: Path):
    write_cache(
        cache_file,
        checked_at=_NOW,
        latest={
            'pydantic-ai': '2.46.0\x1b[31m',
            'pydantic-ai-harness': '0.8.0.1.2',
        },
    )
    monkeypatch.setattr(
        _version_check,
        '_installed_versions',
        lambda: {'pydantic-ai': '2.45.0', 'pydantic-ai-harness': '0.7.0'},
    )

    assert _version_check.cached_updates(_version_check.read_cache()) == []


def test_harness_notice_requires_an_installed_harness(monkeypatch: pytest.MonkeyPatch, cache_file: Path):
    write_cache(
        cache_file,
        checked_at=_NOW,
        latest={'pydantic-ai': '2.46.0', 'pydantic-ai-harness': '0.8.0'},
    )
    monkeypatch.setattr(_version_check, '_installed_versions', lambda: {'pydantic-ai': '2.45.0'})
    cache = _version_check.read_cache()
    assert _version_check.cached_updates(cache) == [('pydantic-ai', '2.46.0')]

    monkeypatch.setattr(
        _version_check,
        '_installed_versions',
        lambda: {'pydantic-ai': '2.45.0', 'pydantic-ai-harness': '0.7.0'},
    )
    assert _version_check.cached_updates(cache) == [
        ('pydantic-ai', '2.46.0'),
        ('pydantic-ai-harness', '0.8.0'),
    ]


@pytest.mark.parametrize(
    ('variable', 'value', 'expected'),
    [
        pytest.param(
            'AI_AGENT',
            'my secret project',
            'pydantic-ai/2.45.0 (Python 3.14.1; TestOS; test-cpu) pydantic-ai-harness/0.8.0 agent/agent',
            id='unknown-name-is-private',
        ),
        pytest.param(
            'CODEX_THREAD_ID',
            '123',
            'pydantic-ai/2.45.0 (Python 3.14.1; TestOS; test-cpu) pydantic-ai-harness/0.8.0 agent/codex',
            id='known-name',
        ),
    ],
)
def test_user_agent_reports_only_known_agent_names(
    variable: str, value: str, expected: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(variable, value)
    monkeypatch.setattr(_display, 'detect_coding_agent', _DETECT_CODING_AGENT)
    monkeypatch.setattr(_version_check.platform, 'python_version', lambda: '3.14.1')
    monkeypatch.setattr(_version_check.platform, 'system', lambda: 'TestOS')
    monkeypatch.setattr(_version_check.platform, 'machine', lambda: 'test-cpu')
    monkeypatch.setattr(models, 'get_user_agent', lambda: 'pydantic-ai/2.45.0')

    def distribution_version(_module: str, distribution: str) -> str | None:
        return '0.8.0' if distribution == 'pydantic-ai-harness' else None

    monkeypatch.setattr(_version_check, '_distribution_version', distribution_version)
    requests = install_transport(monkeypatch, lambda request: httpx2.Response(200, json={'pypi': {}}))

    run_check()

    assert requests[0].headers['user-agent'] == expected


def test_no_coding_agent_adds_no_agent_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_display, 'detect_coding_agent', lambda: None)

    def distribution_version(_module: str, _distribution: str) -> None:
        return None

    monkeypatch.setattr(_version_check, '_distribution_version', distribution_version)
    requests = install_transport(monkeypatch, lambda request: httpx2.Response(200, json={'pypi': {}}))

    run_check()

    assert ' agent/' not in requests[0].headers['user-agent']


@pytest.mark.parametrize(
    ('cached', 'expected'),
    [
        pytest.param([], None, id='absent'),
        pytest.param([ValueError('broken importer')], None, id='importer-error'),
        pytest.param([object()], '1.2.3', id='present'),
    ],
)
def test_harness_version_is_defensive(cached: list[object], expected: str | None, monkeypatch: pytest.MonkeyPatch):
    def find_spec(name: str) -> object | None:
        value = cached[0] if cached else None
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(importlib.util, 'find_spec', find_spec)

    def version(_distribution: str) -> str:
        return '1.2.3'

    monkeypatch.setattr(metadata, 'version', version)

    assert _version_check.harness_version() == expected


@pytest.mark.parametrize(
    'contents',
    [
        pytest.param('[]', id='cache-not-object'),
        pytest.param('{"checked_at": true, "latest": {}}', id='timestamp-is-bool'),
        pytest.param('{"checked_at": "now", "latest": {}}', id='timestamp-is-string'),
        pytest.param('{"checked_at": 1, "latest": []}', id='latest-not-object'),
    ],
)
def test_wrong_cache_types_are_a_miss(contents: str, monkeypatch: pytest.MonkeyPatch, cache_file: Path):
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(contents, encoding='utf-8')
    requests = install_transport(monkeypatch, lambda request: httpx2.Response(200, json={'pypi': {}}))

    assert _version_check.cached_updates(_version_check.read_cache()) == []
    run_check()
    assert len(requests) == 1


def test_cache_paths_follow_the_platform(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv('XDG_CACHE_HOME')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path / 'home')
    monkeypatch.setattr(_utils, 'os', SimpleNamespace(name='posix', environ=os.environ))
    assert _version_check._cache_file() == (  # pyright: ignore[reportPrivateUsage]
        tmp_path / 'home' / '.cache' / 'pydantic-ai' / 'version-check.json'
    )

    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'local'))
    monkeypatch.setattr(_utils, 'os', SimpleNamespace(name='nt', environ=os.environ))
    assert _version_check._cache_file() == (  # pyright: ignore[reportPrivateUsage]
        tmp_path / 'local' / 'pydantic-ai' / 'version-check.json'
    )

    monkeypatch.delenv('LOCALAPPDATA')
    assert _version_check._cache_file() == (  # pyright: ignore[reportPrivateUsage]
        tmp_path / 'home' / 'AppData' / 'Local' / 'pydantic-ai' / 'version-check.json'
    )


def test_invalid_installed_version_has_an_empty_release_tuple(monkeypatch: pytest.MonkeyPatch, cache_file: Path):
    write_cache(cache_file, checked_at=_NOW, latest={'pydantic-ai': '2.46.0'})
    monkeypatch.setattr(_version_check, '_installed_versions', lambda: {'pydantic-ai': 'development'})

    assert _version_check.cached_updates(_version_check.read_cache()) == [('pydantic-ai', '2.46.0')]
