"""Best-effort checks for newer Pydantic AI releases."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import re
import tempfile
import threading
import time
from importlib import metadata
from pathlib import Path

import httpx2
from typing_extensions import TypedDict

VERSION_CHECK_URL = 'https://pydantic.dev/docs/api/versions'
"""Reports the latest release of every package Pydantic publishes, keyed by registry and then by name.

Read as `{'pypi': {'pydantic-ai': {'latest': '2.46.0'}}}`. The endpoint only ever adds keys, so
anything this doesn't recognise is ignored rather than treated as a malformed response.
"""

_CHECK_INTERVAL = 24 * 60 * 60
_VERSION_PATTERN = re.compile(r'^\d+(?:\.\d+){0,3}$')
_RELEASE_PATTERN = re.compile(r'^\d+(?:\.\d+)*')
_DISTRIBUTIONS = ('pydantic-ai', 'pydantic-ai-harness')


class _VersionCache(TypedDict):
    checked_at: float
    latest: dict[str, str]


def version_check_enabled() -> bool:
    """Whether the environment allows the version check."""
    if 'PYDANTIC_AI_NO_VERSION_CHECK' in os.environ:
        return False
    do_not_track = os.environ.get('DO_NOT_TRACK', '')
    return do_not_track.lower() in ('', '0', 'false')


def cached_updates() -> list[tuple[str, str]]:
    """Return newer releases found in the cache, without making a request."""
    if not version_check_enabled() or (cache := _read_cache()) is None:
        return []

    installed = _installed_versions()
    return [
        (distribution, latest)
        for distribution in _DISTRIBUTIONS
        if (latest := cache['latest'].get(distribution)) is not None
        and (installed_version := installed.get(distribution)) is not None
        and _release_tuple(latest) > _release_tuple(installed_version)
    ]


def start_version_check() -> threading.Thread | None:
    """Start a due version check in a daemon thread, returning it for deterministic tests."""
    if not version_check_enabled():
        return None

    now = time.time()
    cache = _read_cache()
    # A timestamp from the future is a clock that was wrong when it was written, not a check that
    # is still fresh: trusting it would put off the next check until the clock caught up with it.
    if cache is not None and 0 <= now - cache['checked_at'] < _CHECK_INTERVAL:
        return None

    thread = threading.Thread(target=_check_for_updates, args=(now, cache), daemon=True)
    thread.start()
    return thread


def _check_for_updates(now: float, cache: _VersionCache | None) -> None:
    try:
        latest = cache['latest'] if cache is not None else {}
        # Record the attempt first so an unavailable endpoint or an interrupted process does not
        # turn every subsequent banner into another request.
        _write_cache({'checked_at': now, 'latest': latest})

        response = httpx2.get(
            VERSION_CHECK_URL,
            headers={'User-Agent': _user_agent()},
            timeout=httpx2.Timeout(timeout=5, connect=2),
        )
        response.raise_for_status()
        if (fetched := _latest_from_response(response.json())) is not None:
            _write_cache({'checked_at': now, 'latest': fetched})
    except Exception:
        # The version check is a courtesy, and a courtesy that fails is not worth an agent run.
        pass


def _cache_file() -> Path:
    if os.name == 'nt':
        base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
    else:
        base = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    return base / 'pydantic-ai' / 'version-check.json'


def _read_cache() -> _VersionCache | None:
    try:
        value = json.loads(_cache_file().read_bytes())
        if not isinstance(value, dict):
            return None
        checked_at = value.get('checked_at')
        if not isinstance(checked_at, (int, float)) or isinstance(checked_at, bool):
            return None
        latest = _validated_latest(value.get('latest'))
        if latest is None:
            return None
        return {'checked_at': float(checked_at), 'latest': latest}
    except Exception:
        return None


def _write_cache(cache: _VersionCache) -> None:
    tmp_path: Path | None = None
    try:
        cache_file = _cache_file()
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=cache_file.parent, prefix=f'.{cache_file.name}.', delete=False
        )
        tmp_path = Path(tmp_file.name)
        with tmp_file:
            json.dump(cache, tmp_file, separators=(',', ':'))
            tmp_file.flush()
        os.replace(tmp_path, cache_file)
    except Exception:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def _validated_latest(value: object) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None

    latest: dict[str, str] = {}
    for distribution in _DISTRIBUTIONS:
        version = value.get(distribution)
        if isinstance(version, str) and _VERSION_PATTERN.fullmatch(version):
            latest[distribution] = version
    return latest


def _latest_from_response(value: object) -> dict[str, str] | None:
    """Flatten the endpoint's `registry -> package -> {'latest': version}` into what the cache keeps."""
    packages = value.get('pypi') if isinstance(value, dict) else None
    if not isinstance(packages, dict):
        return None
    return _validated_latest(
        {name: package.get('latest') for name, package in packages.items() if isinstance(package, dict)}
    )


def _installed_versions() -> dict[str, str]:
    from . import __version__

    installed = {'pydantic-ai': __version__}
    if (harness_version := _distribution_version('pydantic_ai_harness', 'pydantic-ai-harness')) is not None:
        installed['pydantic-ai-harness'] = harness_version
    return installed


def _distribution_version(module: str, distribution: str) -> str | None:
    try:
        if importlib.util.find_spec(module) is not None:
            return metadata.version(distribution)
    except Exception:
        pass
    return None


def _user_agent() -> str:
    # Imported at call time so this module remains safe to reach from the banner during package
    # initialization, like the banner's own version lookup.
    from . import _display, models

    user_agent = (
        f'{models.get_user_agent()} (Python {platform.python_version()}; {platform.system()}; {platform.machine()})'
    )
    if (harness_version := _distribution_version('pydantic_ai_harness', 'pydantic-ai-harness')) is not None:
        user_agent += f' pydantic-ai-harness/{harness_version}'
    if (prices_version := _distribution_version('genai_prices', 'genai-prices')) is not None:
        user_agent += f' genai-prices/{prices_version}'

    if agent := _display.known_coding_agent():
        user_agent += f' agent/{agent}'
    return user_agent


def _release_tuple(version: str) -> tuple[int, ...]:
    match = _RELEASE_PATTERN.match(version)
    if match is None:
        return ()
    release = list(map(int, match.group().split('.')))
    # Trailing zeros are dropped so that `2.46` and `2.46.0` compare as the same release.
    while release and release[-1] == 0:
        release.pop()
    return tuple(release)
