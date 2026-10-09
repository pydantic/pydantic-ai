"""Best-effort checks for newer Pydantic AI releases."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import re
import threading
import time
from importlib import metadata
from pathlib import Path

import httpx2
from typing_extensions import TypedDict

from ._utils import atomic_write_bytes, is_str_dict, user_cache_dir

VERSION_CHECK_URL = 'https://info.pydantic.info/versions.json'
"""Reports the latest release of every package Pydantic publishes, keyed by registry and then by name.

Read as `{'pypi': {'pydantic-ai': {'latest': '2.46.0'}}}`. The endpoint only ever adds keys, so
anything this doesn't recognise is ignored rather than treated as a malformed response.
"""

_CHECK_INTERVAL = 24 * 60 * 60
_VERSION_PATTERN = re.compile(r'^\d+(?:\.\d+){0,3}$')
_RELEASE_PATTERN = re.compile(r'^\d+(?:\.\d+)*')
_PRE_RELEASE_PATTERN = re.compile(r'^\d+(?:\.\d+)*[.-]?(?:a|b|c|rc|alpha|beta|pre|preview|dev)', re.IGNORECASE)
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
        and _is_newer(latest, installed_version)
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
    return user_cache_dir() / 'version-check.json'


def _read_cache() -> _VersionCache | None:
    try:
        data = json.loads(_cache_file().read_bytes())
        if not is_str_dict(data):
            return None
        checked_at = data.get('checked_at')
        if not isinstance(checked_at, (int, float)) or isinstance(checked_at, bool):
            return None
        latest = _validated_latest(data.get('latest'))
        if latest is None:
            return None
        return {'checked_at': float(checked_at), 'latest': latest}
    except Exception:
        return None


def _write_cache(cache: _VersionCache) -> None:
    try:
        atomic_write_bytes(_cache_file(), json.dumps(cache, separators=(',', ':')).encode())
    except Exception:
        pass


def _validated_latest(value: object) -> dict[str, str] | None:
    if not is_str_dict(value):
        return None

    latest: dict[str, str] = {}
    for distribution in _DISTRIBUTIONS:
        version = value.get(distribution)
        if isinstance(version, str) and _VERSION_PATTERN.fullmatch(version):
            latest[distribution] = version
    return latest


def _latest_from_response(value: object) -> dict[str, str] | None:
    """Flatten the endpoint's `registry -> package -> {'latest': version}` into what the cache keeps."""
    if not is_str_dict(value) or not is_str_dict(packages := value.get('pypi')):
        return None
    return _validated_latest(
        {name: package.get('latest') for name, package in packages.items() if is_str_dict(package)}
    )


def _installed_versions() -> dict[str, str]:
    from . import __version__

    installed = {'pydantic-ai': __version__}
    if (harness := harness_version()) is not None:
        installed['pydantic-ai-harness'] = harness
    return installed


def harness_version() -> str | None:
    """The installed Pydantic AI Harness version, or `None` when it isn't installed or can't be read.

    The one lookup the banner, the update notice and the `User-Agent` share, so that the three can't
    disagree about which harness is installed.
    """
    return _distribution_version('pydantic_ai_harness', 'pydantic-ai-harness')


def _distribution_version(module: str, distribution: str) -> str | None:
    try:
        if importlib.util.find_spec(module) is not None:
            return metadata.version(distribution)
    except Exception:
        # Best-effort enrichment: a package is named only when it can be found and named without
        # trouble. `find_spec` raises for a module whose `__spec__` is None and for anything a
        # custom importer objects to, and the distribution can be missing or unreadable.
        pass
    return None


def _user_agent() -> str:
    # Imported at call time so this module remains safe to reach from the banner during package
    # initialization, like the banner's own version lookup.
    from . import _display, models

    user_agent = (
        f'{models.get_user_agent()} (Python {platform.python_version()}; {platform.system()}; {platform.machine()})'
    )
    if (harness := harness_version()) is not None:
        user_agent += f' pydantic-ai-harness/{harness}'
    if (prices_version := _distribution_version('genai_prices', 'genai-prices')) is not None:
        user_agent += f' genai-prices/{prices_version}'

    if agent := _display.known_coding_agent():
        user_agent += f' agent/{agent}'
    return user_agent


def _is_newer(latest: str, installed: str) -> bool:
    """Whether the plain release `latest` comes after `installed`, which may carry any suffix."""
    if (latest_release := _release_tuple(latest)) != (installed_release := _release_tuple(installed)):
        return latest_release > installed_release
    # A pre-release or development build comes before the release it leads up to, so someone on
    # `2.47.0b1` hears when `2.47.0` is out. A post-release or local build of it does not.
    return _PRE_RELEASE_PATTERN.match(installed) is not None


def _release_tuple(version: str) -> tuple[int, ...]:
    match = _RELEASE_PATTERN.match(version)
    if match is None:
        return ()
    release = list(map(int, match.group().split('.')))
    # Trailing zeros are dropped so that `2.46` and `2.46.0` compare as the same release.
    while release and release[-1] == 0:
        release.pop()
    return tuple(release)
