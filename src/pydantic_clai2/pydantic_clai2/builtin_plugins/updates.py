"""The built-in `updates` plugin: show a newer `pydantic-clai2` release in the status row."""

import asyncio
import re
from contextlib import suppress
from importlib import metadata

import httpx2
from pydantic import BaseModel, ValidationError

from ..plugins import PluginHost, SessionEnd, SessionStart

PYPI_URL = 'https://pypi.org/pypi/pydantic-clai2/json'
_FINAL_RELEASE = re.compile(r'\d+(\.\d+)*')


class _ReleaseInfo(BaseModel):
    version: str
    yanked: bool = False


class _Project(BaseModel):
    """The one field CLAI reads from PyPI's JSON API."""

    info: _ReleaseInfo


def final_release(version: str) -> tuple[int, ...] | None:
    """Comparable parts of a final release such as `0.51.0`; `None` for anything else.

    Pre-releases, post-releases, dev builds, and local versions (a source checkout is
    `0.51.0.dev3+abc`) are never compared, so they never see or cause a notice. Trailing
    zeros are dropped because `0.51` and `0.51.0` are the same release.
    """
    if not _FINAL_RELEASE.fullmatch(version):
        return None
    parts = [int(part) for part in version.split('.')]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


async def latest_version(*, transport: httpx2.AsyncBaseTransport | None = None) -> str | None:
    """The newest non-yanked version on PyPI, or `None` when PyPI cannot say."""
    try:
        async with httpx2.AsyncClient(transport=transport, timeout=5, follow_redirects=False) as client:
            response = await client.get(PYPI_URL, headers={'Accept': 'application/json'})
    except httpx2.HTTPError:
        return None
    if response.status_code != 200:
        return None
    try:
        info = _Project.model_validate_json(response.content).info
    except ValidationError:
        return None
    return None if info.yanked else info.version


def activate(host: PluginHost[None]) -> None:
    """Check PyPI once in the background per load; a newer final release shows in the status row."""
    # Headless runs and redirected output have no status row to show a notice in.
    if not host.console.is_terminal:
        return
    try:
        current = final_release(metadata.version('pydantic-clai2'))
    except metadata.PackageNotFoundError:
        return
    if current is None:
        return
    available: str | None = None
    check: asyncio.Task[None] | None = None

    async def run_check() -> None:
        nonlocal available
        latest = await latest_version()
        newer = None if latest is None else final_release(latest)
        if newer is not None and newer > current:
            available = latest

    @host.on('session_start')
    async def start(event: SessionStart) -> None:
        nonlocal check
        # Not awaited: startup never waits for the network.
        check = asyncio.create_task(run_check(), name='clai2-update-check')

    @host.on('session_end')
    async def stop(event: SessionEnd) -> None:
        if check is not None:
            check.cancel()
            with suppress(asyncio.CancelledError):
                await check

    @host.status_segment
    def notice() -> str:
        return f'clai2 {available} available' if available else ''
