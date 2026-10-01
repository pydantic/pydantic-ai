"""`/update`: find a newer CLAI on the chosen channel and reinstall the `uv tool` environment with it.

`stable` follows PyPI releases. `bleeding` follows the newest commit on `main` that touches CLAI. It downloads
that commit's source archive over HTTPS, so it needs no `git`, and installs CLAI with the harness and core
packages from the same archive, whose exact dev pins are not on PyPI.
"""

import os
import re
import shlex
import sys
import tempfile
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

from anyio import CancelScope, run_process, to_thread
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from pydantic_clai2.config import UpdateChannel

if TYPE_CHECKING:
    import httpx

DISTRIBUTION = 'pydantic-clai2'
REPOSITORY = 'https://github.com/pydantic/pydantic-ai'
PACKAGES = {
    'pydantic-clai2': 'src/pydantic_clai2',
    'pydantic-ai-harness': 'src/pydantic_ai_harness',
    'pydantic-ai-slim': 'pydantic_ai_slim',
    'pydantic-graph': 'pydantic_graph',
}
"""Source subdirectories installed together from one commit's archive."""
VERSION_BYPASS = 'UV_DYNAMIC_VERSIONING_BYPASS'
"""The build backend reads versions from Git history, which an archive lacks; this supplies one instead."""
_ARCHIVE = re.compile(re.escape(REPOSITORY) + r'/archive/([0-9a-f]{40})\.tar\.gz')
PYPI_URL = 'https://pypi.org/pypi/pydantic-clai2/json'
COMMITS_URL = 'https://api.github.com/repos/pydantic/pydantic-ai/commits'
TIMEOUT = 10.0
"""Seconds a release lookup may take."""


class _VcsInfo(TypedDict):
    commit_id: str


class _DirectUrl(TypedDict, total=False):
    url: str
    vcs_info: _VcsInfo


class _ProjectInfo(TypedDict):
    version: str


class _PyPIProject(TypedDict):
    info: _ProjectInfo


class _Commit(TypedDict):
    sha: str


_DIRECT_URL = TypeAdapter(_DirectUrl)
_PYPI_PROJECT = TypeAdapter(_PyPIProject)
_COMMITS = TypeAdapter(list[_Commit])


@dataclass(frozen=True, kw_only=True)
class Installed:
    """The running CLAI: its version, the commit it was built from, and whether `uv tool` owns it."""

    version: str
    commit: str | None = None
    tool: bool = False

    @property
    def label(self) -> str:
        """The short commit for a bleeding install, otherwise the version."""
        return self.version if self.commit is None else self.commit[:9]


def installed() -> Installed:
    """Read the running distribution's metadata; `uv tool install` leaves a receipt in the environment.

    The commit comes from a bleeding archive URL, or from a `git+` install made by hand.
    """
    distribution = metadata.distribution(DISTRIBUTION)
    text = distribution.read_text('direct_url.json')
    direct_url = _DIRECT_URL.validate_json(text) if text else _DirectUrl()
    archive = _ARCHIVE.fullmatch(direct_url.get('url', ''))
    vcs = direct_url.get('vcs_info')
    return Installed(
        version=distribution.version,
        commit=archive[1] if archive else None if vcs is None else vcs['commit_id'],
        tool=Path(sys.prefix, 'uv-receipt.toml').is_file(),
    )


def latest(channel: UpdateChannel, *, transport: 'httpx.BaseTransport | None' = None) -> str:
    """The newest PyPI version for `stable`, or the newest `main` commit touching CLAI for `bleeding`."""
    import httpx

    with httpx.Client(transport=transport, timeout=TIMEOUT, follow_redirects=True) as client:
        if channel == 'stable':
            response = client.get(PYPI_URL).raise_for_status()
            return _PYPI_PROJECT.validate_json(response.content)['info']['version']
        response = client.get(
            COMMITS_URL,
            params={'sha': 'main', 'path': PACKAGES[DISTRIBUTION], 'per_page': 1},
            headers={'Accept': 'application/vnd.github+json'},
        ).raise_for_status()
    commits = _COMMITS.validate_json(response.content)
    if not commits:
        raise ValueError('GitHub returned no CLAI commits on main.')
    return commits[0]['sha']


@dataclass(frozen=True, kw_only=True)
class Update:
    """A newer CLAI: a PyPI version for `stable`, a full commit SHA for `bleeding`."""

    channel: UpdateChannel
    target: str

    @property
    def label(self) -> str:
        """The version, or the commit shortened like `git log --oneline`."""
        return self.target if self.channel == 'stable' else self.target[:9]

    def requirement(self, name: str) -> str:
        """`name` from this commit's HTTPS source archive."""
        return f'{name} @ {REPOSITORY}/archive/{self.target}.tar.gz#subdirectory={PACKAGES[name]}'

    def overrides(self) -> str:
        """Replace CLAI's exact pins on the other packages with the same archive; uv keeps these in its receipt."""
        return ''.join(f'{self.requirement(name)}\n' for name in PACKAGES if name != DISTRIBUTION)

    def environment(self) -> dict[str, str]:
        """Variables uv's build needs: a version for the Git-less archive, `0.0.0` with the commit attached."""
        return {} if self.channel == 'stable' else {VERSION_BYPASS: f'0.0.0+{self.label}'}

    def command(self, *, uv: str, overrides: Path | None) -> list[str]:
        """The `uv tool install` that replaces the current install; bleeding reads `overrides`."""
        if self.channel == 'stable':
            return [uv, 'tool', 'install', '--force', f'{DISTRIBUTION}=={self.target}']
        assert overrides is not None, 'a bleeding install needs the overrides file'
        return [uv, 'tool', 'install', '--force', '--overrides', str(overrides), self.requirement(DISTRIBUTION)]


def find_update(channel: UpdateChannel, current: Installed, target: str) -> Update | None:
    """Offer `target` unless it is what runs; a Git install on `stable` is offered the release."""
    if channel == 'bleeding':
        return None if current.commit == target else Update(channel=channel, target=target)
    return None if current.commit is None and current.version == target else Update(channel=channel, target=target)


def find_uv(*, environ: Mapping[str, str] = os.environ, windows: bool = os.name == 'nt') -> str | None:
    """`uv` from the absolute `PATH` entries only.

    Not `shutil.which`: on Windows before Python 3.12 it searches the working directory first, even with
    `path=`, so a repository could supply its own `uv.exe`.
    """
    extensions = environ.get('PATHEXT', '.EXE').split(os.pathsep) if windows else ['']
    for directory in environ.get('PATH', '').split(os.pathsep):
        if not os.path.isabs(directory):
            continue
        for extension in extensions:
            candidate = os.path.join(directory, f'uv{extension}')
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return None


def _in_thread(work: Callable[[], None]) -> None:
    threading.Thread(target=work, name='clai-update-check', daemon=True).start()


async def _run_uv(command: Sequence[str], environment: dict[str, str]) -> int:
    # Inherit the terminal so uv's progress shows; slash commands run with the editor suspended.
    result = await run_process(command, stdout=None, stderr=None, check=False, env={**os.environ, **environment})
    return result.returncode


def _write_overrides(text: str) -> Path:
    # A private file, not a fixed name in a shared temporary folder another user could plant first.
    descriptor, name = tempfile.mkstemp(prefix='clai2-overrides-', suffix='.txt')
    with os.fdopen(descriptor, 'w', encoding='utf-8') as file:
        file.write(text)
    return Path(name)


@dataclass(kw_only=True)
class Updates:
    """The footer's update notice and the `/update` command, for the channel the settings name."""

    channel: Callable[[], UpdateChannel]
    current: Installed = field(default_factory=installed)
    fetch: Callable[[UpdateChannel], str] = latest
    spawn: Callable[[Callable[[], None]], None] = _in_thread
    """Runs the background check; tests run it inline."""
    run: Callable[[Sequence[str], dict[str, str]], Awaitable[int]] = _run_uv
    find_uv: Callable[[], str | None] = find_uv
    restart_required: bool = False
    """Set after a successful install: the running environment was replaced, so the shell exits."""
    _checked: UpdateChannel | None = field(default=None, init=False)
    _found: Update | None = field(default=None, init=False)

    def segment(self) -> str:
        """A status-row hint once a background check finds an update; checks again when the channel changes.

        Only `uv tool` installs check, so a source checkout never contacts PyPI or GitHub on its own.
        """
        if not self.current.tool:
            return ''
        channel = self.channel()
        if channel != self._checked:
            self._checked, self._found = channel, None
            self.spawn(lambda: self._check(channel))
        found = self._found
        return f'update {found.label}: /update' if found is not None and found.channel == channel else ''

    def _check(self, channel: UpdateChannel) -> None:
        import httpx

        # Offline or rate-limited: no notice. `/update` reports the error when asked directly.
        with suppress(httpx.HTTPError, ValueError):
            self._found = find_update(channel, self.current, self.fetch(channel))

    async def command(self, args: list[str]) -> str:
        """Install the newest CLAI on the current channel into this `uv tool` environment."""
        if args:
            raise ValueError('Usage: /update. Pick the channel with /set updates.channel stable|bleeding.')
        channel = self.channel()
        update = find_update(channel, self.current, await to_thread.run_sync(self.fetch, channel))
        if update is None:
            return f'CLAI {self.current.label} is the newest on the {channel} channel.'
        uv = self.find_uv()
        overrides = _write_overrides(update.overrides()) if channel == 'bleeding' else None
        command = update.command(uv=uv or 'uv', overrides=overrides)
        environment = update.environment()
        if uv is None or not self.current.tool:
            # Keep the overrides file: the printed command reads it.
            variables = ''.join(f'{name}={shlex.quote(value)} ' for name, value in environment.items())
            return (
                f'CLAI {update.label} is available on the {channel} channel. CLAI updates itself only when installed '
                f'with `uv tool install` and uv is on PATH. To update by hand, run:\n{variables}{shlex.join(command)}'
            )
        # Cancelling uv halfway could leave the environment half replaced, so let it finish.
        code: int | None = None
        try:
            with CancelScope(shield=True):
                code = await self.run(command, environment)
        finally:
            if overrides is not None:
                overrides.unlink(missing_ok=True)
        if code != 0:
            return f'uv exited with status {code}; see its output above.'
        self.restart_required = True
        return f'Updated CLAI to {update.label} ({channel}). Exiting; start clai2 again to use it.'
