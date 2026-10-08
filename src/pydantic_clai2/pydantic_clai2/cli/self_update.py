"""`/update`: find a newer CLAI on the chosen channel and reinstall the `uv tool` environment with it.

`stable` follows PyPI releases. `main` follows the `clai2-bleeding` GitHub release, which CI refreshes with
sdists built from each `main` commit that changes CLAI or a package it pins. It installs CLAI with the harness
and core sdists from the same commit, whose exact dev pins are not on PyPI. Release downloads are not GitHub
API requests, so neither the check nor the install needs `git` or meets the unauthenticated API rate limit. A
source checkout is updated the same way: the install becomes a `uv tool` CLAI. CLAI then restarts as the new build and resumes the conversation.
On Windows, which locks the files of a running program, the install runs in a new PowerShell window after
CLAI exits.
"""

import base64
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from anyio import CancelScope, run_process, to_thread
from pydantic import StringConstraints, TypeAdapter
from typing_extensions import TypedDict

from pydantic_clai2.config import UpdateChannel

if TYPE_CHECKING:
    import httpx

DISTRIBUTION = 'pydantic-clai2'
PACKAGES = (
    'pydantic-clai2',
    'pydantic-ai-harness[coder]',
    'pydantic-ai-slim[anthropic,mcp,openai]',
    'pydantic-graph',
)
"""Requirements installed together from one commit's sdists; `scripts/build_bleeding.sh` builds the same list."""
BLEEDING_URL = 'https://github.com/pydantic/pydantic-ai/releases/download/clai2-bleeding'
"""The release the `clai2-bleeding` workflow publishes: one sdist per package and the manifest."""
BLEEDING_URL_VARIABLE = 'CLAI_BLEEDING_URL'
"""Points `main` at another copy of the release, such as a fork's or a local folder served over HTTP."""
MANIFEST = 'clai2-bleeding.json'
"""Names the commit the release's sdists were built from."""
_COMMIT_ARCHIVE = re.compile(r'.*[/-]([0-9a-f]{40})\.tar\.gz')
"""A `main` sdist, or the commit archive earlier builds installed: the full commit ends the file name."""
PYPI_URL = 'https://pypi.org/pypi/pydantic-clai2/json'
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


class _Manifest(TypedDict):
    commit: Annotated[str, StringConstraints(pattern=r'^[0-9a-f]{40}$')]


_DIRECT_URL = TypeAdapter(_DirectUrl)
_PYPI_PROJECT = TypeAdapter(_PyPIProject)
_MANIFEST = TypeAdapter(_Manifest)


def bleeding_url(*, environ: Mapping[str, str] = os.environ) -> str:
    """The `clai2-bleeding` release's download folder: `CLAI_BLEEDING_URL` when set, otherwise CLAI's own release."""
    return environ.get(BLEEDING_URL_VARIABLE, '').rstrip('/') or BLEEDING_URL


@dataclass(frozen=True, kw_only=True)
class Installed:
    """The running CLAI: its version, the commit it was built from, and whether `uv tool` owns it."""

    version: str
    commit: str | None = None
    tool: bool = False

    @property
    def label(self) -> str:
        """The short commit for a `main` install, otherwise the version."""
        return self.version if self.commit is None else self.commit[:9]


def installed() -> Installed:
    """Read the running distribution's metadata; `uv tool install` leaves a receipt in the environment.

    The commit comes from a `main` sdist or archive URL, or from a `git+` install made by hand.
    """
    distribution = metadata.distribution(DISTRIBUTION)
    text = distribution.read_text('direct_url.json')
    direct_url = _DIRECT_URL.validate_json(text) if text else _DirectUrl()
    archive = _COMMIT_ARCHIVE.fullmatch(direct_url.get('url', ''))
    vcs = direct_url.get('vcs_info')
    return Installed(
        version=distribution.version,
        commit=archive[1] if archive else None if vcs is None else vcs['commit_id'],
        tool=Path(sys.prefix, 'uv-receipt.toml').is_file(),
    )


def latest(channel: UpdateChannel, *, transport: 'httpx.BaseTransport | None' = None) -> str:
    """The newest PyPI version for `stable`, or the commit the `clai2-bleeding` release was built from for `main`."""
    import httpx

    with httpx.Client(transport=transport, timeout=TIMEOUT, follow_redirects=True) as client:
        if channel == 'stable':
            response = client.get(PYPI_URL).raise_for_status()
            return _PYPI_PROJECT.validate_json(response.content)['info']['version']
        response = client.get(f'{bleeding_url()}/{MANIFEST}').raise_for_status()
    return _MANIFEST.validate_json(response.content)['commit']


@dataclass(frozen=True, kw_only=True)
class Update:
    """A newer CLAI: a PyPI version for `stable`, a full commit SHA for `main`."""

    channel: UpdateChannel
    target: str

    @property
    def label(self) -> str:
        """The version, or the commit shortened like `git log --oneline`."""
        return self.target if self.channel == 'stable' else self.target[:9]

    def requirement(self, name: str) -> str:
        """`name` from this commit's sdist in the `clai2-bleeding` release, named as `scripts/build_bleeding.sh` names it."""
        stem = name.partition('[')[0].replace('-', '_')
        return f'{name} @ {bleeding_url()}/{stem}-{self.target}.tar.gz'

    def overrides(self) -> str:
        """Install CLAI and its pinned packages from one commit, preserving extras that uv overrides replace."""
        return ''.join(f'{self.requirement(name)}\n' for name in PACKAGES)

    def command(self, *, uv: str, overrides: Path | None) -> list[str]:
        """The `uv tool install` that replaces the current install; `main` reads `overrides`."""
        if self.channel == 'stable':
            return [uv, 'tool', 'install', '--force', f'{DISTRIBUTION}=={self.target}']
        assert overrides is not None, 'a main install needs the overrides file'
        return [uv, 'tool', 'install', '--force', '--overrides', str(overrides), DISTRIBUTION]


def find_update(channel: UpdateChannel, current: Installed, target: str) -> Update | None:
    """Offer `target` unless it is what runs; a Git install on `stable` is offered the release."""
    if channel == 'main':
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


def _powershell_quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def powershell(command: Sequence[str]) -> str:
    """The install as one PowerShell line, for Windows."""
    return f'& {" ".join(_powershell_quote(part) for part in command)}'


def after_exit_script(command: Sequence[str], *, pid: int, overrides: Path | None) -> str:
    """A PowerShell script that waits for CLAI to exit, installs, removes `overrides`, and keeps its window open.

    Windows refuses to replace files a running program uses, so the install cannot run inside CLAI.
    """
    lines = [
        f'Wait-Process -Id {pid} -ErrorAction SilentlyContinue',
        # The `clai2.exe` launcher exits just after the Python process it started.
        'Start-Sleep -Seconds 1',
        powershell(command),
        '$code = $LASTEXITCODE',
        *([f'Remove-Item -LiteralPath {_powershell_quote(str(overrides))}'] if overrides is not None else []),
        'if ($code -eq 0) { \'Updated CLAI. Start clai2 again.\' } else { "uv exited with status $code." }',
        "Read-Host 'Press Enter to close'",
    ]
    return '\n'.join(lines)


_CREATE_NEW_CONSOLE = 0x10
"""`subprocess.CREATE_NEW_CONSOLE`, which the module defines only on Windows."""


def _start_detached(argv: Sequence[str]) -> None:  # pragma: no cover -- `creationflags` exists only on Windows
    subprocess.Popen(argv, creationflags=_CREATE_NEW_CONSOLE, close_fds=True)


def install_after_exit(
    script: str,
    *,
    environ: Mapping[str, str] = os.environ,
    start: Callable[[Sequence[str]], None] = _start_detached,
) -> None:
    """Run `script` in a new console with Windows PowerShell, found by full path rather than a `PATH` search."""
    root = environ.get('SystemRoot', r'C:\Windows')
    executable = f'{root}\\System32\\WindowsPowerShell\\v1.0\\powershell.exe'
    encoded = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
    start([executable, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', encoded])


class Relaunch(SystemExit):
    """Raised by `chat` once the shell has closed after `/update`: the CLI starts `executable` in its place.

    A `SystemExit`, so a caller other than the CLI exits as it did before CLAI restarted itself.
    """

    def __init__(self, *, executable: str, session_id: str | None) -> None:
        super().__init__(0)
        self.executable = executable
        self.session_id = session_id


async def tool_executable(uv: str) -> str | None:
    """The `clai2` that `uv tool install` just wrote to uv's tool bin directory, if it is there."""
    result = await run_process([uv, 'tool', 'dir', '--bin'], check=False)
    path = Path(result.stdout.decode().strip(), 'clai2')
    return str(path) if result.returncode == 0 and path.is_file() else None


def _in_thread(work: Callable[[], None]) -> None:
    threading.Thread(target=work, name='clai-update-check', daemon=True).start()


async def _run_uv(command: Sequence[str]) -> int:
    # Inherit the terminal so uv's progress shows; slash commands run with the editor suspended.
    result = await run_process(command, stdout=None, stderr=None, check=False)
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
    run: Callable[[Sequence[str]], Awaitable[int]] = _run_uv
    find_uv: Callable[[], str | None] = find_uv
    windows: bool = os.name == 'nt'
    hand_off: Callable[[str], None] = install_after_exit
    """Starts the Windows install that runs after CLAI exits."""
    locate: Callable[[str], Awaitable[str | None]] = tool_executable
    """Finds the installed `clai2` from the uv executable."""
    restart_required: bool = False
    """Set after a successful install: the running environment was replaced, so the shell exits."""
    relaunch: str | None = None
    """The new `clai2` to start once the shell has exited, when the install found one."""
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
        """Install the newest CLAI on the current channel with `uv tool install`, then restart into it."""
        if args:
            raise ValueError('Usage: /update. Pick the channel with /set updates.channel stable|main.')
        channel = self.channel()
        update = find_update(channel, self.current, await to_thread.run_sync(self.fetch, channel))
        if update is None:
            return f'CLAI {self.current.label} is the newest on the {channel} channel.'
        uv = self.find_uv()
        overrides = _write_overrides(update.overrides()) if channel == 'main' else None
        command = update.command(uv=uv or 'uv', overrides=overrides)
        if uv is None:
            # Keep the overrides file: the printed command reads it.
            shown = powershell(command) if self.windows else shlex.join(command)
            return (
                f'CLAI {update.label} is available on the {channel} channel. CLAI updates itself only when uv is '
                f'on PATH. To update by hand, run:\n{shown}'
            )
        if self.windows:
            # The script removes the overrides file once uv has read it.
            self.hand_off(after_exit_script(command, pid=os.getpid(), overrides=overrides))
            self.restart_required = True
            return (
                f'Installing CLAI {update.label} ({channel}) in a new window once CLAI exits. '
                'Exiting; start clai2 again when it finishes.'
            )
        # Cancelling uv halfway could leave the environment half replaced, so let it finish.
        code: int | None = None
        try:
            with CancelScope(shield=True):
                code = await self.run(command)
        finally:
            if overrides is not None:
                overrides.unlink(missing_ok=True)
        if code != 0:
            return f'uv exited with status {code}; see its output above.'
        self.restart_required = True
        self.relaunch = await self.locate(uv)
        if self.relaunch is None:
            return f'Updated CLAI to {update.label} ({channel}). Exiting; start clai2 again to use it.'
        return f'Updated CLAI to {update.label} ({channel}). Restarting...'
