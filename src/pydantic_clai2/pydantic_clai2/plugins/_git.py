"""Install an explicitly trusted Git repository as a file-based plugin."""

import asyncio
import os
import re
import shutil
import stat
import subprocess
import sys
from collections.abc import AsyncGenerator, Callable, Collection
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from types import TracebackType
from urllib.parse import unquote, urlsplit

from anyio import CancelScope, fail_after

from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.settings_store import canonical_plugin_id
from pydantic_clai2.runtime._processes import kill_process_tree

ADD_USAGE = 'Usage: /plugins add GIT_URL or /plugins add ID MODULE[:ATTR] [JSON]'
CHECKOUTS_DIR = '_git'

_SCP_URL = re.compile(r'[\w.-]+@[\w.-]+:[^:\s].*')
_PLUGIN_ID = re.compile(r'[A-Za-z][A-Za-z0-9_]*')


def parse_repository(source: str) -> tuple[str, str]:
    """Validate a Git transport and derive a safe plugin ID from the repository name."""
    url = source.removeprefix('git+')
    if '://' in url:
        parsed = urlsplit(url)
        if (
            parsed.scheme not in ('https', 'ssh', 'file')
            or (parsed.scheme != 'file' and not parsed.netloc)
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError('Use an HTTPS, SSH, or local file:// Git URL without a query or fragment.')
        path = unquote(parsed.path)
    elif _SCP_URL.fullmatch(url):
        path = url.partition(':')[2]
    else:
        raise ValueError(ADD_USAGE)
    name = path.rstrip('/').rsplit('/', 1)[-1].removesuffix('.git').replace('-', '_').replace('.', '_')
    if _PLUGIN_ID.fullmatch(name) is None:
        raise ValueError(
            'The Git repository name must start with a letter and contain only letters, digits, dots, hyphens, or underscores.'
        )
    return url, canonical_plugin_id(name)


def checkout_dir(plugins_dir: Path, name: str) -> Path:
    """The managed checkout directory for a plugin ID."""
    return plugins_dir.resolve() / CHECKOUTS_DIR / name


@asynccontextmanager
async def install_git_plugin(
    source: str, *, plugins_dir: Path, names: Collection[str]
) -> AsyncGenerator[PluginSettings, None]:
    """Clone without importing; roll back unless the caller successfully saves its declaration."""
    url, name = parse_repository(source)
    if name in names:
        raise ValueError(f'Plugin {name} already exists; choose a repository with a different name.')
    # Keep incomplete checkouts out of drop-in discovery, including in other running CLAI sessions.
    destination = checkout_dir(plugins_dir, name)
    try:
        destination.mkdir(parents=True)
    except FileExistsError as exc:
        raise ValueError(f'Plugin checkout already exists at {destination}; it has not been changed.') from exc
    try:
        try:
            with fail_after(120):
                returncode, stderr = await clone_repository(url, destination)
        except FileNotFoundError as exc:
            raise ValueError(
                'Git is required to install a plugin from a repository. Install git and try again.'
            ) from exc
        except TimeoutError as exc:
            raise ValueError('Cloning the plugin repository timed out after 120 seconds.') from exc
        if returncode:
            detail = stderr.decode(errors='replace').strip()
            raise ValueError(f'Could not clone the plugin repository: {detail}')
        entry = next(
            (path for filename in ('__init__.py', 'plugin.py') if (path := destination / filename).is_file()),
            None,
        )
        if entry is None or entry.is_symlink():
            raise ValueError('A plugin repository must contain a regular __init__.py or plugin.py at its root.')
        yield PluginSettings(id=name, factory=name, path=str(entry))
    except BaseException:
        # Cleanup must not replace the installation error or cancellation.
        with suppress(OSError):
            remove_checkout(destination)
        raise


def remove_checkout(destination: Path, *, windows: bool = os.name == 'nt') -> None:
    """Remove an incomplete checkout, including Git's read-only object files on Windows."""
    if not windows:
        shutil.rmtree(destination)
    elif sys.version_info >= (3, 12):
        shutil.rmtree(destination, onexc=retry_readonly)
    else:
        shutil.rmtree(destination, onerror=retry_readonly_legacy)


def retry_readonly(function: Callable[[str], object], path: str, error: BaseException) -> None:
    """Retry a failed removal after clearing a regular file or directory's read-only bit."""
    if not isinstance(error, PermissionError) or function not in (os.unlink, os.rmdir) or os.path.islink(path):
        raise error
    os.chmod(path, stat.S_IWRITE)
    function(path)


def retry_readonly_legacy(
    function: Callable[[str], object], path: str, error: tuple[type[BaseException], BaseException, TracebackType]
) -> None:
    """Adapt Python 3.11's `rmtree` callback to the exception-based one."""
    retry_readonly(function, path, error[1])


def git_executable(*, windows: bool = os.name == 'nt') -> str:
    """Resolve Windows' native `git.exe` from absolute PATH directories outside the working directory.

    Bare executable names, and `shutil.which` before Python 3.12, search the working directory first.
    Keep POSIX's existing PATH lookup unchanged.
    """
    if not windows:
        return 'git'
    current = Path.cwd().resolve()
    for directory in os.get_exec_path():
        path = Path(directory)
        if not path.is_absolute() or not path.is_dir():
            continue
        path = path.resolve()
        if path == current:
            continue
        executable = path / 'git.exe'
        if executable.is_file():
            return str(executable)
    raise FileNotFoundError('git.exe was not found in an absolute PATH directory outside the working directory.')


async def clone_repository(url: str, destination: Path) -> tuple[int, bytes]:
    """Own Git's process group until cloning finishes or cancellation cleanup completes."""
    spawn = asyncio.create_task(
        asyncio.create_subprocess_exec(
            git_executable(),
            '-c',
            'credential.interactive=false',
            'clone',
            '--depth',
            '1',
            '--',
            url,
            str(destination),
            env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'},
            start_new_session=True,
            # Git must not mistake a workspace path resembling the URL for the trusted repository.
            cwd=destination,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        ),
        name='clai-plugin-clone',
    )
    try:
        process = await asyncio.shield(spawn)
        _, stderr = await process.communicate()
    except BaseException:
        with CancelScope(shield=True):
            process = await spawn
            await kill_process_tree(process)
            await process.communicate()
        raise
    assert process.returncode is not None and stderr is not None
    return process.returncode, stderr
