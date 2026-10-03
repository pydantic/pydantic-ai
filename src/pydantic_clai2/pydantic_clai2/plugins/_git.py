"""Install an explicitly trusted Git repository as a file-based plugin."""

import asyncio
import os
import re
import shutil
import subprocess
from collections.abc import AsyncGenerator, Collection
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import unquote, urlsplit

from anyio import CancelScope, fail_after

from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.settings_store import canonical_plugin_id
from pydantic_clai2.runtime._processes import kill_process_tree

ADD_USAGE = 'Usage: /plugins add GIT_URL or /plugins add ID MODULE[:ATTR] [JSON]'
CHECKOUTS_DIR = '_git'

_SCP_URL = re.compile(r'(?:[\w.-]+@)?[\w.-]+:[^:\s].*')
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
        shutil.rmtree(destination)
        raise


async def clone_repository(url: str, destination: Path) -> tuple[int, bytes]:
    """Own Git's process group until cloning finishes or cancellation cleanup completes."""
    spawn = asyncio.create_task(
        asyncio.create_subprocess_exec(
            'git',
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
