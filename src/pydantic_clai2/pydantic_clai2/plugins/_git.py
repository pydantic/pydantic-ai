"""Install an explicitly trusted Git repository as a file-based plugin."""

import os
import re
import shutil
from collections.abc import Collection
from pathlib import Path
from urllib.parse import unquote, urlsplit

from anyio import fail_after, run_process

from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.settings_store import canonical_plugin_id

_SCP_URL = re.compile(r'(?:[\w.-]+@)?[\w.-]+:[^:\s].*')
_PLUGIN_ID = re.compile(r'[A-Za-z][A-Za-z0-9_]*')


def parse_repository(source: str) -> tuple[str, str]:
    """Validate a Git transport and derive a safe plugin ID from the repository name."""
    url = source.removeprefix('git+')
    if '://' in url:
        parsed = urlsplit(url)
        if (
            parsed.scheme not in ('https', 'http', 'ssh', 'git', 'file')
            or (parsed.scheme != 'file' and not parsed.netloc)
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError('Use a Git HTTPS, SSH, git://, or file:// URL without a query or fragment.')
        path = unquote(parsed.path)
    elif _SCP_URL.fullmatch(url):
        path = url.partition(':')[2]
    else:
        raise ValueError('Usage: /plugins add GIT_URL or /plugins add ID MODULE[:ATTR] [JSON]')
    name = path.rstrip('/').rsplit('/', 1)[-1].removesuffix('.git').replace('-', '_').replace('.', '_')
    if _PLUGIN_ID.fullmatch(name) is None:
        raise ValueError(
            'The Git repository name must start with a letter and contain only letters, digits, dots, hyphens, or underscores.'
        )
    return url, canonical_plugin_id(name)


async def install_git_plugin(source: str, *, plugins_dir: Path, names: Collection[str]) -> PluginSettings:
    """Clone a repository without importing it; failed or cancelled clones leave no installation."""
    url, name = parse_repository(source)
    if name in names:
        raise ValueError(f'Plugin {name} already exists; choose a repository with a different name.')
    # Keep incomplete checkouts out of drop-in discovery, including in other running CLAI sessions.
    destination = plugins_dir.resolve() / '_git' / name
    try:
        destination.mkdir(parents=True)
    except FileExistsError as exc:
        raise ValueError(f'Plugin checkout already exists at {destination}; it has not been changed.') from exc
    try:
        try:
            with fail_after(120):
                result = await run_process(
                    ['git', '-c', 'credential.interactive=false', 'clone', '--depth', '1', '--', url, str(destination)],
                    env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'},
                    start_new_session=True,
                    check=False,
                )
        except FileNotFoundError as exc:
            raise ValueError(
                'Git is required to install a plugin from a repository. Install git and try again.'
            ) from exc
        except TimeoutError as exc:
            raise ValueError('Cloning the plugin repository timed out after 120 seconds.') from exc
        if result.returncode:
            detail = result.stderr.decode(errors='replace').strip()
            raise ValueError(f'Could not clone the plugin repository: {detail}')
        entry = next(
            (path for filename in ('__init__.py', 'plugin.py') if (path := destination / filename).is_file()),
            None,
        )
        if entry is None or entry.is_symlink():
            raise ValueError('A plugin repository must contain a regular __init__.py or plugin.py at its root.')
        return PluginSettings(id=name, factory=name, path=str(entry))
    except BaseException:
        shutil.rmtree(destination)
        raise
