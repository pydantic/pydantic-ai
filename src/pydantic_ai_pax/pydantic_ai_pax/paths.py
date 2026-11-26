"""Pax's folders under the user's config and state directories, moved from their names before the rename.

Import-light, so the entry point can read the splash setting before loading anything else.
"""

import os
from contextlib import suppress
from functools import cache
from pathlib import Path

DIRECTORY_NAME = 'pydantic-ai-pax'
"""The folder Pax keeps its files in, under `$XDG_CONFIG_HOME` and under `$XDG_STATE_HOME`."""

LEGACY_DIRECTORY_NAME = 'pydantic-clai2'
"""The folder's name before the package became `pydantic-ai-pax`; `pax_dir` moves it to `DIRECTORY_NAME`."""


def config_dir() -> Path:
    """The user's Pax folder, honouring `XDG_CONFIG_HOME`."""
    return pax_dir(Path(os.getenv('XDG_CONFIG_HOME', str(Path.home() / '.config'))))


def state_dir() -> Path:
    """Where Pax keeps state that is not a preference, honouring an absolute `XDG_STATE_HOME`."""
    root = os.environ.get('XDG_STATE_HOME', '')
    return pax_dir(Path(root) if os.path.isabs(root) else Path.home() / '.local' / 'state')


@cache
def pax_dir(parent: Path) -> Path:
    """Pax's folder in `parent`, after moving a folder under the old name there, once per process.

    A folder under the old name that is itself a symlink stays where it points, and this process uses it.
    Otherwise the old path is left as a symlink to the new folder, so an older Pax that is still installed or running
    finds the same files instead of starting an empty folder. An SQLite database another process holds open
    keeps working, as a rename on one filesystem keeps its files, `-wal` and `-shm` included.

    When another Pax moves the folder at the same time, this one uses the moved folder. When the move fails for
    another reason, such as a `parent` that spans filesystems or is not writable, this process keeps using the
    old folder, so its data is never split between the two.
    """
    current = parent / DIRECTORY_NAME
    legacy = parent / LEGACY_DIRECTORY_NAME
    if current.exists() or not legacy.is_dir():
        return current
    if legacy.is_symlink():
        # The user's own link, such as one into a dotfiles checkout: moving the link would not move the files.
        return legacy
    try:
        os.rename(legacy, current)
    except OSError:
        # Another Pax moved it first: the folder is there, or the old one is already gone.
        return current if current.exists() or not legacy.exists() else legacy
    # Without the link, such as where symlinks need privileges, an older Pax starts an empty folder of its own.
    with suppress(OSError):
        legacy.symlink_to(DIRECTORY_NAME, target_is_directory=True)
    return current
