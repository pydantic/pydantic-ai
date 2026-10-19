"""The live workspaces, one folder each under CLAI's state directory, readable by you alone.

`$XDG_STATE_HOME/pydantic-clai2/workspaces/registry/ID/` holds `workspace.json` (the record), and while
its CLAI runs, `token` (that CLAI's access token) and `session.log` (its output). Every CLAI process
reads the same folders, so a workspace made in one session is listed, opened, and destroyed from another.
"""

import os
import re
import secrets
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import TypeAdapter, ValidationError

from pydantic_clai2.builtin_plugins.workspaces.kinds import Teardown
from pydantic_clai2.config.settings_store import state_dir

WorkspaceStatus = Literal['ready', 'running', 'stopped']
"""`ready` until its CLAI first starts, `running` while it runs, `stopped` once it ended."""

_ID = re.compile(r'[a-z0-9][a-z0-9-]{0,47}')


def workspaces_dir() -> Path:
    """Where the workspaces plugin keeps its registry and the clones it made."""
    return state_dir() / 'workspaces'


def check_id(workspace_id: str) -> str:
    """`workspace_id` when it can name a workspace, so it is one safe path component; `LookupError` otherwise."""
    if not _ID.fullmatch(workspace_id):
        raise LookupError(f'No workspace {workspace_id!r}.')
    return workspace_id


@dataclass(frozen=True, kw_only=True)
class SessionProcess:
    """A workspace's CLAI: a `clai2 --web` serving on loopback, with its own token kept in `token_file`."""

    pid: int
    port: int
    token_file: Path
    log: Path
    started: datetime

    @property
    def url(self) -> str:
        """The page's address, without the token."""
        return f'http://127.0.0.1:{self.port}/'


@dataclass(frozen=True, kw_only=True)
class Workspace:
    """One workspace: where its checkout is, what it was made from, and its CLAI while one runs."""

    id: str
    kind: str
    backend: Literal['local-worktree']
    repo: str
    """The repository as given: a local path or a URL."""
    ref: str
    """The branch, tag, or commit it started from, as given; empty for the repository's `HEAD`."""
    branch: str
    path: Path
    root: Path
    """The folder its checkout was made in; destroying it removes nothing outside."""
    status: WorkspaceStatus
    created: datetime
    teardown: Teardown
    model: str = ''
    """The model its CLAI starts with; empty for your saved default."""
    session: SessionProcess | None = None


_RECORD = TypeAdapter(Workspace)


class Registry:
    """Workspace records on disk, one folder each, written atomically and readable by you alone."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def folder(self, workspace_id: str) -> Path:
        """The folder of a workspace's record, token, and log."""
        return self.directory / check_id(workspace_id)

    def save(self, workspace: Workspace) -> None:
        """Write the record, replacing the last one."""
        folder = self.folder(workspace.id)
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        write_private(folder / 'workspace.json', _RECORD.dump_json(workspace, indent=2).decode())

    def get(self, workspace_id: str) -> Workspace:
        """A workspace's record; `LookupError` for an unknown or unreadable one."""
        try:
            return _RECORD.validate_json((self.folder(workspace_id) / 'workspace.json').read_bytes())
        except (OSError, ValidationError) as exc:
            raise LookupError(f'No workspace {workspace_id!r}.') from exc

    def all(self) -> list[Workspace]:
        """Every readable record, oldest first; a folder another build wrote in a shape this one cannot read is skipped."""
        if not self.directory.is_dir():
            return []
        found: list[Workspace] = []
        for folder in self.directory.iterdir():
            try:
                found.append(self.get(folder.name))
            except LookupError:
                continue
        return sorted(found, key=lambda workspace: workspace.created)

    def forget(self, workspace_id: str) -> None:
        """Delete a workspace's record, token, and log."""
        shutil.rmtree(self.folder(workspace_id), ignore_errors=True)


def write_private(path: Path, text: str) -> None:
    """Replace `path` with `text` in a file only you can read, through a fresh temporary file renamed over it."""
    temporary = path.with_name(f'.{path.name}.{secrets.token_hex(8)}.tmp')
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as file:
            file.write(text)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
