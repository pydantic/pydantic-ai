"""Where `/mcp` keeps servers: a user file, plus project files that load only once trusted.

A repository can ship `.clai/mcp_servers.json` and Claude Code's `.mcp.json`. A stdio server
runs a program, so a cloned repository must not be able to start one by shipping either.
Trust is recorded on the user side, keyed by the file's path and a SHA-256 of its bytes: any
edit makes the file untrusted again, and a repository cannot trust itself. A symlinked file
(or `.clai` folder) is never trusted, so a repository cannot borrow trust given to a file
elsewhere. Files are opened with `O_NOFOLLOW` where the platform has it, so a symlink swapped
in after the check is not read. This follows Code Puppy's `/mcp trust`.
"""

import hashlib
import os
import stat
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..credential_store import write_private
from ..project_settings import find_project_file
from ..settings_store import config_dir
from ._settings import HTTPServer, Server, Servers, SSEServer, StdioServer

PROJECT_MCP_FILE = Path('.clai') / 'mcp_servers.json'
CLAUDE_MCP_FILE = Path('.mcp.json')
PROJECT_MCP_FILES = (PROJECT_MCP_FILE, CLAUDE_MCP_FILE)
"""Project files in precedence order: when both define a name, the earlier file wins."""
TrustState = Literal['trusted', 'changed', 'untrusted']


class UserFile(BaseModel):
    """The user's `mcp.json`: servers and accepted project files."""

    model_config = ConfigDict(extra='forbid', frozen=True, hide_input_in_errors=True)
    servers: Servers = Field(default_factory=dict[str, StdioServer | HTTPServer | SSEServer])
    trusted_projects: dict[str, str] = Field(default_factory=dict[str, str])
    """Absolute project file path to the SHA-256 accepted by `/mcp trust`."""


class ProjectFile(BaseModel):
    """A repository's `.clai/mcp_servers.json`."""

    model_config = ConfigDict(extra='forbid', frozen=True, hide_input_in_errors=True)
    servers: Servers = Field(default_factory=dict[str, StdioServer | HTTPServer | SSEServer])


class ClaudeProjectFile(ProjectFile):
    """Claude Code's project `.mcp.json`: the same servers under `mcpServers`, `type` optional for stdio."""

    servers: Servers = Field(default_factory=dict[str, StdioServer | HTTPServer | SSEServer], alias='mcpServers')


_FORMATS: dict[Path, type[ProjectFile]] = {PROJECT_MCP_FILE: ProjectFile, CLAUDE_MCP_FILE: ClaudeProjectFile}


class MCPStore:
    """Read and write the user file; find and gate the project file."""

    def __init__(self, directory: Path | None = None, *, workspace: Path | None = None) -> None:
        """`directory` defaults to the CLAI config folder, `workspace` to the current directory."""
        directory = directory or config_dir()
        self.path = directory / 'mcp.json'
        self.logs = directory / 'mcp_logs'
        self._workspace = workspace

    def load(self) -> UserFile:
        """The saved file, or an empty one. A malformed file fails loudly with its path."""
        try:
            return UserFile.model_validate_json(self.path.read_bytes())
        except FileNotFoundError:
            return UserFile()
        except ValidationError as exc:
            raise ValueError(f'{self.path}: {exc}') from exc

    def save(self, data: UserFile) -> None:
        """Replace the file atomically, readable only by the user since headers may be sensitive."""
        write_private(path=self.path, value=data.model_dump_json(indent=2, exclude_defaults=True) + '\n')

    def put(self, name: str, server: Server) -> None:
        """Add or replace one user server."""
        data = self.load()
        self.save(data.model_copy(update={'servers': {**data.servers, name: server}}))

    def delete(self, name: str) -> bool:
        """Forget one user server; `False` when there was none."""
        data = self.load()
        if name not in data.servers:
            return False
        self.save(data.model_copy(update={'servers': {k: v for k, v in data.servers.items() if k != name}}))
        return True

    def project_files(self) -> list[Path]:
        """The nearest copy of each `PROJECT_MCP_FILES` name between the workspace and the git root."""
        return list(self._project_files())

    def _project_files(self) -> dict[Path, type[ProjectFile]]:
        workspace = self._workspace or Path.cwd()
        found = ((find_project_file(workspace, name), model) for name, model in _FORMATS.items())
        return {path: model for path, model in found if path is not None}

    def trust_state(self, path: Path) -> TrustState:
        """Whether the file's current bytes are the ones the user accepted."""
        accepted = self.load().trusted_projects.get(_key(path))
        if accepted is None or not _regular(path):
            return 'untrusted'
        try:
            current = _digest(_read_regular(path))
        except OSError:
            return 'changed'
        return 'trusted' if accepted == current else 'changed'

    def trust(self, *paths: Path) -> None:
        """Accept the files' current bytes: all of them, or none when one is a symlink."""
        if linked := [str(path) for path in paths if not _regular(path)]:
            raise ValueError(f'{", ".join(linked)}: a symlink; only a file inside the repository can be trusted.')
        data = self.load()
        try:
            digests = {_key(path): _digest(_read_regular(path)) for path in paths}
        except OSError as exc:
            raise ValueError(f'Cannot trust a project MCP file: {exc}') from exc
        trusted = {**data.trusted_projects, **digests}
        self.save(data.model_copy(update={'trusted_projects': trusted}))

    def revoke(self, *paths: Path) -> list[Path]:
        """Withdraw acceptance; the paths that had it."""
        data = self.load()
        revoked = [path for path in paths if _key(path) in data.trusted_projects]
        if revoked:
            keys = {_key(path) for path in revoked}
            trusted = {k: v for k, v in data.trusted_projects.items() if k not in keys}
            self.save(data.model_copy(update={'trusted_projects': trusted}))
        return revoked

    def project_servers(self) -> dict[Path, Servers]:
        """Each trusted project file's servers, in precedence order; absent and untrusted files are left out."""
        accepted = self.load().trusted_projects
        loaded = ((path, _read_trusted(path, model, accepted)) for path, model in self._project_files().items())
        return {path: servers for path, servers in loaded if servers is not None}


def _read_trusted(path: Path, model: type[ProjectFile], accepted: dict[str, str]) -> Servers | None:
    """The file's servers when its bytes are the accepted ones.

    The bytes are read once, so the file cannot be swapped between the hash check and parsing.
    """
    if not _regular(path):
        return None
    try:
        content = _read_regular(path)
    except OSError:
        return None
    if accepted.get(_key(path)) != _digest(content):
        return None
    try:
        return model.model_validate_json(content).servers
    except ValidationError as exc:
        raise ValueError(f'{path}: {exc}') from exc


def _key(path: Path) -> str:
    return str(path.absolute())


def _regular(path: Path) -> bool:
    return not (path.is_symlink() or path.parent.is_symlink())


_OPEN_FLAGS = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
"""`O_NONBLOCK` so opening a FIFO a repository ships returns at once instead of waiting for a writer."""


def _read_regular(path: Path) -> bytes:
    """The file's bytes; an `OSError` when it is now a symlink, or anything but a regular file."""
    with os.fdopen(os.open(path, _OPEN_FLAGS), 'rb') as file:
        if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
            raise OSError(f'{path} is not a regular file')
        return file.read()


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
