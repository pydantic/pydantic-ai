"""Where `/mcp` keeps servers: a user file, plus a project file that loads only once trusted.

A stdio server runs a program, so a cloned repository must not be able to start one by
shipping `.clai/mcp_servers.json`. Trust is recorded on the user side, keyed by the file's
path and a SHA-256 of its bytes: any edit makes the file untrusted again, and a repository
cannot trust itself. A symlinked file (or `.clai` folder) is never trusted, so a repository
cannot borrow trust given to a file elsewhere. This follows Code Puppy's `/mcp trust`.
"""

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..credential_store import write_private
from ..project_settings import find_project_file
from ..settings_store import config_dir
from ._settings import HTTPServer, Server, Servers, SSEServer, StdioServer

PROJECT_MCP_FILE = Path('.clai') / 'mcp_servers.json'
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

    def project_file(self) -> Path | None:
        """The nearest `.clai/mcp_servers.json` between the workspace and the git root."""
        return find_project_file(self._workspace or Path.cwd(), PROJECT_MCP_FILE)

    def trust_state(self, path: Path) -> TrustState:
        """Whether the file's current bytes are the ones the user accepted."""
        accepted = self.load().trusted_projects.get(_key(path))
        if accepted is None or not _regular(path):
            return 'untrusted'
        try:
            current = _digest(path.read_bytes())
        except OSError:
            return 'changed'
        return 'trusted' if accepted == current else 'changed'

    def trust(self, path: Path) -> None:
        """Accept the file's current bytes."""
        if not _regular(path):
            raise ValueError(f'{path} is a symlink; only a file inside the repository can be trusted.')
        data = self.load()
        trusted = {**data.trusted_projects, _key(path): _digest(path.read_bytes())}
        self.save(data.model_copy(update={'trusted_projects': trusted}))

    def revoke(self, path: Path) -> bool:
        """Withdraw acceptance; `False` when it was never given."""
        data = self.load()
        key = _key(path)
        if key not in data.trusted_projects:
            return False
        trusted = {k: v for k, v in data.trusted_projects.items() if k != key}
        self.save(data.model_copy(update={'trusted_projects': trusted}))
        return True

    def project_servers(self) -> Servers:
        """The project file's servers when trusted; empty when absent or not trusted.

        The bytes are read once, so the file cannot be swapped between the hash check and parsing.
        """
        path = self.project_file()
        if path is None or not _regular(path):
            return {}
        try:
            content = path.read_bytes()
        except OSError:
            return {}
        if self.load().trusted_projects.get(_key(path)) != _digest(content):
            return {}
        try:
            return ProjectFile.model_validate_json(content).servers
        except ValidationError as exc:
            raise ValueError(f'{path}: {exc}') from exc


def _key(path: Path) -> str:
    return str(path.absolute())


def _regular(path: Path) -> bool:
    return not (path.is_symlink() or path.parent.is_symlink())


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
