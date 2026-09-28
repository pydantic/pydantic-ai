"""Workspace API, backend protocols, and implementations."""

from .bubblewrap import BubblewrapWorkspace
from .local import LocalWorkspaceBackend
from .protocol import (
    CommandResult,
    FileEntry,
    SupportsCommands,
    SupportsFilesystem,
    SupportsRealpath,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceReadOnlyError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from .readonly import ReadOnlyWorkspace
from .ssh import SSHWorkspaceBackend
from .unavailable import UnavailableWorkspace
from .workspace import Workspace, WrapperWorkspace

__all__ = (
    'BubblewrapWorkspace',
    'CommandResult',
    'FileEntry',
    'LocalWorkspaceBackend',
    'ReadOnlyWorkspace',
    'SSHWorkspaceBackend',
    'Workspace',
    'WrapperWorkspace',
    'WorkspaceBackend',
    'WorkspaceCommand',
    'WorkspaceError',
    'WorkspaceOutputLimitError',
    'WorkspaceReadOnlyError',
    'WorkspaceRef',
    'WorkspaceTimeoutError',
    'WorkspaceUnavailableError',
    'SupportsCommands',
    'SupportsFilesystem',
    'SupportsRealpath',
    'UnavailableWorkspace',
)
