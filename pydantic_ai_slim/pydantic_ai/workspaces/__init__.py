"""Workspace API, backend protocols, and implementations."""

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
from .unavailable import UnavailableWorkspace
from .workspace import Workspace, WrapperWorkspace

__all__ = (
    'CommandResult',
    'FileEntry',
    'LocalWorkspaceBackend',
    'ReadOnlyWorkspace',
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
