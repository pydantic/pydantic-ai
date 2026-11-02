"""Workspace API, backend protocols, and implementations."""

from .local import LocalWorkspaceBackend
from .mounts import Mount, MountError, SupportsMounts
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
    'Mount',
    'MountError',
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
    'SupportsMounts',
    'SupportsRealpath',
    'UnavailableWorkspace',
)
