"""Workspace API, backend protocols, and implementations."""

from .local import LocalWorkspace
from .protocol import (
    CommandResult,
    FileEntry,
    SupportsFilesystem,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceFileEntry,
    WorkspaceRef,
    WorkspaceResult,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from .readonly import ReadOnlyWorkspace
from .unavailable import UnavailableWorkspace
from .workspace import FileWindow, Workspace, WrapperWorkspace

__all__ = (
    'CommandResult',
    'FileEntry',
    'FileWindow',
    'LocalWorkspace',
    'ReadOnlyWorkspace',
    'Workspace',
    'WrapperWorkspace',
    'WorkspaceBackend',
    'WorkspaceCommand',
    'WorkspaceError',
    'WorkspaceFileEntry',
    'WorkspaceRef',
    'WorkspaceResult',
    'WorkspaceTimeoutError',
    'WorkspaceUnavailableError',
    'SupportsFilesystem',
    'UnavailableWorkspace',
)
