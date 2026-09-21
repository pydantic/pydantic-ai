"""Workspace API, backend protocols, and implementations."""

from .composite import CompositeFilesystem, FilesystemMount
from .local import LocalWorkspace
from .protocol import (
    CommandResult,
    FileEntry,
    SupportsCommands,
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
    'CompositeFilesystem',
    'FileEntry',
    'FileWindow',
    'FilesystemMount',
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
    'SupportsCommands',
    'SupportsFilesystem',
    'UnavailableWorkspace',
)
