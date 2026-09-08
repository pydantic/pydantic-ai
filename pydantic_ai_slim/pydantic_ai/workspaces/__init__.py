"""Workspace API, backend protocols, and implementations."""

from ._lazy import LazyWorkspace
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
from .workspace import FileWindow, Workspace

__all__ = (
    'CommandResult',
    'FileEntry',
    'FileWindow',
    'LazyWorkspace',
    'LocalWorkspace',
    'ReadOnlyWorkspace',
    'Workspace',
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
