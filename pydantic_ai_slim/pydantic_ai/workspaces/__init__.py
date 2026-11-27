"""Workspace API, backend protocols, and implementations."""

from ._file_links import file_links
from .authority import AuthorityWorkspace
from .local import LocalWorkspaceBackend
from .protocol import (
    CommandResult,
    FileEntry,
    SupportsCommands,
    SupportsFilesystem,
    SupportsPlatform,
    SupportsRealpath,
    SupportsRunInDirectory,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspacePlatform,
    WorkspaceReadOnlyError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from .readonly import ReadOnlyWorkspace
from .unavailable import UnavailableWorkspace
from .windows import WindowsWorkspaceBackend
from .workspace import Workspace, WorkspacePaths, WrapperWorkspace, lowest_authority_under

__all__ = (
    'AuthorityWorkspace',
    'CommandResult',
    'FileEntry',
    'LocalWorkspaceBackend',
    'ReadOnlyWorkspace',
    'WindowsWorkspaceBackend',
    'Workspace',
    'WorkspacePaths',
    'WorkspacePlatform',
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
    'SupportsPlatform',
    'SupportsRealpath',
    'SupportsRunInDirectory',
    'UnavailableWorkspace',
    'file_links',
    'lowest_authority_under',
)
