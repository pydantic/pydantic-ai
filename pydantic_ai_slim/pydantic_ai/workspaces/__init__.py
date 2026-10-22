"""Workspace API, backend protocols, and implementations."""

from .computer import (
    Click,
    ComputerAction,
    Drag,
    HoldKeys,
    MouseButton,
    MouseDown,
    MouseUp,
    MoveMouse,
    Point,
    PressKeys,
    Scroll,
    TypeText,
)
from .local import LocalWorkspaceBackend
from .protocol import (
    CommandResult,
    FileEntry,
    SupportsCommands,
    SupportsComputerUse,
    SupportsFilesystem,
    SupportsRealpath,
    UnsupportedComputerActionError,
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
from .x11 import X11Computer

__all__ = (
    'Click',
    'CommandResult',
    'ComputerAction',
    'Drag',
    'FileEntry',
    'HoldKeys',
    'PressKeys',
    'LocalWorkspaceBackend',
    'MouseButton',
    'MouseDown',
    'MouseUp',
    'MoveMouse',
    'Point',
    'ReadOnlyWorkspace',
    'Scroll',
    'TypeText',
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
    'SupportsComputerUse',
    'SupportsFilesystem',
    'SupportsRealpath',
    'UnavailableWorkspace',
    'UnsupportedComputerActionError',
    'X11Computer',
)
