"""Sandbox facade, backend protocols, and implementations."""

from .local import LocalSandbox
from .protocol import (
    SandboxBackend,
    SandboxCommand,
    SandboxFileEntry,
    SandboxFilesystem,
    SandboxOutputChunk,
    SandboxProcess,
    SandboxResult,
    SupportsFilesystem,
    SupportsReadBytesRange,
    SupportsStart,
    SupportsStream,
)
from .references import SandboxConnector, SandboxRef
from .sandbox import FileWindow, Sandbox
from .unavailable import UnavailableSandbox

__all__ = (
    'FileWindow',
    'LocalSandbox',
    'Sandbox',
    'SandboxBackend',
    'SandboxConnector',
    'SandboxCommand',
    'SandboxFileEntry',
    'SandboxFilesystem',
    'SandboxOutputChunk',
    'SandboxProcess',
    'SandboxResult',
    'SandboxRef',
    'SupportsFilesystem',
    'SupportsReadBytesRange',
    'SupportsStart',
    'SupportsStream',
    'UnavailableSandbox',
)
