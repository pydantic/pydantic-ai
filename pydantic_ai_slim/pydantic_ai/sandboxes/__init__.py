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
    SupportsStart,
    SupportsStream,
)
from .references import SandboxProvider, SandboxRef
from .sandbox import FileWindow, Sandbox
from .unavailable import UnavailableSandbox

__all__ = (
    'FileWindow',
    'LocalSandbox',
    'Sandbox',
    'SandboxBackend',
    'SandboxProvider',
    'SandboxCommand',
    'SandboxFileEntry',
    'SandboxFilesystem',
    'SandboxOutputChunk',
    'SandboxProcess',
    'SandboxResult',
    'SandboxRef',
    'SupportsFilesystem',
    'SupportsStart',
    'SupportsStream',
    'UnavailableSandbox',
)
