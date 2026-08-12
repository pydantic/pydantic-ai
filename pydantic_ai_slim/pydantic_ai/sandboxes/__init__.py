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
from .references import SandboxRef
from .sandbox import FileWindow, Sandbox, SandboxResolver
from .unavailable import UnavailableSandbox

__all__ = (
    'FileWindow',
    'LocalSandbox',
    'Sandbox',
    'SandboxBackend',
    'SandboxCommand',
    'SandboxFileEntry',
    'SandboxFilesystem',
    'SandboxOutputChunk',
    'SandboxProcess',
    'SandboxRef',
    'SandboxResolver',
    'SandboxResult',
    'SupportsFilesystem',
    'SupportsStart',
    'SupportsStream',
    'UnavailableSandbox',
)
