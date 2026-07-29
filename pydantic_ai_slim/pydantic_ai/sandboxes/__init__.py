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
    SupportsReadBytesRange,
)
from .sandbox import FileWindow, Sandbox

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
    'SandboxResult',
    'SupportsReadBytesRange',
)
