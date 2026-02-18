"""Execution environment abstractions for code execution.

This package provides:

- `ExecutionEnvironment` — abstract base class for code execution environments
- `ExecutionProcess` — interactive process handle with bidirectional I/O
- `ExecutionToolset` — toolset exposing coding-agent-style tools backed by an environment
- `ExecuteResult`, `FileInfo` — result types

Implementations:

- `environments.docker.DockerSandbox` — Docker container-based sandbox (isolated)
- `environments.local.LocalEnvironment` — local subprocess environment (no isolation, for dev/testing)
- `environments.e2b.E2BSandbox` — hosted sandbox via E2B (isolated)
- `environments.memory.MemoryEnvironment` — in-memory environment for testing
"""

from ._base import ExecuteResult, ExecutionEnvironment, ExecutionProcess, FileInfo
from ._toolset import ExecutionToolset

__all__ = (
    'ExecuteResult',
    'ExecutionEnvironment',
    'ExecutionProcess',
    'ExecutionToolset',
    'FileInfo',
)
