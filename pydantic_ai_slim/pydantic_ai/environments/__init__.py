"""Execution environment abstractions for agents.

This package provides:

- `ExecutionEnvironment` — abstract base class for execution environments
- `ExecutionProcess` — interactive process handle with bidirectional I/O
- `ExecutionEnvironmentToolset` — toolset exposing coding-agent-style tools backed by an environment
- `ExecuteResult`, `FileInfo` — result types

Implementations:

- `environments.docker.DockerEnvironment` — Docker container-based sandbox (isolated)
- `environments.local.LocalEnvironment` — local subprocess environment (no isolation, for dev/testing)
- `environments.e2b.E2BEnvironment` — hosted sandbox via E2B (isolated)
- `environments.memory.MemoryEnvironment` — in-memory environment for testing
- `environments.monty.MontyEnvironment` — Monty sandboxed interpreter for code execution
"""

from ._base import ExecuteResult, ExecutionEnvironment, ExecutionProcess, FileInfo
from ._toolset import ExecutionEnvironmentToolset

__all__ = (
    'ExecuteResult',
    'ExecutionEnvironment',
    'ExecutionEnvironmentToolset',
    'ExecutionProcess',
    'FileInfo',
)
