"""Execution environment abstractions for agents.

This package provides:

- `ExecutionEnvironment` — abstract base class for execution environments
- `ExecutionProcess` — interactive process handle with bidirectional I/O
- `ExecutionEnvironmentToolset` — toolset exposing coding-agent-style tools backed by an environment
- `ExecutionResult`, `FileInfo` — result types

Implementations:

- `environments.docker.DockerEnvironment` — Docker container-based sandbox (isolated)
- `environments.local.LocalEnvironment` — local subprocess environment (no isolation, for dev/testing)
- `environments.memory.MemoryEnvironment` — in-memory environment for testing
"""

from pydantic_ai.toolsets.execution_environment import ExecutionEnvironmentToolset

from ._base import EnvToolName, ExecutionEnvironment, ExecutionProcess, ExecutionResult, FileInfo

__all__ = (
    'EnvToolName',
    'ExecutionResult',
    'ExecutionEnvironment',
    'ExecutionEnvironmentToolset',
    'ExecutionProcess',
    'FileInfo',
)
