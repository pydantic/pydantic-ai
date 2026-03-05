"""Execution environment abstractions for agents.

This package provides:

- `ExecutionEnvironment` — abstract base class for execution environments
- `ExecutionProcess` — interactive process handle with bidirectional I/O
- `ExecutionResult` — result type
- `TextFileReadResult` — paginated UTF-8 file read result

Implementations:

- `environments.docker.DockerEnvironment` — Docker container-based sandbox (isolated)
- `environments.local.LocalEnvironment` — local subprocess environment (no isolation, for dev/testing)
"""

from ._base import EnvCapability, ExecutionEnvironment, ExecutionProcess, ExecutionResult, TextFileReadResult

__all__ = (
    'EnvCapability',
    'ExecutionResult',
    'TextFileReadResult',
    'ExecutionEnvironment',
    'ExecutionProcess',
)
