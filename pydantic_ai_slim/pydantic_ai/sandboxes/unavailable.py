"""A sandbox backend that reports why execution is unavailable.

[`UnavailableSandbox`][pydantic_ai.sandboxes.UnavailableSandbox] gives sandbox operations
the same explicit failure mode. Pydantic AI uses it where a live execution
environment cannot safely exist, and applications can pass one deliberately to disable the
default local sandbox with a policy-specific explanation.

It implements the process-start and filesystem opt-ins natively so every operation surfaces
the configured reason instead of raising `NotImplementedError` at the facade.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from typing_extensions import Never

from pydantic_ai.exceptions import UserError

from .protocol import SandboxCommand

if TYPE_CHECKING:
    from .protocol import (
        SandboxBackend,
        SupportsFilesystem,
        SupportsStart,
    )

__all__ = ('UnavailableSandbox',)


class _UnavailableFilesystem:
    def __init__(self, reason: str):
        self._reason = reason

    async def read_bytes(self, path: str) -> Never:
        raise UserError(self._reason)

    async def write_bytes(self, path: str, data: bytes) -> Never:
        raise UserError(self._reason)

    async def stat(self, path: str) -> Never:
        raise UserError(self._reason)

    async def list_dir(self, path: str) -> Never:
        raise UserError(self._reason)

    async def make_dir(self, path: str) -> Never:
        raise UserError(self._reason)

    async def remove(self, path: str) -> Never:
        raise UserError(self._reason)

    async def exists(self, path: str) -> Never:
        raise UserError(self._reason)


class UnavailableSandbox:
    """A `SandboxBackend` whose every operation raises `UserError` with a configured reason."""

    provider = 'unavailable'
    sandbox_id = 'unavailable'

    def __init__(self, reason: str):
        self.reason = reason
        self.fs = _UnavailableFilesystem(reason)

    async def run(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        output_limit: int | None = None,
    ) -> Never:
        raise UserError(self.reason)

    async def working_dir(self) -> Never:
        raise UserError(self.reason)

    async def start(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        output_limit: int | None = None,
    ) -> Never:
        raise UserError(self.reason)


if TYPE_CHECKING:
    _backend_conforms: SandboxBackend = UnavailableSandbox('')
    _start_conforms: SupportsStart = UnavailableSandbox('')
    _fs_conforms: SupportsFilesystem = UnavailableSandbox('')
