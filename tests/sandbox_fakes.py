from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, cast

from pydantic_ai import RunPreparationContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.sandboxes import SandboxBackend, SandboxProvider


@dataclass(frozen=True)
class FakeSandboxResult:
    exit_code: int = 0
    stdout: str = 'connected'
    stderr: str = ''
    stdout_dropped: int = 0
    stderr_dropped: int = 0


class FakeSandboxHandle:
    """Minimal sandbox identity for paths that do not touch backend operations."""

    provider = 'fake'

    def __init__(self, sandbox_id: str = 'fake-sandbox') -> None:
        self.sandbox_id = sandbox_id


class RecordingSandboxBackend:
    provider = 'fake'

    def __init__(self, sandbox_id: str) -> None:
        self.sandbox_id = sandbox_id
        self.commands: list[str | Sequence[str]] = []

    async def run(
        self,
        command: str | Sequence[str],
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        output_limit: int | None = None,
    ) -> FakeSandboxResult:
        self.commands.append(command)
        return FakeSandboxResult()

    async def working_dir(self) -> str:
        return '/workspace'


class RecordingSandboxProvider(SandboxProvider):
    def __init__(self) -> None:
        self.sandbox_ids: list[str] = []
        self.backends: list[RecordingSandboxBackend] = []

    @property
    def provider(self) -> str:
        return 'fake'

    async def connect(self, sandbox_id: str) -> SandboxBackend:
        self.sandbox_ids.append(sandbox_id)
        backend = RecordingSandboxBackend(sandbox_id)
        self.backends.append(backend)
        return backend


class CreateOnlySandboxProvider(SandboxProvider):
    """Provisions and reconnects, inheriting `SandboxProvider`'s no-op `teardown`.

    Every lifecycle call is appended to `events`, so tests can pin both the counts and the
    order in which creation, connection, and destruction happened.
    """

    def __init__(self) -> None:
        self.events: list[str] = []
        self.backends: list[RecordingSandboxBackend] = []
        self._created = 0

    @property
    def provider(self) -> str:
        return 'fake'

    async def create(self) -> SandboxBackend:
        self._created += 1
        return self._backend('create', f'created-{self._created}')

    async def connect(self, sandbox_id: str) -> SandboxBackend:
        return self._backend('connect', sandbox_id)

    def _backend(self, event: str, sandbox_id: str) -> RecordingSandboxBackend:
        self.events.append(f'{event}:{sandbox_id}')
        backend = RecordingSandboxBackend(sandbox_id)
        self.backends.append(backend)
        return backend


class LifecycleSandboxProvider(CreateOnlySandboxProvider):
    """A `CreateOnlySandboxProvider` that also destroys the sandboxes it made."""

    async def teardown(self, sandbox_id: str) -> None:
        self.events.append(f'teardown:{sandbox_id}')


class FailingTeardownSandboxProvider(CreateOnlySandboxProvider):
    """A provider whose `teardown` always fails, e.g. because the sandbox is already gone."""

    async def teardown(self, sandbox_id: str) -> None:
        self.events.append(f'teardown-failed:{sandbox_id}')
        raise RuntimeError(f'sandbox {sandbox_id!r} is already gone')


class SandboxContributingCapability(AbstractCapability[Any]):
    """Capability whose sandbox contribution is rejected before the handle is used."""

    def get_sandbox(self, ctx: RunPreparationContext[Any]) -> AbstractAsyncContextManager[SandboxBackend]:
        return nullcontext(cast(SandboxBackend, FakeSandboxHandle()))  # pragma: no cover
