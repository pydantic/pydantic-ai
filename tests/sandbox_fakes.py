from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.sandboxes import SandboxBackend, SandboxRef


@dataclass(frozen=True)
class FakeSandboxResult:
    exit_code: int = 0
    stdout: str = 'connected'
    stderr: str = ''


class FakeSandboxHandle:
    """Sandbox identity for paths that must never execute operations: doing so fails loudly."""

    provider = 'fake'

    def __init__(self, sandbox_id: str = 'fake-sandbox') -> None:
        self.sandbox_id = sandbox_id

    async def run(
        self,
        command: str | Sequence[str],
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> FakeSandboxResult:
        raise AssertionError('FakeSandboxHandle must not execute commands')  # pragma: no cover

    async def working_dir(self) -> str:
        raise AssertionError('FakeSandboxHandle must not execute operations')  # pragma: no cover


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
    ) -> FakeSandboxResult:
        self.commands.append(command)
        return FakeSandboxResult()

    async def working_dir(self) -> str:
        return '/workspace'


class ConnectOnlySandboxCapability(AbstractCapability[Any]):
    """Recognizes `'fake'` refs and connects to them; never provisions anything.

    The connect-only shape of the lifecycle hooks: only `get_sandbox` is overridden, so this
    capability serves `SandboxRef` run arguments (and refs provisioned elsewhere) without ever
    owning a lifecycle.
    """

    def __init__(self) -> None:
        self.sandbox_ids: list[str] = []
        self.backends: list[RecordingSandboxBackend] = []

    def reset(self) -> None:
        """Restore pristine state, so module-level capabilities can be shared across tests."""
        self.sandbox_ids.clear()
        self.backends.clear()

    async def get_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> SandboxBackend | None:
        if ref.provider != 'fake':
            return None
        self.sandbox_ids.append(ref.sandbox_id)
        backend = RecordingSandboxBackend(ref.sandbox_id)
        self.backends.append(backend)
        return backend


class AcquireOnlySandboxCapability(AbstractCapability[Any]):
    """Provisions per run and reconnects, inheriting the no-op `release_sandbox`.

    Every lifecycle call is appended to `events`, so tests can pin both the counts and the
    order in which acquisition, connection, and release happened.
    """

    id = 'test-sandbox'

    def __init__(self) -> None:
        self.events: list[str] = []
        self.backends: list[RecordingSandboxBackend] = []
        self._created = 0

    def reset(self) -> None:
        """Restore pristine state, so module-level capabilities can be shared across tests."""
        self.events.clear()
        self.backends.clear()
        self._created = 0

    async def acquire_sandbox(self, ctx: RunContext[Any]) -> SandboxRef:
        self._created += 1
        sandbox_id = f'created-{self._created}'
        self.events.append(f'acquire:{sandbox_id}')
        return SandboxRef(provider='fake', sandbox_id=sandbox_id)

    async def get_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> SandboxBackend | None:
        if ref.provider != 'fake':
            return None
        self.events.append(f'connect:{ref.sandbox_id}')
        backend = RecordingSandboxBackend(ref.sandbox_id)
        self.backends.append(backend)
        return backend


class LifecycleSandboxCapability(AcquireOnlySandboxCapability):
    """An `AcquireOnlySandboxCapability` that also releases its sandbox leases."""

    async def release_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> None:
        self.events.append(f'release:{ref.sandbox_id}')


class PerRunLifecycleSandboxCapability(LifecycleSandboxCapability):
    """Rebuild the lifecycle supplier from `RunContext` while sharing test observations."""

    def __init__(self, events: list[str] | None = None) -> None:
        super().__init__()
        if events is not None:
            self.events = events

    async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
        return PerRunLifecycleSandboxCapability(self.events)


class DecliningSandboxCapability(AbstractCapability[Any]):
    """A supplier that overrides `acquire_sandbox` but declines every run.

    Declining is a first-class supplier shape (contribute only for some runs), so tests use
    the call count to pin that the fall-through to the next supplier happened exactly once,
    in the right place.
    """

    def __init__(self) -> None:
        self.acquire_calls = 0

    def reset(self) -> None:
        """Restore pristine state, so module-level capabilities can be shared across tests."""
        self.acquire_calls = 0

    async def acquire_sandbox(self, ctx: RunContext[Any]) -> SandboxRef | None:
        self.acquire_calls += 1
        return None


class FailingReleaseSandboxCapability(AcquireOnlySandboxCapability):
    """A capability whose `release_sandbox` always fails, e.g. because the sandbox is already gone."""

    async def release_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> None:
        self.events.append(f'release-failed:{ref.sandbox_id}')
        raise RuntimeError(f'sandbox {ref.sandbox_id!r} is already gone')


class SandboxContributingCapability(AbstractCapability[Any]):
    """Capability whose sandbox contribution is rejected before anything is provisioned."""

    async def acquire_sandbox(self, ctx: RunContext[Any]) -> SandboxRef:
        return SandboxRef(provider='fake', sandbox_id='fake-sandbox')  # pragma: no cover
