"""Public-API resource accounting for the opt-in live conformance suite.

Only IDs, status, timestamps, versions, and exception class names enter the ledger.
Never serialize SDK clients, environment values, command arguments, or exceptions.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import ParamSpec, Protocol

import anyio
from render.experimental.sandbox import Sandbox, SandboxNotFoundError

_P = ParamSpec('_P')


class SandboxControl(Protocol):
    async def terminate(self, sandbox_id: str, *, owner_id: str | None = None) -> None: ...

    async def from_id(self, sandbox_id: str, *, owner_id: str | None = None) -> Sandbox: ...


@dataclass
class ResourceRecord:
    id: str
    status: str
    observed_at: str
    terminated_at: str | None = None
    cleanup_attempts: int = 0
    errors: list[str] = field(default_factory=list[str])

    @property
    def terminal(self) -> bool:
        return self.status in {'terminated', 'not_found'} or (
            self.status == 'errored' and self.terminated_at is not None
        )


class ResourceLedger:
    """Record every returned creation ID before handing it to the backend.

    A killed test process can leave this JSON for recovery. A lost creation response
    cannot supply an ID, so every live sandbox also has a bounded provider lifetime.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: dict[str, ResourceRecord] = {}
        self.started_at = datetime.now(UTC).isoformat()
        self.audit_complete = False
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            'kind': 'public-workspace-conformance-resources',
            'started_at': self.started_at,
            'versions': {name: version(name) for name in ('render', 'pydantic-ai-slim')},
            'audit_complete': self.audit_complete,
            'resources': [asdict(record) for record in self.records.values()],
        }
        temporary = self.path.with_suffix(self.path.suffix + '.tmp')
        temporary.write_text(json.dumps(document, indent=2) + '\n')
        temporary.replace(self.path)

    def record_created(self, sandbox: Sandbox) -> None:
        self.records[sandbox.id] = ResourceRecord(
            id=sandbox.id,
            status=sandbox.status,
            observed_at=datetime.now(UTC).isoformat(),
        )
        self._write()

    def recording_create(self, create: Callable[_P, Awaitable[Sandbox]]) -> Callable[_P, Awaitable[Sandbox]]:
        async def tracked(*args: _P.args, **kwargs: _P.kwargs) -> Sandbox:
            sandbox = await create(*args, **kwargs)
            # No suspension between receiving an ID and persisting it.
            self.record_created(sandbox)
            return sandbox

        return tracked

    async def terminate_owned(self, control: SandboxControl, sandbox_id: str, *, timeout: float = 30.0) -> None:
        """Terminate only IDs created by this run, then inspect public API state."""
        if sandbox_id not in self.records:
            raise ValueError('refusing to terminate a sandbox not created by this test run')
        record = self.records[sandbox_id]
        record.cleanup_attempts += 1
        self._write()
        with anyio.CancelScope(shield=True):
            try:
                with anyio.fail_after(timeout):
                    try:
                        await control.terminate(sandbox_id)
                    except SandboxNotFoundError:
                        record.status = 'not_found'
                    except Exception as exc:
                        # A failed response may follow successful termination. Audit it anyway.
                        record.errors.append(type(exc).__name__)
                    while True:
                        try:
                            sandbox = await control.from_id(sandbox_id)
                        except SandboxNotFoundError:
                            record.status = 'not_found'
                        else:
                            record.status = sandbox.status
                            record.terminated_at = (
                                sandbox.terminated_at.isoformat() if sandbox.terminated_at is not None else None
                            )
                        record.observed_at = datetime.now(UTC).isoformat()
                        self._write()
                        if record.terminal:
                            return
                        await anyio.sleep(0.25)
            except Exception as exc:
                record.errors.append(type(exc).__name__)
                self._write()
                raise RuntimeError(f'termination not confirmed for {sandbox_id}') from None

    async def cleanup_and_audit(self, control: SandboxControl, *, timeout: float = 30.0) -> None:
        """Attempt every resource even when one cleanup fails; fail teardown on leaks."""
        failed: list[str] = []
        with anyio.CancelScope(shield=True):
            for sandbox_id in self.records:
                try:
                    await self.terminate_owned(control, sandbox_id, timeout=timeout)
                except RuntimeError:
                    failed.append(sandbox_id)
            self.audit_complete = not failed
            self._write()
        if failed:
            raise RuntimeError('sandbox cleanup audit failed: ' + ', '.join(failed))
