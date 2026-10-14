"""Offline checks for live-suite resource recording and cleanup audits."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ...conftest import try_import

with try_import() as imports_successful:
    from render.experimental.sandbox import Sandbox, SandboxNotFoundError

    from .workspace_live_helpers import ResourceLedger

pytestmark = pytest.mark.skipif(not imports_successful(), reason='render not installed')


def _snapshot(sandbox_id: str, status: str = 'running') -> Sandbox:
    return Sandbox(
        sandbox_id,
        status,
        'starter',
        'deny-all',
        'oregon',
        3600,
        datetime.now(UTC),
        datetime.now(UTC) if status == 'terminated' else None,
    )


class _FakeControl:
    """Only the public terminate/from_id surface, with no HTTP or credentials."""

    def __init__(self, statuses: dict[str, str]) -> None:
        self.statuses = statuses
        self.calls: list[str] = []
        self.fail_terminate: set[str] = set()
        self.stuck: set[str] = set()
        self.gone: set[str] = set()

    async def terminate(self, sandbox_id: str, *, owner_id: str | None = None) -> None:
        self.calls.append(sandbox_id)
        if sandbox_id in self.gone:
            raise SandboxNotFoundError('already gone')
        if sandbox_id not in self.stuck:
            self.statuses[sandbox_id] = 'terminated'
        if sandbox_id in self.fail_terminate:
            raise RuntimeError('credential-shaped exception text must never enter the ledger')

    async def from_id(self, sandbox_id: str, *, owner_id: str | None = None) -> Sandbox:
        if sandbox_id in self.gone:
            raise SandboxNotFoundError('already gone')
        return _snapshot(sandbox_id, self.statuses[sandbox_id])


async def test_creation_is_persisted_before_returning_to_backend(tmp_path: Path) -> None:
    path = tmp_path / 'resources.json'
    ledger = ResourceLedger(path)

    async def create(*, token: str) -> Sandbox:
        return _snapshot('sbx-recorded')

    created = await ledger.recording_create(create)(token='do-not-serialize-me')
    data = json.loads(path.read_text())
    assert data['resources'][0]['id'] == created.id
    assert not data['audit_complete']
    assert 'do-not-serialize-me' not in path.read_text()


async def test_cleanup_audits_every_id_after_uncertain_termination(tmp_path: Path) -> None:
    ledger = ResourceLedger(tmp_path / 'resources.json')
    for sandbox_id in ('sbx-first', 'sbx-second'):
        ledger.record_created(_snapshot(sandbox_id))
    control = _FakeControl({sandbox_id: 'running' for sandbox_id in ledger.records})
    control.fail_terminate.add('sbx-first')
    await ledger.cleanup_and_audit(control)
    assert control.calls == ['sbx-first', 'sbx-second']
    assert ledger.audit_complete
    assert all(record.terminal for record in ledger.records.values())
    assert 'credential-shaped' not in ledger.path.read_text()
    assert ledger.records['sbx-first'].errors == ['RuntimeError']


async def test_cleanup_failure_retains_ids_and_still_cleans_other_resources(tmp_path: Path) -> None:
    ledger = ResourceLedger(tmp_path / 'resources.json')
    for sandbox_id in ('sbx-stuck', 'sbx-clean'):
        ledger.record_created(_snapshot(sandbox_id))
    control = _FakeControl({sandbox_id: 'running' for sandbox_id in ledger.records})
    control.stuck.add('sbx-stuck')
    with pytest.raises(RuntimeError, match='sbx-stuck'):
        await ledger.cleanup_and_audit(control, timeout=0.01)
    assert control.calls == ['sbx-stuck', 'sbx-clean']
    data = json.loads(ledger.path.read_text())
    assert not data['audit_complete']
    assert {record['id'] for record in data['resources']} == {'sbx-stuck', 'sbx-clean'}
    assert ledger.records['sbx-clean'].terminal


async def test_cleanup_refuses_an_unowned_sandbox(tmp_path: Path) -> None:
    ledger = ResourceLedger(tmp_path / 'resources.json')
    control = _FakeControl({'sbx-external': 'running'})
    with pytest.raises(ValueError, match='not created by this test run'):
        await ledger.terminate_owned(control, 'sbx-external')
    assert control.calls == []


async def test_cleanup_audits_already_absent_owned_resource(tmp_path: Path) -> None:
    ledger = ResourceLedger(tmp_path / 'resources.json')
    ledger.record_created(_snapshot('sbx-gone'))
    control = _FakeControl({'sbx-gone': 'running'})
    control.gone.add('sbx-gone')
    await ledger.cleanup_and_audit(control)
    assert ledger.audit_complete
    assert ledger.records['sbx-gone'].status == 'not_found'
    assert ledger.records['sbx-gone'].terminal
    assert control.calls == ['sbx-gone']
