"""The live tier's session sweep, `LiveMachines` in `conftest.py`, run over the fake boxd SDK."""

from __future__ import annotations

import pytest
from boxd import APIConnectionError, APIStatusError, NotFoundError

from pydantic_ai_harness.boxd_sandbox import BoxdSandbox, BoxdSandboxBackend

from .conftest import LiveMachines
from .fake_boxd import FakeBoxd, FakeMachines


async def test_the_sweep_deletes_the_machines_a_cleanup_missed(
    fake_boxd: FakeBoxd, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`destroy` takes a machine boxd does not list yet as deleted already, and nothing destroys a fork copy no
    model response carried; the sweep deletes both, retrying while boxd does not list one, after a delete boxd
    failed with `INTERNAL`, and after a dropped connection. A machine a call deleted is left out."""
    live = LiveMachines(poll=0)
    live.watch(monkeypatch, FakeMachines)
    sandbox = BoxdSandbox()

    deleted = BoxdSandboxBackend()
    await deleted.get_sandbox()
    assert deleted.ref is not None
    await sandbox.destroy(deleted.ref)

    missed = BoxdSandboxBackend()
    await missed.get_sandbox()
    assert missed.ref is not None
    copy = await sandbox.fork(missed.ref)
    fake_boxd.get_errors.append(NotFoundError('VM not found', code='not_found', grpc_code=5))
    await sandbox.destroy(missed.ref)
    assert {machine.id: machine.status for machine in fake_boxd.machines.values()} == {
        deleted.ref.id: 'destroyed',
        missed.ref.id: 'running',
        copy.id: 'running',
    }
    assert live.pending == {missed.ref.id, copy.id}

    fake_boxd.delete_errors += [
        NotFoundError('VM not found', code='not_found', grpc_code=5),
        APIStatusError("couldn't destroy that machine", code='internal', grpc_code=13),
        APIConnectionError('the connection dropped before the reply arrived'),
    ]
    await live.sweep(fake_boxd.client().machines)
    assert {machine.status for machine in fake_boxd.machines.values()} == {'destroyed'}
    assert live.pending == set()
    assert fake_boxd.delete_errors == []
