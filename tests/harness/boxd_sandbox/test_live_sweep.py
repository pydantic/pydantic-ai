"""The live tier's session sweep, `LiveMachines` in `conftest.py`, run over the fake boxd SDK."""

from __future__ import annotations

import os

import pytest
from boxd import APIConnectionError, APIStatusError, AuthenticationError, NotFoundError

from pydantic_ai.workspaces import WorkspaceUnavailableError
from pydantic_ai_harness.boxd_sandbox import BoxdSandbox, BoxdSandboxBackend

from .conftest import LiveMachines, delete_live_machine
from .fake_boxd import FakeBoxd, FakeMachines

# The fake runs a machine's commands under the host's `/bin/sh`.
pytestmark = pytest.mark.skipif(os.name != 'posix', reason='needs a POSIX host')


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
    assert (live.recorded, live.pending) == (3, {missed.ref.id, copy.id})

    fake_boxd.delete_errors += [
        NotFoundError('VM not found', code='not_found', grpc_code=5),
        APIStatusError("couldn't destroy that machine", code='internal', grpc_code=13),
        APIConnectionError('the connection dropped before the reply arrived'),
    ]
    await live.sweep(fake_boxd.client().machines)
    assert {machine.status for machine in fake_boxd.machines.values()} == {'destroyed'}
    assert live.pending == set()
    assert fake_boxd.delete_errors == []


async def test_the_sweep_stops_on_refused_credentials(fake_boxd: FakeBoxd, monkeypatch: pytest.MonkeyPatch) -> None:
    """A delete refused for its credentials fails the session at once instead of retrying until `timeout`."""
    live = LiveMachines(poll=0)
    live.watch(monkeypatch, FakeMachines)
    backend = BoxdSandboxBackend()
    await backend.get_sandbox()
    assert backend.ref is not None

    fake_boxd.reject_credentials = True
    with pytest.RaisesGroup(AuthenticationError):
        await live.sweep(fake_boxd.client().machines)
    assert live.pending == {backend.ref.id}


async def test_a_cleanup_whose_delete_boxd_fails_leaves_the_machine_to_the_sweep(
    fake_boxd: FakeBoxd, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A live test's cleanup whose delete boxd fails with `INTERNAL` logs the machine instead of failing the test;
    the machine stays recorded, and the sweep deletes it. Refused credentials still fail the cleanup."""
    live = LiveMachines(poll=0)
    live.watch(monkeypatch, FakeMachines)
    backend = BoxdSandboxBackend()
    await backend.get_sandbox()
    assert backend.ref is not None

    fake_boxd.delete_errors.append(APIStatusError("couldn't destroy that machine", code='internal', grpc_code=13))
    await delete_live_machine(backend.ref)
    assert (fake_boxd.only.status, live.pending) == ('running', {backend.ref.id})
    assert f'boxd did not delete machine {backend.ref.id}' in caplog.text

    fake_boxd.reject_credentials = True
    with pytest.raises(WorkspaceUnavailableError):
        await delete_live_machine(backend.ref)
    fake_boxd.reject_credentials = False

    await live.sweep(fake_boxd.client().machines)
    assert (fake_boxd.only.status, live.pending) == ('destroyed', set())
