"""Tests for `ModalSandbox(mounts=...)`, against the fake `modal`, which answers for the mount helper.

The helper itself runs only in a real sandbox: see `test_mounts_live.py`.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness import MountError
from pydantic_ai_harness.modal_sandbox import ModalSandbox
from tests.harness.mounts.probing import bucket, probing

from .fake_modal import FakeModal, FakeSandbox


def helper_replies(fake_modal: FakeModal, stdout: str = '{"recovered": []}\n', exit_code: int = 0) -> None:
    """Answer the helper's runs with this output, put the sandbox in `/root`, and echo every other command."""

    def respond(argv: list[str], timeout: int | None) -> tuple[bytes, bytes, int]:
        if argv[4:6] == ['python3', '-c']:
            return stdout.encode(), b'', exit_code
        if argv == ['pwd', '-P']:
            return b'/root\n', b'', 0
        return (' '.join(argv) + '\n').encode(), b'', 0

    fake_modal.responder = respond


def root_runs(sandbox: FakeSandbox) -> list[tuple[list[str], str | None, int | None, dict[str, Any]]]:
    """Each helper run's argv, workdir and Modal timeout, with the request it read from stdin."""
    return [
        (call.argv, call.workdir, call.timeout, json.loads(process.stdin.data))
        for call, process in zip(sandbox.exec_calls, sandbox.processes)
        if call.argv[4:6] == ['python3', '-c']
    ]


async def test_a_run_mounts_with_the_keys_only_on_stdin(fake_modal: FakeModal) -> None:
    helper_replies(fake_modal)

    result = await probing(ModalSandbox[str](warn_if_no_tools=False, runtime='vm', mounts=[bucket()])).run(
        'go', deps=''
    )

    assert result.output == '{"probe":"echo contained\\n"}'
    (sandbox,) = fake_modal.sandboxes
    ((argv, workdir, timeout, request),) = root_runs(sandbox)
    assert argv[:2] == ['sh', '-c'] and 'gvisor' in argv[2] and argv[3] == 'sh'
    assert workdir == '/'
    # Modal stops the helper before the mount call's own 1860 second wait runs out.
    assert timeout == 1830
    assert request['working_dir'] == '/root'
    assert request['mounts'][0]['keys']['secret_access_key'] == 'the-secret-key'
    assert next(process for process in sandbox.processes if process.stdin.data).stdin.closed
    assert not any('the-secret-key' in f'{call.argv} {call.env}' for call in sandbox.exec_calls)


async def test_mounts_can_be_a_function_of_the_run(fake_modal: FakeModal) -> None:
    helper_replies(fake_modal)

    await probing(ModalSandbox[str](warn_if_no_tools=False, mounts=lambda ctx: [bucket(key=ctx.deps)])).run(
        'go', deps='customer-a-key'
    )

    (sandbox,) = fake_modal.sandboxes
    assert [request['mounts'][0]['keys']['access_key_id'] for *_, request in root_runs(sandbox)] == ['customer-a-key']


async def test_an_attached_sandbox_is_mounted_too(fake_modal: FakeModal) -> None:
    helper_replies(fake_modal)

    await probing(ModalSandbox[str](warn_if_no_tools=False, mounts=[bucket()])).run(
        'go', deps='', workspace=WorkspaceRef(provider='modal', id='sb-earlier')
    )

    (sandbox,) = fake_modal.sandboxes
    assert len(root_runs(sandbox)) == 1


async def test_a_gvisor_sandbox_is_refused_with_the_runtime_it_needs(fake_modal: FakeModal) -> None:
    helper_replies(fake_modal, '', 86)

    with pytest.raises(MountError) as raised:
        await probing(ModalSandbox[str](warn_if_no_tools=False, mounts=[bucket()])).run('go', deps='')

    assert str(raised.value) == (
        'Modal sandbox sb-owned runs on gVisor, which cannot mount the FUSE file systems mounts use. '
        "Create it on Modal's VM runtime: `ModalSandbox(runtime='vm')`."
    )


async def test_the_helpers_refusal_fails_the_operation(fake_modal: FakeModal) -> None:
    helper_replies(
        fake_modal, '{"error": "conflict", "message": "data already shows another bucket in this sandbox."}\n'
    )

    with pytest.raises(MountError, match='data already shows another bucket in this sandbox'):
        await probing(ModalSandbox[str](warn_if_no_tools=False, mounts=[bucket()])).run('go', deps='')


def test_a_spec_cannot_set_mounts() -> None:
    assert 'mounts' not in inspect.signature(ModalSandbox.from_spec).parameters
    with pytest.raises(UserError, match='`mounts` holds keys, so an agent spec cannot set it'):
        ModalSandbox[None].from_spec(mounts=[])
