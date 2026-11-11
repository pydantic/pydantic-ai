"""Tests for `E2BSandbox(mounts=...)`, against the fake `e2b`, which answers for the mount helper.

The helper itself runs only in a real sandbox: see `test_mounts_live.py`.
"""

from __future__ import annotations

import inspect

import pytest
from e2b.exceptions import NotFoundException, SandboxException

from pydantic_ai.workspaces import WorkspaceError, WorkspaceRef
from pydantic_ai_harness import MountError
from pydantic_ai_harness.e2b_sandbox import E2BSandbox
from tests.harness.mounts._requests import Request, parse_request
from tests.harness.mounts.sandbox_fixtures import bucket, probing

from .fake_e2b import FakeE2B, FakeSandbox


def helper_replies(
    fake_e2b: FakeE2B, stdout: str = '{"recovered": []}\n', stderr: str = '', exit_code: int = 0
) -> None:
    """Answer the helper's runs with this output, put the sandbox in `/home/user`, and echo every other command."""

    def respond(command: str, timeout: float | None) -> tuple[str, str, int]:
        if command.startswith('python3 -c '):
            return stdout, stderr, exit_code
        if command == 'pwd -P':
            return '/home/user\n', '', 0
        return f'{command}\n', '', 0

    fake_e2b.responder = respond


def root_requests(sandbox: FakeSandbox) -> list[Request]:
    return [parse_request(data) for _, data in sandbox.commands.stdin_sent]


async def test_a_run_mounts_as_root_with_the_keys_only_on_stdin(fake_e2b: FakeE2B) -> None:
    helper_replies(fake_e2b)

    result = await probing(E2BSandbox[str](mounts=[bucket()])).run('go', deps='')

    assert result.output == '{"probe":"echo contained\\n"}'
    (sandbox,) = fake_e2b.sandboxes
    (root_call,) = [call for call in sandbox.commands.calls if call.user == 'root']
    assert root_call.command.startswith('python3 -c ')
    assert (root_call.cwd, root_call.stdin, root_call.background) == ('/', True, True)
    root_handle = sandbox.commands.handles[sandbox.commands.calls.index(root_call)]
    assert [pid for pid, _ in sandbox.commands.stdin_sent] == [root_handle.pid]
    (request,) = root_requests(sandbox)
    assert request['working_dir'] == '/home/user'
    assert request['mounts'][0]['keys']['secret_access_key'] == 'the-secret-key'
    assert not any('the-secret-key' in f'{call.command} {call.envs}' for call in sandbox.commands.calls)


async def test_mounts_can_be_a_function_of_the_run(fake_e2b: FakeE2B) -> None:
    helper_replies(fake_e2b)

    await probing(E2BSandbox[str](mounts=lambda ctx: [bucket(key=ctx.deps)])).run('go', deps='customer-a-key')

    (sandbox,) = fake_e2b.sandboxes
    assert [request['mounts'][0]['keys']['access_key_id'] for request in root_requests(sandbox)] == ['customer-a-key']


async def test_an_attached_sandbox_is_mounted_too(fake_e2b: FakeE2B) -> None:
    helper_replies(fake_e2b)
    sandbox = fake_e2b.new_sandbox('sbx-earlier')

    await probing(E2BSandbox[str](mounts=[bucket()])).run(
        'go', deps='', workspace=WorkspaceRef(provider='e2b', id='sbx-earlier')
    )

    assert len(root_requests(sandbox)) == 1


async def test_the_helpers_refusal_fails_the_operation(fake_e2b: FakeE2B) -> None:
    helper_replies(fake_e2b, '{"error": "conflict", "message": "data already shows another bucket in this sandbox."}\n')

    with pytest.raises(MountError, match='data already shows another bucket in this sandbox'):
        await probing(E2BSandbox[str](mounts=[bucket()])).run('go', deps='')


async def test_a_helper_that_exits_before_its_input_arrives_reports_why(fake_e2b: FakeE2B) -> None:
    helper_replies(fake_e2b, '', 'sh: python3: not found\n', 127)
    fake_e2b.stdin_error = NotFoundException('process with pid 1 not found')

    with pytest.raises(MountError) as raised:
        await probing(E2BSandbox[str](mounts=[bucket()])).run('go', deps='')

    assert str(raised.value) == 'Could not mount storage in the sandbox: sh: python3: not found'


async def test_a_failed_wait_kills_the_helper(fake_e2b: FakeE2B) -> None:
    helper_replies(fake_e2b)
    fake_e2b.root_wait_error = SandboxException('the output stream broke')

    with pytest.raises(WorkspaceError) as raised:
        await probing(E2BSandbox[str](mounts=[bucket()])).run('go', deps='')

    assert str(raised.value) == 'Could not mount storage in the E2B sandbox: the output stream broke'
    (sandbox,) = fake_e2b.sandboxes
    killed = [call.user for call, handle in zip(sandbox.commands.calls, sandbox.commands.handles) if handle.killed]
    assert killed == ['root']


def test_from_spec_leaves_out_mounts() -> None:
    assert 'mounts' not in inspect.signature(E2BSandbox.from_spec).parameters
