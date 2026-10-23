"""Which filesystems a sandbox mounts for commands, and what it asks the sandbox to do; `integration_tests/cloud_storage`
mounts them for real."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

pytest.importorskip('obstore')

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import CommandResult, Mount, MountError, Workspace, WorkspaceUnavailableError
from pydantic_ai_harness._mounts import Mounter
from pydantic_ai_harness.cloud_storage import AzureFilesystem, GCSFilesystem, S3Filesystem
from pydantic_ai_harness.docker_sandbox import DockerSandboxBackend

from ...workspace_fakes import MemoryFilesystem

pytestmark = pytest.mark.anyio

FUSE = ['--device', '/dev/fuse', '--cap-add', 'SYS_ADMIN']
IN_AN_HOUR = datetime.now(UTC) + timedelta(hours=1)
EXPIRING: Any = {'access_key_id': 'a', 'secret_access_key': 'b', 'token': 't', 'expires_at': IN_AN_HOUR}
EXPIRED: Any = {**EXPIRING, 'expires_at': datetime.now(UTC) - timedelta(minutes=1)}
LONG_LIVED: Any = {'access_key_id': 'a', 'secret_access_key': 'b', 'token': None, 'expires_at': None}
AZURE = 'https://acme.blob.core.windows.net/data'
AZURE_KEY: Any = {'access_key': 'k', 'expires_at': None}
SAS: Any = {'sas_token': 'sv=2024&sig=x', 'expires_at': IN_AN_HOUR}


def _docker(*mounts: Mount) -> Workspace:
    return Workspace(DockerSandboxBackend(image='python:3.12', docker_args=FUSE), mounts=mounts)


@pytest.mark.parametrize(
    'filesystem',
    [
        S3Filesystem('s3://acme/data', credentials=EXPIRING),
        S3Filesystem('s3://acme/data', credentials=lambda: EXPIRING),
        S3Filesystem('s3://acme/data', credentials=LONG_LIVED, share_long_lived_credentials=True),
        AzureFilesystem(AZURE, credentials=SAS),
        AzureFilesystem(AZURE, credentials=AZURE_KEY, share_long_lived_credentials=True),
    ],
)
def test_s3_and_azure_with_keys_that_can_be_handed_over_are_mounted(filesystem: Any) -> None:
    _docker(Mount('data', filesystem, commands=True))


@pytest.mark.parametrize(
    ('filesystem', 'reason'),
    [
        (S3Filesystem('s3://acme/data'), 'S3Filesystem finds its keys in the environment'),
        (S3Filesystem('s3://acme/data', credentials=LONG_LIVED), 'do not expire'),
        (S3Filesystem('s3://acme/data', credentials=EXPIRED), 'have expired'),
        (AzureFilesystem(AZURE, credentials=AZURE_KEY), 'do not expire'),
        (AzureFilesystem(AZURE, credentials=EXPIRING), 'need an `access_key` or a `sas_token`'),
        (S3Filesystem('s3://acme/data', credentials=SAS), 'need `access_key_id`'),
        (GCSFilesystem('gs://acme/data', credentials=EXPIRING), 'GCSFilesystem cannot be mounted for commands'),
        (MemoryFilesystem(), 'MemoryFilesystem cannot be mounted for commands'),
    ],
)
def test_a_mount_that_cannot_work_is_refused_with_the_reason(filesystem: Any, reason: str) -> None:
    with pytest.raises(UserError, match=f"^Mount 'data' can't be shown to commands: .*{reason}"):
        _docker(Mount('data', filesystem, commands=True))


def test_docker_needs_fuse() -> None:
    mount = Mount('data', S3Filesystem('s3://acme/data', credentials=EXPIRING), commands=True)
    with pytest.raises(UserError, match='only when `docker_args` give the container FUSE'):
        Workspace(DockerSandboxBackend(image='python:3.12'), mounts=[mount])


class Sandbox:
    """Answers the mount helper as a sandbox would, recording what it was asked."""

    def __init__(self, reply: dict[str, Any] | str | None = None) -> None:
        self.reply = reply or {'recovered': []}
        self.gone: Exception | None = None
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    async def run_as_root(self, argv: Sequence[str], stdin: bytes) -> CommandResult:
        self.calls.append((list(argv), json.loads(stdin)))
        if self.gone is not None:
            raise self.gone
        answer = self.reply if isinstance(self.reply, str) else json.dumps(self.reply)
        return CommandResult(exit_code=0, stdout=f'sudo: banner\n{answer}\n', stderr='')


async def test_keys_go_to_the_sandbox_in_rclones_terms() -> None:
    sandbox = Sandbox()
    s3 = S3Filesystem('s3://acme/data/', credentials=EXPIRING, region='eu-west-1')
    azure = AzureFilesystem(
        AZURE,
        credentials={'sas_token': [('sv', '2024'), ('sig', 'x')], 'expires_at': IN_AN_HOUR},
        endpoint='https://acme.blob.example',
    )

    await Mounter(sandbox.run_as_root, attached=False).ensure(
        [Mount('s3', s3, commands=True), Mount('az', azure, commands=True, read_only=True)], '/w'
    )

    ((argv, request),) = sandbox.calls
    assert argv[:2] == ['python3', '-c']
    assert request['mounts'][1]['remote'] == {
        'type': 'azureblob',
        'account': 'acme',
        'endpoint': 'https://acme.blob.example',
    }
    assert [(mount['path'], mount['root'], mount['read_only']) for mount in request['mounts']] == [
        ('/w/s3', 'acme/data', False),
        ('/w/az', 'data', True),
    ]
    assert list(request['keys'].values()) == [
        {'access_key_id': 'a', 'secret_access_key': 'b', 'session_token': 't'},
        {'sas_url': 'https://acme.blob.example/data?sv=2024&sig=x'},
    ]


async def test_mounts_are_ensured_again_before_keys_from_a_function_expire_or_the_lease_ends() -> None:
    soon = datetime.now(UTC) + timedelta(minutes=5)
    keys: Any = {**EXPIRING, 'expires_at': soon}
    mounter = Mounter(Sandbox().run_as_root, attached=False)

    from_function = S3Filesystem('s3://acme/data', credentials=lambda: keys)
    assert await mounter.ensure([Mount('data', from_function, commands=True)], '/w') == soon
    # Given keys can't be renewed, so only the lease, which lasts 15 minutes, needs renewing.
    given = S3Filesystem('s3://acme/data', credentials=EXPIRING)
    again = await mounter.ensure([Mount('data', given, commands=True)], '/w')
    assert again is not None and timedelta(minutes=14) < again - datetime.now(UTC) <= timedelta(minutes=15)


async def test_keys_from_a_function_are_checked_when_fetched() -> None:
    azure = AzureFilesystem(AZURE, credentials=lambda: EXPIRING)
    with pytest.raises(MountError, match='need an `access_key` or a `sas_token`'):
        await Mounter(Sandbox().run_as_root, attached=False).ensure([Mount('data', azure, commands=True)], '/w')


async def test_the_helpers_refusal_and_recovered_files_reach_the_caller() -> None:
    mount = Mount('data', S3Filesystem('s3://acme/data', credentials=EXPIRING), commands=True)
    busy = Sandbox({'error': 'busy', 'message': 'Another run is using this sandbox with different mounts.'})
    with pytest.raises(MountError, match=r'^Another run is using this sandbox'):
        await Mounter(busy.run_as_root, attached=True).ensure([mount], '/w')

    # A helper that never got to answer, such as without Python, is reported with what it printed.
    with pytest.raises(MountError, match=r'^Could not mount storage in the sandbox: sudo: banner\npython3: not found$'):
        await Mounter(Sandbox('python3: not found').run_as_root, attached=True).ensure([mount], '/w')

    recovered = Sandbox({'recovered': ['/w/.pydantic-ai-recovered/x-data']})
    with pytest.warns(UserWarning, match=r'moved to /w/\.pydantic-ai-recovered/x-data'):
        await Mounter(recovered.run_as_root, attached=True).ensure([mount], '/w')


async def test_only_a_sandbox_that_may_hold_an_earlier_runs_mounts_is_checked_for_them() -> None:
    assert not Mounter(Sandbox().run_as_root, attached=False).needed([])

    sandbox = Sandbox()
    mounter = Mounter(sandbox.run_as_root, attached=True)
    assert mounter.needed([])
    assert await mounter.ensure([], '/w') is None
    # The helper runs only if something is mounted, so a sandbox without Python is fine.
    ((argv, request),) = sandbox.calls
    assert argv[:2] == ['sh', '-c'] and request['mounts'] == []
    assert not mounter.needed([])


async def test_a_run_releases_only_mounts_it_holds_and_never_fails_for_it() -> None:
    mount = Mount('data', S3Filesystem('s3://acme/data', credentials=EXPIRING), commands=True)
    sandbox = Sandbox()
    mounter = Mounter(sandbox.run_as_root, attached=False)
    await mounter.release()
    assert sandbox.calls == []

    await mounter.ensure([mount], '/w')
    # Another workspace of the same run has nothing more to do.
    assert not mounter.needed([])
    await mounter.release()
    await mounter.release()
    assert [request for _, request in sandbox.calls[1:]] == [{'release': mounter.owner}]
    # The mounts stay in the sandbox, for the next run there to keep or remove.
    assert mounter.needed([])

    # A sandbox that can't be asked lets the lease end by itself, rather than failing the run.
    for gone in (MountError('The sandbox stopped.'), WorkspaceUnavailableError('The sandbox is gone.')):
        sandbox.gone = None
        await mounter.ensure([mount], '/w')
        sandbox.gone = gone
        await mounter.release()
