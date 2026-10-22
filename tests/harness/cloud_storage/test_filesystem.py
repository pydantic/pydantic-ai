"""Cloud storage filesystems against obstore's in-memory store; `integration_tests/cloud_storage` runs the same rules
on real servers."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip('obstore')

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import Mount, ReadOnlyWorkspace, Workspace, WorkspaceReadOnlyError, WorkspaceRef
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness import FileSystem, Shell
from pydantic_ai_harness.cloud_storage import AzureFilesystem, GCSFilesystem, S3Filesystem

from .memory_bucket import MemoryBucket

pytestmark = pytest.mark.anyio


class TestConformance(WorkspaceBackendSuite):
    @pytest.fixture
    def backend(self) -> MemoryBucket:
        return MemoryBucket()


async def test_a_bucket_mounts_as_a_folder() -> None:
    bucket = MemoryBucket()
    workspace = Workspace(MemoryBucket(), mounts=[Mount('data', bucket)])

    await workspace.write_text('data/reports/q3.md', '# Q3')
    await workspace.make_dir('data/empty')

    assert await bucket.read_bytes('/reports/q3.md') == b'# Q3'
    # The marker that keeps `empty` is not listed.
    assert [entry.name for entry in await workspace.list_dir('data/empty')] == []
    assert [entry.path for entry in await workspace.list_dir('data')] == ['/data/empty', '/data/reports']


async def test_the_bucket_root_cannot_be_removed() -> None:
    with pytest.raises(ValueError, match='cannot remove the root'):
        await MemoryBucket().remove('/')
    with pytest.raises(ValueError, match='absolute'):
        await MemoryBucket().read_bytes('a.txt')


def test_credentials_are_not_part_of_equality_or_repr() -> None:
    keys: Any = {'access_key_id': 'AKIA', 'secret_access_key': 'secret', 'token': None, 'expires_at': None}
    bucket = S3Filesystem('s3://acme/data', credentials=keys)

    assert bucket == S3Filesystem('s3://acme/data')
    assert bucket != S3Filesystem('s3://acme/other')
    assert 'secret' not in repr(bucket)
    assert bucket.ref == WorkspaceRef(provider='s3', id='s3://acme/data')


@pytest.mark.parametrize(
    ('make', 'error'),
    [
        (lambda: S3Filesystem('gs://acme'), 'S3Filesystem URLs start with `s3://`'),
        (lambda: GCSFilesystem('s3://acme'), 'GCSFilesystem URLs start with `gs://`'),
        (lambda: AzureFilesystem('s3://acme'), 'AzureFilesystem URLs start with `az://` or `https://`'),
        (lambda: AzureFilesystem('https://acme.example.com/data'), r'https://<account>\.blob\.core\.windows\.net/'),
        (lambda: AzureFilesystem('az://data'), 'needs `account=`'),
    ],
)
def test_each_class_takes_its_own_clouds_urls(make: Any, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        make()


def test_each_cloud_gets_its_own_settings() -> None:
    assert S3Filesystem('s3://a', endpoint='http://s3', region='auto')._config() == {  # pyright: ignore[reportPrivateUsage]
        'endpoint': 'http://s3',
        'region': 'auto',
    }
    assert GCSFilesystem('gs://a', endpoint='http://gcs')._config() == {'base_url': 'http://gcs'}  # pyright: ignore[reportPrivateUsage]
    assert AzureFilesystem('az://c', account='acct')._config() == {'account_name': 'acct'}  # pyright: ignore[reportPrivateUsage]
    assert AzureFilesystem('https://acct.blob.core.windows.net/c')._config() == {}  # pyright: ignore[reportPrivateUsage]


async def test_a_bucket_is_a_whole_workspace_without_a_shell() -> None:
    bucket = MemoryBucket()
    await bucket.write_bytes('/handbook.md', b'# Handbook')
    model = TestModel(call_tools=[])
    agent = Agent(model, capabilities=[FileSystem(), Shell()])

    result = await agent.run('Read the handbook.', workspace=ReadOnlyWorkspace(Workspace(bucket)))

    workspace = result.workspace
    assert workspace.ref == WorkspaceRef(provider='memory', id='memory:///')
    assert await workspace.read_text('handbook.md') == '# Handbook'
    with pytest.raises(WorkspaceReadOnlyError):
        await workspace.write_text('handbook.md', 'changed')
    # File tools are offered; shell tools aren't, as there is no shell.
    parameters = model.last_model_request_parameters
    assert parameters is not None
    names = {tool.name for tool in parameters.function_tools}
    assert 'read_file' in names
    assert 'run_command' not in names
