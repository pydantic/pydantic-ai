"""`Bucket` against obstore's in-memory store; `integration_tests/bucket` runs the same rules on real servers."""

from __future__ import annotations

import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import Mount, Workspace, WorkspaceReadOnlyError, WorkspaceRef
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.bucket import Bucket, BucketWorkspace, BucketWorkspaceBackend

pytestmark = pytest.mark.anyio


class TestBucketConformance(WorkspaceBackendSuite):
    @pytest.fixture
    def backend(self) -> BucketWorkspaceBackend:
        return BucketWorkspaceBackend(Bucket('memory:///'))


async def test_a_bucket_mounts_as_a_folder() -> None:
    bucket = Bucket('memory:///')
    workspace = Workspace(BucketWorkspaceBackend(Bucket('memory:///')), mounts=[Mount('data', bucket)])

    await workspace.write_text('data/reports/q3.md', '# Q3')
    await workspace.make_dir('data/empty')

    assert await bucket.read_bytes('/reports/q3.md') == b'# Q3'
    # The marker that keeps `empty` is not listed.
    assert [entry.name for entry in await workspace.list_dir('data/empty')] == []
    assert [entry.path for entry in await workspace.list_dir('data')] == ['/data/empty', '/data/reports']


async def test_the_bucket_root_cannot_be_removed() -> None:
    with pytest.raises(ValueError, match='cannot remove the root'):
        await Bucket('memory:///').remove('/')
    with pytest.raises(ValueError, match='absolute'):
        await Bucket('memory:///').read_bytes('a.txt')


def test_credentials_are_not_part_of_equality_or_repr() -> None:
    keys = {'access_key_id': 'AKIA', 'secret_access_key': 'secret', 'token': None, 'expires_at': None}
    bucket = Bucket('s3://acme/data', credentials=keys)  # pyright: ignore[reportArgumentType]

    assert bucket == Bucket('s3://acme/data')
    assert bucket != Bucket('s3://acme/other')
    assert 'secret' not in repr(bucket)


async def test_bucket_workspace_is_the_whole_workspace() -> None:
    bucket = Bucket('memory:///')
    await bucket.write_bytes('/handbook.md', b'# Handbook')
    agent = Agent(TestModel(), capabilities=[BucketWorkspace(bucket, read_only=True)])

    result = await agent.run('Read the handbook.')

    workspace = result.workspace
    assert workspace.ref == WorkspaceRef(provider='bucket', id='memory:///')
    assert await workspace.read_text('handbook.md') == '# Handbook'
    with pytest.raises(WorkspaceReadOnlyError):
        await workspace.write_text('handbook.md', 'changed')
    # A ref for another bucket is left to other capabilities.
    with pytest.raises(UserError, match='recognized'):
        await agent.run('Go.', workspace=WorkspaceRef(provider='bucket', id='s3://other'))
