"""Which filesystems a sandbox mounts for shell commands; `integration_tests/cloud_storage` mounts them for real."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

pytest.importorskip('obstore')

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import Mount, Workspace
from pydantic_ai_harness.cloud_storage import AzureFilesystem, GCSFilesystem, S3Filesystem
from pydantic_ai_harness.docker_sandbox import DockerSandboxBackend

from ...workspace_fakes import MemoryFilesystem
from .memory_bucket import MemoryBucket

FUSE = ['--device', '/dev/fuse', '--cap-add', 'SYS_ADMIN']
EXPIRING: Any = {
    'access_key_id': 'a',
    'secret_access_key': 'b',
    'token': 't',
    'expires_at': datetime.now(UTC) + timedelta(hours=1),
}
EXPIRED: Any = {**EXPIRING, 'expires_at': datetime.now(UTC) - timedelta(minutes=1)}
AZURE_KEY: Any = {'access_key': 'k', 'expires_at': None}
LONG_LIVED: Any = {'access_key_id': 'a', 'secret_access_key': 'b', 'token': None, 'expires_at': None}
AZURE = 'https://acme.blob.core.windows.net/data'


@pytest.mark.parametrize(
    'filesystem',
    [
        S3Filesystem('s3://acme/data', credentials=EXPIRING),
        S3Filesystem('s3://acme/data', credentials=lambda: EXPIRING),
        S3Filesystem('s3://acme/data', credentials=LONG_LIVED, share_long_lived_credentials=True),
        AzureFilesystem(AZURE, credentials=AZURE_KEY, share_long_lived_credentials=True),
    ],
)
def test_a_sandbox_with_fuse_mounts_s3_and_azure_with_keys_it_can_hand_over(filesystem: Any) -> None:
    mount = Mount('data', filesystem, shell=True)

    Workspace(DockerSandboxBackend(image='python:3.12', docker_args=FUSE), mounts=[mount])
    with pytest.raises(UserError, match='only when `docker_args` give the container FUSE'):
        Workspace(DockerSandboxBackend(image='python:3.12'), mounts=[mount])


@pytest.mark.parametrize(
    ('filesystem', 'reason'),
    [
        (S3Filesystem('s3://acme/data'), 'finds its keys in the environment'),
        (S3Filesystem('s3://acme/data', credentials=LONG_LIVED), 'do not expire'),
        (S3Filesystem('s3://acme/data', credentials=EXPIRED), 'have expired'),
        (AzureFilesystem(AZURE, credentials=AZURE_KEY), 'do not expire'),
        # An Azure filesystem given S3 keys has no account key to mount with.
        (AzureFilesystem(AZURE, credentials=EXPIRING), 'needs an access key'),
        (GCSFilesystem('gs://acme/data', credentials=EXPIRING), 'GCSFilesystem cannot be mounted'),
        (MemoryBucket(), 'MemoryBucket cannot be mounted'),
        (MemoryFilesystem(), 'MemoryFilesystem cannot be mounted'),
    ],
)
def test_a_shell_mount_that_cannot_work_is_refused_with_the_reason(filesystem: Any, reason: str) -> None:
    with pytest.raises(UserError, match=reason):
        Workspace(
            DockerSandboxBackend(image='python:3.12', docker_args=FUSE), mounts=[Mount('data', filesystem, shell=True)]
        )
    # File tools reach it either way.
    Workspace(DockerSandboxBackend(image='python:3.12', docker_args=FUSE), mounts=[Mount('data', filesystem)])
