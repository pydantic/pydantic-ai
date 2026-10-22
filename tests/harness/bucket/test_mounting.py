"""Which buckets a sandbox mounts for shell commands; `integration_tests/bucket` mounts them for real."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from pydantic_ai.workspaces import Mount
from pydantic_ai_harness.bucket import Bucket
from pydantic_ai_harness.docker_sandbox import DockerSandboxBackend

from ...workspace_fakes import MemoryStorage

FUSE = ['--device', '/dev/fuse', '--cap-add', 'SYS_ADMIN']
EXPIRING: Any = {
    'access_key_id': 'a',
    'secret_access_key': 'b',
    'token': 't',
    'expires_at': datetime.now(UTC) + timedelta(hours=1),
}
AZURE_KEY: Any = {'access_key': 'k', 'expires_at': None}
LONG_LIVED: Any = {'access_key_id': 'a', 'secret_access_key': 'b', 'token': None, 'expires_at': None}


@pytest.mark.parametrize(
    ('storage', 'mounted'),
    [
        (Bucket('s3://acme/data', credentials=EXPIRING), True),
        (Bucket('s3://acme/data', credentials=lambda: EXPIRING), True),
        (
            Bucket('https://acme.blob.core.windows.net/data', credentials=AZURE_KEY, share_long_lived_credentials=True),
            True,
        ),
        # An Azure bucket with S3 keys can't be mounted.
        (Bucket('https://acme.blob.core.windows.net/data', credentials=EXPIRING), False),
        (Bucket('s3://acme/data', credentials=LONG_LIVED, share_long_lived_credentials=True), True),
        # Keys that never expire, unless the bucket says so, and keys found in the environment stay out.
        (Bucket('s3://acme/data', credentials=LONG_LIVED), False),
        (Bucket('s3://acme/data'), False),
        (Bucket('memory:///', credentials=EXPIRING), False),
        (Bucket('gs://acme/data', credentials=EXPIRING), False),
        (Bucket('https://acme.s3.amazonaws.com/data', credentials=EXPIRING), False),
        (MemoryStorage(), False),
    ],
)
def test_a_sandbox_with_fuse_mounts_buckets_with_expiring_keys(storage: Any, mounted: bool) -> None:
    mount = Mount('data', storage)

    assert DockerSandboxBackend(image='python:3.12', docker_args=FUSE).can_mount(mount) is mounted
    assert not DockerSandboxBackend(image='python:3.12').can_mount(mount)
