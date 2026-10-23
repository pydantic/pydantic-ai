"""Cloud storage filesystems against real object storage servers (see `conftest.py`).

The unit suite in `tests/harness/cloud_storage` runs the workspace conformance rules on obstore's in-memory store, which has
the same listing model but not the servers' behaviour. What only a server answers:

- that each cloud's listing, delete and missing-object replies map onto the filesystem rules;
- that a missing bucket or container reads as `WorkspaceUnavailableError` and rejected keys as `PermissionError`;
- that a credentials function is called for keys, as a source of expiring keys needs;
- that a folder an S3 console made (an empty `<name>/` object) lists as a folder.

Run with `make integration-cloud-storage`.
"""

from __future__ import annotations

import base64
import uuid
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import httpx
import pytest

from pydantic_ai.workspaces import WorkspaceUnavailableError
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.cloud_storage import AzureFilesystem

from .conftest import (
    AZURE_KEY,
    BUCKET,
    S3_ENDPOINT,
    Filesystem,
    azure_bucket,
    create_azure_container,
    create_s3_bucket,
    s3_bucket,
)

pytestmark = pytest.mark.anyio


class TestConformance(WorkspaceBackendSuite):
    """The workspace backend rules, with the bucket as the whole workspace."""

    @pytest.fixture
    def backend(self, new_bucket: Callable[..., Filesystem]) -> Filesystem:
        return new_bucket()


async def test_a_missing_bucket_is_unavailable(new_bucket: Callable[..., Filesystem]) -> None:
    """Retrying can't help, so it is not a transient error."""
    bucket = new_bucket()
    # A SAS token covers one container, so Azure refuses it before saying that another one is missing.
    credentials = {'access_key': AZURE_KEY, 'expires_at': None} if isinstance(bucket, AzureFilesystem) else None
    missing = replace(
        bucket,
        url=bucket.url.replace(BUCKET, f'pai-missing-{uuid.uuid4().hex[:8]}'),
        credentials=credentials or bucket.credentials,
    )

    with pytest.raises(WorkspaceUnavailableError, match='does not exist'):
        await missing.read_bytes('/a.txt')


async def test_rejected_keys_are_a_permission_error() -> None:
    """Azurite checks the key; SeaweedFS without an identity file accepts any."""
    create_azure_container()
    wrong = base64.b64encode(b'not the key').decode()
    bucket = azure_bucket(f'{BUCKET}/{uuid.uuid4().hex}', credentials={'access_key': wrong, 'expires_at': None})

    with pytest.raises(PermissionError, match='denied'):
        await bucket.read_bytes('/a.txt')


async def test_a_credentials_function_supplies_the_keys(new_bucket: Callable[..., Filesystem]) -> None:
    """A function is how expiring keys reach the bucket."""
    keys = new_bucket().credentials
    calls: list[None] = []

    def fresh_keys() -> Any:
        calls.append(None)
        return keys() if callable(keys) else keys

    storage = new_bucket(credentials=fresh_keys)
    await storage.write_bytes('/a.txt', b'a')

    assert await storage.read_bytes('/a.txt') == b'a'
    assert calls


async def test_a_console_folder_lists_as_a_folder() -> None:
    """S3 consoles make a folder by writing an empty `<name>/` object, which obstore lists without the slash."""
    create_s3_bucket()
    folder = f'{BUCKET}/{uuid.uuid4().hex}'
    bucket = s3_bucket(folder)
    await bucket.write_bytes('/docs/a.txt', b'a')
    httpx.put(f'{S3_ENDPOINT}/{folder}/docs/').raise_for_status()

    assert [(entry.name, entry.is_dir) for entry in await bucket.list_dir('/')] == [('docs', True)]
