"""The storage servers both test files use: SeaweedFS for S3 and Azurite for Azure Blob Storage.

Start them so a sandbox container can share their network, as CI does:

```bash
docker run -d --name pai-s3 -p 8333:8333 -p 10000:10000 chrislusf/seaweedfs:4.48 server -s3 -dir=/data -volume.max=0
docker run -d --network container:pai-s3 mcr.microsoft.com/azure-storage/azurite:3.35.0 \
    azurite-blob --blobHost 0.0.0.0
```

Without a reachable server the tests skip, unless `BUCKET_REQUIRE_LIVE` is set (CI does), where they fail instead.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from email.utils import formatdate
from typing import Any, NoReturn

import httpx
import pytest

from pydantic_ai_harness.bucket import Bucket

S3_ENDPOINT = 'http://127.0.0.1:8333'
AZURE_ENDPOINT = 'http://127.0.0.1:10000/devstoreaccount1'
AZURE_ACCOUNT = 'devstoreaccount1'
AZURE_KEY = 'Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=='
"""Azurite's fixed development account, documented at
<https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azurite#well-known-storage-account-and-key>."""

BUCKET = 'pai-integration'
"""One bucket (or container) per server, with a new folder for each test: SeaweedFS gives each bucket
storage volumes of its own and runs out of them after a few dozen."""


def unavailable(message: str) -> NoReturn:
    """Skip, or fail where the servers must be there."""
    if os.environ.get('BUCKET_REQUIRE_LIVE', '').lower() in {'1', 'true', 'yes'}:
        pytest.fail(message)
    pytest.skip(message)


def s3_keys() -> Any:
    """Keys that expire, the kind a sandbox is handed; SeaweedFS without an identity file accepts any."""
    expires_at = datetime.now(UTC) + timedelta(minutes=15)
    return {'access_key_id': 'any', 'secret_access_key': 'any', 'token': None, 'expires_at': expires_at}


def create_s3_bucket() -> None:
    """Create the S3 test bucket unless it exists."""
    try:
        response = httpx.put(f'{S3_ENDPOINT}/{BUCKET}')
        if response.status_code != 409:  # It exists already.
            response.raise_for_status()
    except httpx.HTTPError as error:
        unavailable(f'no reachable S3 server at {S3_ENDPOINT}: {error}')


def create_azure_container() -> None:
    """Create the Azure test container unless it exists."""
    # The Shared Key scheme, as Azure documents it for a request with no body and no other headers:
    # https://learn.microsoft.com/en-us/rest/api/storageservices/authorize-with-shared-key
    date, version = formatdate(usegmt=True), '2021-08-06'
    resource = f'/{AZURE_ACCOUNT}/{AZURE_ACCOUNT}/{BUCKET}\nrestype:container'
    to_sign = f'PUT\n\n\n\n\n\n\n\n\n\n\n\nx-ms-date:{date}\nx-ms-version:{version}\n{resource}'
    signature = base64.b64encode(hmac.new(base64.b64decode(AZURE_KEY), to_sign.encode(), hashlib.sha256).digest())
    headers = {
        'x-ms-date': date,
        'x-ms-version': version,
        'Authorization': f'SharedKey {AZURE_ACCOUNT}:{signature.decode()}',
    }
    try:
        response = httpx.put(f'{AZURE_ENDPOINT}/{BUCKET}?restype=container', headers=headers)
        if response.status_code != 409:  # It exists already.
            response.raise_for_status()
    except httpx.HTTPError as error:
        unavailable(f'no reachable Azure Blob server at {AZURE_ENDPOINT}: {error}')


def s3_bucket(folder: str, credentials: Any = s3_keys) -> Bucket:
    """A `Bucket` for `<bucket>/<folder>` on the S3 server."""
    return Bucket(f's3://{folder}', endpoint=S3_ENDPOINT, region='us-east-1', credentials=credentials)


def azure_bucket(folder: str, credentials: Any = None) -> Bucket:
    """A `Bucket` for `<container>/<folder>` on the Azure server."""
    return Bucket(
        f'az://{folder}',
        endpoint=AZURE_ENDPOINT,
        options={'account_name': AZURE_ACCOUNT},
        credentials=credentials or {'access_key': AZURE_KEY, 'expires_at': None},
        # Azurite has only its one account key, and shell mounts on Azure take an account key.
        share_long_lived_credentials=True,
    )


_CLOUDS = {'s3': (create_s3_bucket, s3_bucket), 'azure': (create_azure_container, azure_bucket)}


@pytest.fixture(params=list(_CLOUDS))
def new_bucket(request: pytest.FixtureRequest) -> Callable[..., Bucket]:
    """Makes a `Bucket` for a new folder in one cloud's test bucket; keyword arguments override its settings."""
    create, bucket = _CLOUDS[request.param]
    create()

    def make(credentials: Any = None) -> Bucket:
        folder = f'{BUCKET}/{uuid.uuid4().hex}'
        return bucket(folder) if credentials is None else bucket(folder, credentials)

    return make
