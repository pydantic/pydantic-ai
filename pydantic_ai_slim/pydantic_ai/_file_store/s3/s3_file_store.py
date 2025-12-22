import os
from collections.abc import Callable
from dataclasses import dataclass, field
from mimetypes import guess_type
from typing import Any, cast

import httpx

from .utils import S3Error, create_presigned_url, sign_request

DEFAULT_TTL = 3600
"""Default TTL in seconds for presigned URLs."""


@dataclass
class S3FileStore:
    """S3-compatible file store for LLM media workflows.

    Designed for uploading generated media and providing URLs that can be
    passed back to LLMs instead of raw bytes.

    Works with any S3-compatible backend: AWS S3, Cloudflare R2, MinIO,
    DigitalOcean Spaces, Backblaze B2, etc.

    Attributes:
        bucket: Bucket name
        endpoint: S3-compatible endpoint URL, or None to use S3_ENDPOINT env var
            - AWS S3: https://s3.{region}.amazonaws.com
            - R2: https://{account_id}.r2.cloudflarestorage.com
            - MinIO: https://minio.example.com
        access_key_id: Access key ID, or None to use S3_ACCESS_KEY_ID env var
        secret_access_key: Secret access key, or None to use S3_SECRET_ACCESS_KEY env var
        region: AWS region, or None to use S3_REGION env var (use "auto" for R2, "us-east-1" for MinIO/others)
        public_url: Public/CDN URL base for the bucket (e.g., https://media.example.com)
        ttl: TTL in seconds for presigned URLs (default: 3600)
        custom_download_uri: Optional callback to override get_download_uri logic

    Example:
        ```python
        # Using environment variables (S3_ENDPOINT, S3_ACCESS_KEY_ID,
        # S3_SECRET_ACCESS_KEY, S3_REGION)
        store = S3FileStore(bucket='my-bucket')

        # Explicit configuration
        store = S3FileStore(
            bucket='my-bucket',
            endpoint='https://s3.us-east-1.amazonaws.com',
            access_key_id='...',
            secret_access_key='...',
            region='us-east-1',
        )

        # Public bucket with CDN
        store = S3FileStore(
            bucket='media',
            public_url='https://media.example.com',
        )

        # Upload generated image
        await store.store('images/gen-123.png', image_bytes)

        # Get URL for LLM
        url = store.get_download_uri('images/gen-123.png')
        ```
    """

    bucket: str
    endpoint: str | None = None
    """S3-compatible endpoint URL, falls back to S3_ENDPOINT env var."""
    access_key_id: str | None = None
    """Access key ID, falls back to S3_ACCESS_KEY_ID env var."""
    secret_access_key: str | None = None
    """Secret access key, falls back to S3_SECRET_ACCESS_KEY env var."""
    region: str | None = None
    """AWS region, falls back to S3_REGION env var."""
    public_url: str | None = None
    """Public/CDN URL base for the bucket (e.g., https://media.example.com)"""
    ttl: int = DEFAULT_TTL
    """TTL in seconds for presigned URLs (if applicable)"""
    custom_download_uri: Callable[[str], str] | None = None

    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Resolve from environment variables
        self.endpoint = self.endpoint or os.getenv('S3_ENDPOINT')
        self.access_key_id = self.access_key_id or os.getenv('S3_ACCESS_KEY_ID')
        self.secret_access_key = self.secret_access_key or os.getenv('S3_SECRET_ACCESS_KEY')
        self.region = self.region or os.getenv('S3_REGION')

        # Validate required fields
        if not self.endpoint:
            raise ValueError('endpoint is required: pass it directly or set S3_ENDPOINT env var')
        if not self.access_key_id:
            raise ValueError('access_key_id is required: pass it directly or set S3_ACCESS_KEY_ID env var')
        if not self.secret_access_key:
            raise ValueError('secret_access_key is required: pass it directly or set S3_SECRET_ACCESS_KEY env var')
        if not self.region:
            raise ValueError('region is required: pass it directly or set S3_REGION env var')

        self.endpoint = self.endpoint.rstrip('/')
        if self.public_url is not None:
            self.public_url = self.public_url.rstrip('/')

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args: Any):
        await self.close()

    def _bucket_url(self, key: str = '') -> str:
        """Build the full bucket URL for a key."""
        key = key.lstrip('/')
        if key:
            return f'{self.endpoint}/{self.bucket}/{key}'
        return f'{self.endpoint}/{self.bucket}'

    async def _request(
        self,
        method: str,
        key: str,
        body: bytes = b'',
        extra_headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Sign and execute an S3 request."""
        url = self._bucket_url(key)
        headers = extra_headers or {}

        signed_headers = sign_request(
            method=method,
            url=url,
            headers=headers,
            body=body,
            access_key=cast(str, self.access_key_id),
            secret_key=cast(str, self.secret_access_key),
            region=cast(str, self.region),
        )

        client = await self._get_client()
        response = await client.request(method, url, headers=signed_headers, content=body)
        return response

    def _check_response(self, response: httpx.Response):
        """Raise S3Error if response indicates failure."""
        if not response.is_success:
            raise S3Error(response.status_code, response.text)

    async def verify_access(self) -> bool:
        """Verify that credentials have access to the bucket.

        Makes a HEAD request to the bucket root.
        """
        response = await self._request('HEAD', '')

        if response.status_code == 200:
            return True
        elif response.status_code == 404:
            raise S3Error(404, f"Bucket '{self.bucket}' not found")
        elif response.status_code == 403:
            raise S3Error(403, f"Access denied to bucket '{self.bucket}'")
        else:
            self._check_response(response)
            return False

    async def store(self, key: str, data: bytes) -> str:
        """Store data in the bucket.

        Args:
            key: Object key (path within bucket)
            data: Raw bytes to store

        Returns:
            The object key that was stored
        """
        content_type, _ = guess_type(key)
        content_type = content_type or 'application/octet-stream'
        headers = {
            'content-type': content_type,
            'content-length': str(len(data)),
        }

        response = await self._request('PUT', key, body=data, extra_headers=headers)
        self._check_response(response)
        return key

    async def retrieve(self, key: str) -> bytes:
        """Retrieve data from the bucket.

        Args:
            key: Object key to retrieve

        Returns:
            Raw bytes of the object
        """
        response = await self._request('GET', key)
        self._check_response(response)
        return response.content

    def get_download_uri(self, key: str) -> str:
        """Get a download URI for the object.

        If custom_download_uri is set, uses that callback.
        If public_url is set, returns the public/CDN URL.
        Otherwise returns a presigned URL with the configured TTL.

        Args:
            key: Object key

        Returns:
            URL for downloading the object
        """
        # Custom callback takes precedence
        if self.custom_download_uri is not None:
            return self.custom_download_uri(key)

        key = key.lstrip('/')

        # Public bucket: return CDN URL
        if self.public_url is not None:
            return f'{self.public_url}/{key}'

        # Private bucket: return presigned URL
        return create_presigned_url(
            url=self._bucket_url(key),
            access_key=cast(str, self.access_key_id),
            secret_key=cast(str, self.secret_access_key),
            region=cast(str, self.region),
            expires_in=self.ttl,
        )

    async def exists(self, key: str) -> bool:
        """Check if an object exists."""
        response = await self._request('HEAD', key)
        return response.status_code == 200

    async def delete(self, key: str) -> bool:
        """Delete an object from the bucket.

        Returns:
            True if deleted, False if didn't exist
        """
        response = await self._request('DELETE', key)
        return response.status_code in (200, 204, 404)
