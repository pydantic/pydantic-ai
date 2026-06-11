"""S3-compatible media store with handrolled AWS SigV4 signing.

Targets AWS S3, Cloudflare R2, MinIO, and other S3-compatible endpoints.
Implements path-style URLs (`<endpoint>/<bucket>/<key>`) -- the lowest common
denominator across providers. Only PUT / GET / HEAD are implemented;
multipart, lifecycle, and listing operations are out of scope (any blob we
store is sub-5GB by construction so multipart is unnecessary).

The SigV4 implementation follows the AWS reference algorithm but is local
to this module to keep `pydantic-ai-harness` free of `botocore` / `boto3`.
If a provider needs deeper integration, write a subclass that overrides
`_request`.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from urllib.parse import quote

from pydantic_ai_harness.experimental.media._store import (
    _EMPTY_CONTEXT,  # pyright: ignore[reportPrivateUsage]
    KeyStrategy,
    MediaContext,
    PublicUrlResolver,
    _resolve_public_url,  # pyright: ignore[reportPrivateUsage]
    default_key_strategy,
    media_uri_for,
    parse_media_uri,
)

if TYPE_CHECKING:
    import httpx


_ALGORITHM = 'AWS4-HMAC-SHA256'
_SERVICE = 's3'
# Conservative subset for x-amz-meta-* keys: ASCII letters/digits/dash; lowercase.
_META_KEY_RE = re.compile(r'^[a-zA-Z0-9-]+$')


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _hex_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac_sha256(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()


def _canonical_uri(path: str) -> str:
    """Percent-encode each path segment but keep `/` separators.

    AWS canonicalization requires double-encoding for non-S3 services but
    single-encoding for S3 -- we use single-encoding throughout.
    """
    return '/' + '/'.join(quote(segment, safe='') for segment in path.lstrip('/').split('/'))


def _signing_key(secret_access_key: str, date_stamp: str, region: str) -> bytes:
    k_date = _hmac_sha256(f'AWS4{secret_access_key}'.encode(), date_stamp)
    k_region = _hmac_sha256(k_date, region)
    k_service = _hmac_sha256(k_region, _SERVICE)
    return _hmac_sha256(k_service, 'aws4_request')


def sign_request(
    *,
    method: str,
    host: str,
    path: str,
    body: bytes,
    region: str,
    access_key_id: str,
    secret_access_key: str,
    content_type: str | None = None,
    extra_signed_headers: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> dict[str, str]:
    """Return the headers (including `Authorization`) to send with an S3 request.

    `path` should start with `/` and already include any bucket prefix --
    e.g. `/my-bucket/object-key`. Query strings are not supported in v1
    (we never send them for PUT/GET/HEAD of a single object).

    `extra_signed_headers` (lowercase keys) are included in the canonical
    request and signature -- e.g. `x-amz-meta-*` user metadata headers.
    """
    timestamp = now or _utcnow()
    amz_date = timestamp.strftime('%Y%m%dT%H%M%SZ')
    date_stamp = timestamp.strftime('%Y%m%d')

    payload_hash = _hex_sha256(body)
    headers_to_sign: dict[str, str] = {
        'host': host,
        'x-amz-content-sha256': payload_hash,
        'x-amz-date': amz_date,
    }
    if content_type is not None:
        headers_to_sign['content-type'] = content_type
    if extra_signed_headers:
        for k, v in extra_signed_headers.items():
            headers_to_sign[k.lower()] = v

    sorted_keys = sorted(headers_to_sign)
    canonical_headers = ''.join(f'{k}:{headers_to_sign[k]}\n' for k in sorted_keys)
    signed_headers = ';'.join(sorted_keys)

    canonical_request = '\n'.join(
        [
            method.upper(),
            _canonical_uri(path),
            '',
            canonical_headers,
            signed_headers,
            payload_hash,
        ]
    )

    credential_scope = f'{date_stamp}/{region}/{_SERVICE}/aws4_request'
    string_to_sign = '\n'.join(
        [
            _ALGORITHM,
            amz_date,
            credential_scope,
            _hex_sha256(canonical_request.encode('utf-8')),
        ]
    )

    signature = hmac.new(
        _signing_key(secret_access_key, date_stamp, region),
        string_to_sign.encode('utf-8'),
        hashlib.sha256,
    ).hexdigest()

    authorization = (
        f'{_ALGORITHM} '
        f'Credential={access_key_id}/{credential_scope}, '
        f'SignedHeaders={signed_headers}, '
        f'Signature={signature}'
    )
    return {**headers_to_sign, 'authorization': authorization}


def _meta_headers_from_context(context: MediaContext) -> dict[str, str]:
    """Map `context.metadata` to `x-amz-meta-*` headers (lowercase, ASCII-safe keys).

    Raises `ValueError` for keys containing characters disallowed by HTTP
    header naming. Values are passed through verbatim -- callers must keep
    them ASCII-clean (S3 returns 400 for non-ASCII without `*=` RFC 5987
    encoding, which we don't support in v1).
    """
    headers: dict[str, str] = {}
    for key, value in context.metadata.items():
        if not _META_KEY_RE.fullmatch(key):
            raise ValueError(
                f'metadata key {key!r} is not a valid HTTP header token; use ASCII letters / digits / dashes'
            )
        headers[f'x-amz-meta-{key.lower()}'] = value
    return headers


class S3MediaStore:
    """S3-compatible content-addressed store using path-style URLs + SigV4.

    Works with AWS S3 (`endpoint='https://s3.<region>.amazonaws.com'`),
    Cloudflare R2 (`endpoint='https://<account_id>.r2.cloudflarestorage.com'`,
    `region='auto'`), MinIO, and other compatible providers.

    Customisation:

    - `key_strategy=` -- `Callable[[str, MediaContext], str]`. Default
      strategy yields `<key_prefix><sha256>.bin`. Override when the bucket
      layout demands a specific path / extension scheme. See the
      `KeyStrategy` docstring for the read-time caveat.
    - `public_url=` -- sync or async callable that takes
      `(uri, MediaContext)` and returns a URL (or `None`). Use
      `make_static_public_url(...)` for public bucket / CDN setups;
      provide your own async callable for presigned URLs (TTL captured in
      the closure; `MediaContext` available for content-type-specific
      response headers etc.). Without a resolver `public_url(...)` returns
      `None`.
    - `client=` -- bring your own `httpx.AsyncClient` for connection pooling.

    **Metadata persistence**: `context.metadata` is sent as signed
    `x-amz-meta-*` headers on PUT (subject to ASCII naming rules) and
    read back via `get_metadata(uri)`, which strips the `x-amz-meta-`
    prefix from the HEAD response headers.
    """

    def __init__(
        self,
        *,
        bucket: str,
        endpoint: str,
        region: str,
        access_key_id: str,
        secret_access_key: str,
        key_prefix: str = '',
        client: httpx.AsyncClient | None = None,
        key_strategy: KeyStrategy = default_key_strategy,
        public_url: PublicUrlResolver | None = None,
    ) -> None:
        self._bucket = bucket
        self._endpoint = endpoint.rstrip('/')
        self._region = region
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._key_prefix = key_prefix
        self._client = client
        self._key_strategy = key_strategy
        self._public_url_resolver = public_url

    def _object_path(self, uri: str, context: MediaContext) -> str:
        key = self._key_strategy(uri, context)
        return f'/{self._bucket}/{self._key_prefix}{key}'

    def _host(self) -> str:
        host = self._endpoint.split('://', 1)[1]
        return host.split('/', 1)[0]

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: bytes = b'',
        content_type: str | None = None,
        meta_headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        import httpx

        headers = sign_request(
            method=method,
            host=self._host(),
            path=path,
            body=body,
            region=self._region,
            access_key_id=self._access_key_id,
            secret_access_key=self._secret_access_key,
            content_type=content_type,
            extra_signed_headers=meta_headers,
        )
        # The signature covers `_canonical_uri(path)` (each segment percent-encoded).
        # Send that exact byte sequence as the wire path via `raw_path`, otherwise
        # httpx applies its own (looser) path encoding and a custom key_prefix /
        # key_strategy emitting reserved chars (`@`, `(`, `=`, ...) would diverge
        # from the signed path -> SignatureDoesNotMatch.
        url = httpx.URL(self._endpoint).copy_with(raw_path=_canonical_uri(path).encode('ascii'))
        if self._client is not None:
            return await self._client.request(method, url, content=body, headers=headers)
        async with httpx.AsyncClient() as client:
            return await client.request(method, url, content=body, headers=headers)

    async def put(self, data: bytes, *, context: MediaContext = _EMPTY_CONTEXT) -> str:
        uri = media_uri_for(data)
        meta = _meta_headers_from_context(context)
        response = await self._request(
            'PUT',
            self._object_path(uri, context),
            body=data,
            content_type=context.media_type,
            meta_headers=meta or None,
        )
        if response.status_code // 100 != 2:
            raise RuntimeError(f'S3 PUT failed for {uri}: {response.status_code} {response.text[:200]}')
        return uri

    async def get(self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT) -> bytes:
        digest = parse_media_uri(uri)
        response = await self._request('GET', self._object_path(uri, context))
        if response.status_code == 404:
            raise FileNotFoundError(f'media not found: {digest}')
        if response.status_code // 100 != 2:
            raise RuntimeError(f'S3 GET failed for {digest}: {response.status_code} {response.text[:200]}')
        return response.content

    async def exists(self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT) -> bool:
        digest = parse_media_uri(uri)
        response = await self._request('HEAD', self._object_path(uri, context))
        if response.status_code == 404:
            return False
        if response.status_code // 100 != 2:
            raise RuntimeError(f'S3 HEAD failed for {digest}: {response.status_code} {response.text[:200]}')
        return True

    async def public_url(self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT) -> str | None:
        return await _resolve_public_url(self._public_url_resolver, uri, context)

    async def get_metadata(self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT) -> Mapping[str, str]:
        digest = parse_media_uri(uri)
        response = await self._request('HEAD', self._object_path(uri, context))
        if response.status_code == 404:
            raise FileNotFoundError(f'media not found: {digest}')
        if response.status_code // 100 != 2:
            raise RuntimeError(f'S3 HEAD failed for {digest}: {response.status_code} {response.text[:200]}')
        out: dict[str, str] = {}
        for key, value in response.headers.items():
            lower = key.lower()
            if lower.startswith('x-amz-meta-'):
                out[lower[len('x-amz-meta-') :]] = value
        return out
