"""VCR configuration for `S3MediaStore` cassette tests.

The cassettes are committed alongside the tests so CI can replay them
without R2 / AWS credentials. Recording is opt-in via
`pytest --record-mode=once` (or `=new_episodes`/`=all`) with real
`S3_*` env vars; replay is the default mode and what CI runs.

Sanitisation policy: every recorded cassette is rewritten on disk to
swap the real R2 account-id subdomain and bucket name for fixed
placeholders, and drop the `Authorization` header. The test setup uses
the placeholder endpoint + bucket so request matching still succeeds on
replay. See `_rewrite_request` / `_rewrite_response` below.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportMissingTypeStubs=false
from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from vcr.request import Request as VcrRequest  # pyright: ignore[reportMissingTypeStubs]

# Public placeholders baked into the committed cassettes. Tests pass
# these *exact* values when constructing `S3MediaStore`, so the replay
# URI matches the recorded URI even though both were sanitised.
SANITIZED_HOST = 'account.r2.cloudflarestorage.com'
SANITIZED_BUCKET = 'harness-test-bucket'
SANITIZED_ENDPOINT = f'https://{SANITIZED_HOST}'
SANITIZED_REGION = 'auto'


def _real_account_host_pattern() -> re.Pattern[str] | None:  # pragma: no cover
    """Build a regex that matches the real R2 host so we can scrub it."""
    endpoint = os.environ.get('S3_ENDPOINT')
    if not endpoint:
        return None
    match = re.match(r'https?://([^/]+)', endpoint)
    if not match:
        return None
    return re.compile(re.escape(match.group(1)))


def _real_bucket_pattern() -> re.Pattern[str] | None:  # pragma: no cover
    bucket = os.environ.get('S3_BUCKET_NAME')
    if not bucket:
        return None
    return re.compile(r'/' + re.escape(bucket) + r'/')


def _rewrite_request(request: VcrRequest) -> VcrRequest:  # pragma: no cover
    """Strip account-id, bucket name, and credentials from recorded request."""
    host_pat = _real_account_host_pattern()
    if host_pat is not None:
        request.uri = host_pat.sub(SANITIZED_HOST, request.uri)
    bucket_pat = _real_bucket_pattern()
    if bucket_pat is not None:
        request.uri = bucket_pat.sub(f'/{SANITIZED_BUCKET}/', request.uri)
    # VCR already drops Authorization via `filter_headers`, but Host is set
    # by httpx independently — overwrite it so cassettes never carry the
    # real account subdomain.
    if 'host' in request.headers:
        request.headers['host'] = SANITIZED_HOST
    return request


_DROP_RESPONSE_HEADERS = frozenset(
    {
        'cf-ray',
        'cf-cache-status',
        'x-amz-version-id',
        'x-amz-request-id',
        'x-amz-id-2',
        'x-amz-checksum-crc64nvme',
        'x-amz-checksum-crc32',
        'x-amz-checksum-crc32c',
        'x-amz-checksum-sha1',
        'x-amz-checksum-sha256',
    }
)


def _rewrite_response(response: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
    """Sanitise the response: drop noisy / identifying headers and any error body.

    For non-2xx responses (typically the gzipped XML R2/AWS error envelope,
    which can mention the bucket) we blank the body entirely and strip
    `Content-Encoding`. Our `S3MediaStore` only inspects `status_code` for
    4xx and `response.text[:200]` for 5xx — no test in this module relies
    on the error body shape.
    """
    host_pat = _real_account_host_pattern()
    bucket_pat = _real_bucket_pattern()
    headers = response.get('headers', {})
    for header_name in list(headers.keys()):
        if header_name.lower() in _DROP_RESPONSE_HEADERS:
            del headers[header_name]
            continue
        values = headers[header_name]
        if not isinstance(values, list):
            continue
        new_values: list[str] = []
        for v in values:
            if not isinstance(v, str):
                new_values.append(v)
                continue
            if host_pat is not None:
                v = host_pat.sub(SANITIZED_HOST, v)
            if bucket_pat is not None:
                v = bucket_pat.sub(f'/{SANITIZED_BUCKET}/', v)
            new_values.append(v)
        headers[header_name] = new_values

    status = response.get('status', {})
    code = status.get('code') if isinstance(status, dict) else None
    is_success = isinstance(code, int) and 200 <= code < 300
    body = response.get('body', {})
    if not is_success and isinstance(body, dict):
        # Drop any provider error envelope — it can name the bucket inside
        # the gzipped XML. Tests only read the status code on this path.
        body['string'] = b''
        for header_name in list(headers.keys()):
            if header_name.lower() in ('content-encoding', 'content-length', 'transfer-encoding'):
                del headers[header_name]
    return response


@pytest.fixture(scope='module')
def vcr_config() -> dict[str, Any]:
    """Per-module VCR configuration. Cassettes live next to the tests.

    Matching: method + scheme + host + path + body. Headers (including
    SigV4 `authorization` and `x-amz-date`) are NOT part of matching —
    they regenerate per replay and would otherwise miss every time.

    Record mode is whatever `--record-mode` says (default `none`).
    """
    return {
        'filter_headers': [
            ('authorization', 'REDACTED'),
            ('x-amz-date', 'REDACTED'),
        ],
        'before_record_request': _rewrite_request,
        'before_record_response': _rewrite_response,
        'match_on': ['method', 'scheme', 'host', 'path', 'body'],
    }


@pytest.fixture
def anyio_backend() -> str:
    """Restrict the S3 cassette tests to asyncio — we don't need trio cassettes."""
    return 'asyncio'


@pytest.fixture
def s3_credentials() -> dict[str, str]:
    """Real R2 creds when env is set; sanitised placeholders otherwise.

    The placeholders match the values baked into the scrubbed cassettes
    (see `_rewrite_request` / `_rewrite_response` above), so replay works
    against `tests/media/cassettes/` with no env vars at all — exactly what
    CI runs.

    **Why the placeholders double as a leakage canary:** if the scrubber
    ever misses a value when re-recording, the cassette will contain the
    real bucket / account id while the replay test still constructs URLs
    from the placeholder constants — the URL matcher will fail and CI
    will surface the leak. Reusing this pattern across the suite (always
    pass placeholder values, scrub on write) catches accidental
    credential / private-data exposure in committed cassettes.

    `region` is hardcoded to `'auto'` because R2 rejects every other name
    and the SigV4 region is part of the credential scope (filtered from
    the cassette `Authorization`, so it does not affect replay matching).
    Override the fixture in another conftest if recording against AWS S3.
    """
    return {
        'bucket': os.environ.get('S3_BUCKET_NAME', SANITIZED_BUCKET),
        'endpoint': os.environ.get('S3_ENDPOINT', SANITIZED_ENDPOINT),
        'region': 'auto',
        'access_key_id': os.environ.get('S3_ACCESS_KEY_ID', 'AKIAIOSFODNN7EXAMPLE'),
        'secret_access_key': os.environ.get('S3_SECRET_ACCESS_KEY', 'REDACTED-FOR-REPLAY'),
    }


@pytest.fixture
def s3_store(s3_credentials: dict[str, str]) -> Any:
    """`S3MediaStore` built from the credentials fixture, with a fixed key prefix.

    The key prefix is part of the URL path that lands in the cassette, so
    keep it stable across re-records.
    """
    from pydantic_ai_harness.media import S3MediaStore

    return S3MediaStore(
        bucket=s3_credentials['bucket'],
        endpoint=s3_credentials['endpoint'],
        region=s3_credentials['region'],
        access_key_id=s3_credentials['access_key_id'],
        secret_access_key=s3_credentials['secret_access_key'],
        key_prefix='harness-vcr/',
    )
