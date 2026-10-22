"""VCR configuration for `S3MediaStore` cassette tests.

The cassettes are committed alongside the tests so CI can replay them
without R2 / AWS credentials. Recording is opt-in via
`pytest --record-mode=once` (or `=new_episodes`/`=all`) with real
`S3_*` env vars; replay is the default mode and what CI runs.

Sanitisation policy: every recorded cassette is rewritten on disk to
swap the real R2 account-id subdomain and bucket name for fixed
placeholders; the repo-wide hooks drop the `Authorization` header. The
test setup uses the placeholder endpoint + bucket so request matching
still succeeds on replay. See `_rewrite_request` / `_rewrite_response` below.
"""

from __future__ import annotations

import importlib.util
import os
import re
from typing import Any

import pytest
from cassetter import RawRequest, RawResponse

from pydantic_ai_harness.media import S3MediaStore
from tests import cassette_hooks

# `pymongo` is gated on the `mongodb` extra, so an install without it can't import
# the Mongo store tests. Ignore them at collection then. A conditional expression
# rather than an `if` statement: branch coverage traces statement arcs, and no
# single environment can take both arms of an install-dependent branch.
collect_ignore = ['test_mongo.py'] if importlib.util.find_spec('pymongo') is None else []

# Public placeholders baked into the committed cassettes. Tests pass
# these *exact* values when constructing `S3MediaStore`, so the replay
# URI matches the recorded URI even though both were sanitised.
SANITIZED_HOST = 'account.r2.cloudflarestorage.com'
SANITIZED_BUCKET = 'harness-test-bucket'
SANITIZED_ENDPOINT = f'https://{SANITIZED_HOST}'
SANITIZED_REGION = 'auto'


def _real_account_host_pattern() -> re.Pattern[str] | None:  # pragma: lax no cover
    """Build a regex that matches the real R2 host so we can scrub it."""
    endpoint = os.environ.get('S3_ENDPOINT')
    if not endpoint:
        return None
    match = re.match(r'https?://([^/]+)', endpoint)
    if not match:
        return None
    return re.compile(re.escape(match.group(1)))


def _real_bucket_pattern() -> re.Pattern[str] | None:  # pragma: lax no cover
    bucket = os.environ.get('S3_BUCKET_NAME')
    if not bucket:
        return None
    return re.compile(r'/' + re.escape(bucket) + r'/')


def _rewrite_request(request: RawRequest) -> RawRequest:  # pragma: lax no cover
    """Strip account-id, bucket name, and credentials from recorded request."""
    request = cassette_hooks.before_record_request(request)
    host_pat = _real_account_host_pattern()
    if host_pat is not None:
        request.uri = host_pat.sub(SANITIZED_HOST, request.uri)
    bucket_pat = _real_bucket_pattern()
    if bucket_pat is not None:
        request.uri = bucket_pat.sub(f'/{SANITIZED_BUCKET}/', request.uri)
    # The repo-wide hooks drop `authorization` and `x-amz-date`, but `host` is set by httpx
    # independently — overwrite it so cassettes never carry the real account subdomain.
    if 'host' in request.headers:
        request.headers['host'] = [SANITIZED_HOST]
    return request


def _rewrite_response(response: RawResponse) -> RawResponse:  # pragma: lax no cover
    """Sanitise the response: drop noisy / identifying headers and any error body.

    The repo-wide hooks already drop the `cf-*` and `x-amz-*` request ids and checksums. For non-2xx
    responses (typically the gzipped XML R2/AWS error envelope, which can mention the bucket) we blank
    the body entirely and strip `Content-Encoding`. Our `S3MediaStore` only inspects `status_code` for
    4xx and `response.text[:200]` for 5xx — no test in this module relies on the error body shape.
    """
    response = cassette_hooks.before_record_response(response)
    host_pat = _real_account_host_pattern()
    bucket_pat = _real_bucket_pattern()
    for header_name, values in response.headers.items():
        new_values: list[str] = []
        for v in values:
            if host_pat is not None:
                v = host_pat.sub(SANITIZED_HOST, v)
            if bucket_pat is not None:
                v = bucket_pat.sub(f'/{SANITIZED_BUCKET}/', v)
            new_values.append(v)
        response.headers[header_name] = new_values

    if not 200 <= response.status < 300:
        # Drop any provider error envelope — it can name the bucket inside
        # the gzipped XML. Tests only read the status code on this path.
        response.body = b''
        for header_name in ('content-encoding', 'content-length', 'transfer-encoding'):
            response.headers.pop(header_name, None)
    return response


@pytest.fixture(scope='module')
def vcr_config(vcr_config: dict[str, Any]) -> dict[str, Any]:
    """Per-module cassette configuration. Cassettes live next to the tests.

    Matching: method + URI + body. Headers (including SigV4 `authorization`
    and `x-amz-date`) are NOT part of matching — they regenerate per replay
    and would otherwise miss every time.

    Record mode is whatever `--record-mode` says (default `none`).
    """
    return {
        **vcr_config,
        'before_record_request': _rewrite_request,
        'before_record_response': _rewrite_response,
        'match_on': ['method', 'uri', 'body'],
    }


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

    return S3MediaStore(
        bucket=s3_credentials['bucket'],
        endpoint=s3_credentials['endpoint'],
        region=s3_credentials['region'],
        access_key_id=s3_credentials['access_key_id'],
        secret_access_key=s3_credentials['secret_access_key'],
        key_prefix='harness-vcr/',
    )
