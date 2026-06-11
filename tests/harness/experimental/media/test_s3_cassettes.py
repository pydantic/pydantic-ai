"""VCR cassette tests for `S3MediaStore` against a real R2 endpoint.

The cassettes under `tests/media/cassettes/` were recorded against a
Cloudflare R2 bucket and then sanitised (account id, bucket name,
`Authorization`, `x-amz-date`, identifying response headers, error
bodies) by hooks in `conftest.py::vcr_config`. CI replays them — no
creds needed.

To re-record (or add new cases):

```
~/.claude/scripts/env-run .env -- uv run pytest \\
    tests/media/test_s3_cassettes.py \\
    --record-mode=once
```

Recording reads real R2 creds from the `s3_credentials` fixture; replay
falls back to the sanitised placeholders that match the committed
cassettes. Drift between real values and placeholders (e.g. scrubber
misses a field) shows up as a replay-time URL mismatch — the canary.
"""

from __future__ import annotations

import pytest

from pydantic_ai_harness.experimental.media import MediaContext, S3MediaStore, media_uri_for

pytestmark = pytest.mark.anyio

# Deterministic payload so the URI / object key are stable across re-records.
_PAYLOAD = b'pydantic-ai-harness step-persistence VCR cassette payload v1'
_MISSING_URI = 'media+sha256://' + ('0' * 64)
_CONTEXT = MediaContext(media_type='application/octet-stream')


class TestS3MediaStoreCassettes:
    """Replay-driven verification of `S3MediaStore` against a real R2 server."""

    @pytest.mark.vcr
    async def test_put_succeeds(self, s3_store: S3MediaStore) -> None:
        uri = await s3_store.put(_PAYLOAD, context=_CONTEXT)
        assert uri == media_uri_for(_PAYLOAD)

    @pytest.mark.vcr
    async def test_exists_present_after_put(self, s3_store: S3MediaStore) -> None:
        await s3_store.put(_PAYLOAD, context=_CONTEXT)
        assert await s3_store.exists(media_uri_for(_PAYLOAD)) is True

    @pytest.mark.vcr
    async def test_exists_returns_false_for_missing(self, s3_store: S3MediaStore) -> None:
        assert await s3_store.exists(_MISSING_URI) is False

    @pytest.mark.vcr
    async def test_get_round_trips_bytes(self, s3_store: S3MediaStore) -> None:
        await s3_store.put(_PAYLOAD, context=_CONTEXT)
        fetched = await s3_store.get(media_uri_for(_PAYLOAD))
        assert fetched == _PAYLOAD

    @pytest.mark.vcr
    async def test_get_missing_raises(self, s3_store: S3MediaStore) -> None:
        with pytest.raises(FileNotFoundError):
            await s3_store.get(_MISSING_URI)
