"""Dependency-free S3-compatible FileStore for LLM media workflows.

Implements AWS Signature Version 4 signing using only stdlib (for signing).
Uses httpx for async HTTP. Works with any S3-compatible storage (AWS S3, R2, MinIO, etc.)
"""

import hashlib
import hmac
from datetime import datetime, timezone
from urllib.parse import quote, urlparse

# =============================================================================
# AWS Signature V4 Implementation
# =============================================================================


def _sha256_hex(data: bytes) -> str:
    """SHA256 hash as lowercase hex string."""
    return hashlib.sha256(data).hexdigest()


def _hmac_sha256(key: bytes, msg: str) -> bytes:
    """HMAC-SHA256 signature."""
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()


def _derive_signing_key(secret: str, date: str, region: str, service: str) -> bytes:
    """Derive the signing key through the HMAC chain.

    secret -> date -> region -> service -> aws4_request
    """
    k_date = _hmac_sha256(f'AWS4{secret}'.encode(), date)
    k_region = _hmac_sha256(k_date, region)
    k_service = _hmac_sha256(k_region, service)
    k_signing = _hmac_sha256(k_service, 'aws4_request')
    return k_signing


def _uri_encode(s: str, encode_slash: bool = True) -> str:
    """URI encode per AWS spec.

    - Encode everything except unreserved chars: A-Z, a-z, 0-9, -, _, ., ~
    - Optionally preserve forward slashes for paths
    """
    safe = '-_.~' if encode_slash else '-_.~/'
    return quote(s, safe=safe)


def _create_canonical_request(
    method: str,
    path: str,
    query_string: str,
    headers: dict[str, str],
    signed_headers: list[str],
    payload_hash: str,
) -> str:
    """Create the canonical request string per AWS SigV4 spec."""
    canonical_uri = _uri_encode(path, encode_slash=False) or '/'
    canonical_query = query_string

    canonical_headers = ''
    for key in sorted(signed_headers):
        value = headers[key].strip()
        canonical_headers += f'{key}:{value}\n'

    signed_headers_str = ';'.join(sorted(signed_headers))

    return '\n'.join(
        [
            method,
            canonical_uri,
            canonical_query,
            canonical_headers,
            signed_headers_str,
            payload_hash,
        ]
    )


def _create_string_to_sign(
    timestamp: str,
    date: str,
    region: str,
    service: str,
    canonical_request: str,
) -> str:
    """Create the string to sign."""
    algorithm = 'AWS4-HMAC-SHA256'
    credential_scope = f'{date}/{region}/{service}/aws4_request'
    hashed_request = _sha256_hex(canonical_request.encode('utf-8'))

    return '\n'.join(
        [
            algorithm,
            timestamp,
            credential_scope,
            hashed_request,
        ]
    )


def sign_request(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes,
    access_key: str,
    secret_key: str,
    region: str,
    service: str = 's3',
) -> dict[str, str]:
    """Sign an HTTP request using AWS Signature Version 4.

    Returns:
        a new headers dict with Authorization and required headers added.
    """
    parsed = urlparse(url)
    host = parsed.netloc
    path = parsed.path or '/'
    query_string = parsed.query or ''

    now = datetime.now(timezone.utc)
    timestamp = now.strftime('%Y%m%dT%H%M%SZ')
    date = now.strftime('%Y%m%d')

    payload_hash = _sha256_hex(body)

    headers = {k.lower(): v for k, v in headers.items()}
    headers['host'] = host
    headers['x-amz-date'] = timestamp
    headers['x-amz-content-sha256'] = payload_hash

    signed_headers = sorted(headers.keys())

    canonical_request = _create_canonical_request(
        method=method,
        path=path,
        query_string=query_string,
        headers=headers,
        signed_headers=signed_headers,
        payload_hash=payload_hash,
    )

    string_to_sign = _create_string_to_sign(
        timestamp=timestamp,
        date=date,
        region=region,
        service=service,
        canonical_request=canonical_request,
    )

    signing_key = _derive_signing_key(secret_key, date, region, service)
    signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()

    credential_scope = f'{date}/{region}/{service}/aws4_request'
    signed_headers_str = ';'.join(signed_headers)
    authorization = (
        f'AWS4-HMAC-SHA256 '
        f'Credential={access_key}/{credential_scope}, '
        f'SignedHeaders={signed_headers_str}, '
        f'Signature={signature}'
    )

    headers['authorization'] = authorization
    return headers


def create_presigned_url(
    url: str,
    access_key: str,
    secret_key: str,
    region: str,
    expires_in: int,
    service: str = 's3',
) -> str:
    """Create a presigned URL for GET requests.

    Args:
        url: The full object URL
        access_key: AWS access key
        secret_key: AWS secret key
        region: AWS region
        expires_in: TTL in seconds
        service: Service name (default: s3)

    Returns:
        Presigned URL with signature in query string
    """
    parsed = urlparse(url)
    host = parsed.netloc
    path = parsed.path or '/'

    now = datetime.now(timezone.utc)
    timestamp = now.strftime('%Y%m%dT%H%M%SZ')
    date = now.strftime('%Y%m%d')

    credential_scope = f'{date}/{region}/{service}/aws4_request'
    credential = f'{access_key}/{credential_scope}'

    # Build canonical query string (must be sorted)
    query_params = {
        'X-Amz-Algorithm': 'AWS4-HMAC-SHA256',
        'X-Amz-Credential': credential,
        'X-Amz-Date': timestamp,
        'X-Amz-Expires': str(expires_in),
        'X-Amz-SignedHeaders': 'host',
    }
    canonical_query = '&'.join(f'{_uri_encode(k)}={_uri_encode(v)}' for k, v in sorted(query_params.items()))

    # For presigned URLs, payload is always UNSIGNED-PAYLOAD
    canonical_request = _create_canonical_request(
        method='GET',
        path=path,
        query_string=canonical_query,
        headers={'host': host},
        signed_headers=['host'],
        payload_hash='UNSIGNED-PAYLOAD',
    )

    string_to_sign = _create_string_to_sign(
        timestamp=timestamp,
        date=date,
        region=region,
        service=service,
        canonical_request=canonical_request,
    )

    signing_key = _derive_signing_key(secret_key, date, region, service)
    signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()

    return f'{parsed.scheme}://{host}{path}?{canonical_query}&X-Amz-Signature={signature}'


class S3Error(Exception):
    """S3 operation error."""

    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(f'S3 Error {status}: {message}')
