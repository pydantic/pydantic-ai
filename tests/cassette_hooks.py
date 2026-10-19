"""What cassetter records, and how a request is matched, across the whole suite.

`before_record_request` / `before_record_response` run on live traffic before it is written to a cassette:
they drop noisy headers, scrub credentials the built-in filters don't know about, and normalize the smart
quotes LLM APIs return so cassettes and the snapshots built from them stay ASCII. `normalize_uri` runs on
both recorded and incoming URIs at match time, so a cassette recorded against one AWS account or GCP
project replays for another.
"""

from __future__ import annotations

import gzip
import json
import re
import unicodedata
import urllib.parse
import zlib
from typing import Any, cast

import brotli
from cassetter import RawRequest, RawResponse, SkipRecording

from pydantic_ai._utils import is_str_dict

# Smart quote and special character normalization.
# LLM APIs sometimes return smart quotes and special Unicode characters in responses.
# These are captured in cassettes, which then populate snapshots
# which in turn cause linter complaints about non-ASCII characters.
# Fixing these manually in the snapshots doesn't help,
# because the snapshots are asserted on test reruns against the cassettes.
# Normalizing to ASCII equivalents ensures consistent, portable cassette files and stable snapshots.
SMART_CHAR_MAP = {
    '\u2018': "'",  # LEFT SINGLE QUOTATION MARK
    '\u2019': "'",  # RIGHT SINGLE QUOTATION MARK
    '\u201c': '"',  # LEFT DOUBLE QUOTATION MARK
    '\u201d': '"',  # RIGHT DOUBLE QUOTATION MARK
    '\u2013': '-',  # EN DASH
    '\u2014': '--',  # EM DASH
    '\u2026': '...',  # HORIZONTAL ELLIPSIS
}
SMART_CHAR_TRANS = str.maketrans(SMART_CHAR_MAP)


def normalize_smart_chars(text: str) -> str:
    """Normalize smart quotes and special characters to ASCII equivalents."""
    # First use the translation table for known characters
    text = text.translate(SMART_CHAR_TRANS)
    # Then apply NFKC normalization for any remaining special chars
    return unicodedata.normalize('NFKC', text)


def normalize_body(obj: Any) -> Any:
    """Recursively normalize smart characters in all strings within a data structure."""
    if isinstance(obj, str):
        return normalize_smart_chars(obj)
    elif isinstance(obj, dict):
        return {k: normalize_body(v) for k, v in cast('dict[Any, Any]', obj).items()}
    elif isinstance(obj, list):
        return [normalize_body(item) for item in cast('list[Any]', obj)]
    return obj


FILTERED_HEADER_PREFIXES = ['anthropic-', 'cf-', 'x-']
FILTERED_HEADERS = {
    'authorization',
    'chatgpt-account-id',
    'cookie',
    'date',
    'openai-organization',
    'openai-project',
    'request-id',
    'server',
    'user-agent',
    'via',
    'set-cookie',
    'api-key',
}
ALLOWED_HEADER_PREFIXES = {
    # required by huggingface_hub.file_download used by test_embeddings.py::TestSentenceTransformers
    'x-xet-',
    # required for Bedrock embeddings to preserve token count headers
    'x-amzn-bedrock-',
}
ALLOWED_HEADERS = {
    # required by huggingface_hub.file_download used by test_embeddings.py::TestSentenceTransformers
    'x-repo-commit',
    'x-linked-size',
    'x-linked-etag',
    # required for test_google_model_file_search_tool
    'x-goog-upload-url',
    'x-goog-upload-status',
    # recorded as `gen_ai.response.id` on TypeSafe's `decide` spans
    'x-typesafe-request-id',
}

SCRUBBED_JSON_FIELDS = ('access_token', 'id_token', 'refresh_token', 'safety_identifier')
SCRUBBED_FORM_FIELDS = (
    'assertion',
    'client_id',
    'client_secret',
    'code',
    'code_verifier',
    'refresh_token',
    'RoleArn',
    'RoleSessionName',
)

# Token exchanges carry credentials in both directions and are never what a test is about.
SKIPPED_ENDPOINTS = {
    ('oauth2.googleapis.com', '/token'),
    ('auth.openai.com', '/oauth/token'),
}

_AWS_ACCOUNT_ID_IN_ARN = re.compile(r'(arn(?:%3A|:)aws(?:%3A|:)bedrock(?:%3A|:)[^:%]*(?:%3A|:))\d{12}((?:%3A|:))')
_SCRUBBED_AWS_ACCOUNT_ID = r'\g<1>123456789012\2'
_BEDROCK_HOST = re.compile(r'bedrock-runtime\.[a-z0-9-]+\.amazonaws\.com')
_VERTEX_HOST = re.compile(r'[a-z0-9-]+-aiplatform\.googleapis\.com')
_VERTEX_LOCATION = re.compile(r'/locations/[a-z0-9-]+/')
_VERTEX_PROJECT = re.compile(r'/projects/[a-z0-9-]+/')
_SAFETY_IDENTIFIER = re.compile(r'("safety_identifier"\s*:\s*)"(?:\\.|[^"\\])*"')


def scrub_aws_account_id(uri: str) -> str:
    return _AWS_ACCOUNT_ID_IN_ARN.sub(_SCRUBBED_AWS_ACCOUNT_ID, uri)


def normalize_uri(uri: str) -> str:
    """Erase the region, project and account a cassette was recorded against before matching.

    Bedrock cassettes are recorded in whatever region the recording developer's credentials use, and
    Vertex AI cassettes against their GCP project; neither is what the test asserts on.
    """
    uri = scrub_aws_account_id(uri)
    uri = _BEDROCK_HOST.sub('bedrock-runtime.REGION.amazonaws.com', uri)
    uri = _VERTEX_HOST.sub('aiplatform.googleapis.com', uri)
    uri = _VERTEX_LOCATION.sub('/locations/REGION/', uri)
    return _VERTEX_PROJECT.sub('/projects/PROJECT/', uri)


def filter_headers(headers: dict[str, list[str]]) -> dict[str, list[str]]:
    """Drop headers that carry credentials, request IDs or other per-run noise."""
    return {
        name: value
        for name, value in ((name.lower(), value) for name, value in headers.items())
        if name not in FILTERED_HEADERS
        and (
            not any(name.startswith(prefix) for prefix in FILTERED_HEADER_PREFIXES)
            or name in ALLOWED_HEADERS
            or any(name.startswith(prefix) for prefix in ALLOWED_HEADER_PREFIXES)
        )
    }


def _content_type(headers: dict[str, list[str]]) -> str:
    return next(iter(headers.get('content-type', [])), '')


def _decompress(body: bytes, headers: dict[str, list[str]]) -> bytes:
    """Inflate a body so it can be scrubbed (botocore hands over the wire bytes).

    `content-encoding` is only dropped once the body is actually inflated; anything left encoded
    is decoded by cassetter after the hook.
    """
    encoding = headers.get('content-encoding', [])
    if 'br' in encoding:
        body = cast('bytes', brotli.decompress(body))  # pyright: ignore[reportUnknownMemberType]
    elif 'gzip' in encoding or body[:2] == b'\x1f\x8b':
        try:
            body = gzip.decompress(body)
        except (gzip.BadGzipFile, zlib.error):
            return body
    else:
        return body
    headers.pop('content-encoding', None)
    return body


def scrub_json_body(body: bytes) -> bytes:
    """Normalize smart characters and blank the credentials a JSON body may carry.

    Some endpoints (e.g. resumable file uploads) send a non-JSON body under an `application/json`
    content-type; keep the raw body rather than crashing the hook.
    """
    try:
        parsed: Any = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body
    parsed = normalize_body(parsed)
    if is_str_dict(parsed):
        for field in SCRUBBED_JSON_FIELDS:
            if parsed.get(field) is not None:
                parsed[field] = 'scrubbed'
    return json.dumps(parsed).encode('utf-8')


def scrub_form_credentials(body: bytes) -> bytes:
    """Redact credentials from an `application/x-www-form-urlencoded` body."""
    query_params = urllib.parse.parse_qs(body.decode('utf-8'))
    for key in SCRUBBED_FORM_FIELDS:
        if key in query_params:
            query_params[key] = ['scrubbed']
    return urllib.parse.urlencode(query_params, doseq=True).encode('utf-8')


def scrub_xml_credentials(body: bytes) -> bytes:
    """Redact AWS STS credentials from a `text/xml` body."""
    text = body.decode('utf-8')
    if '<Credentials>' not in text:
        return body
    text = re.sub(r'<AccessKeyId>[^<]+</AccessKeyId>', '<AccessKeyId>SCRUBBED</AccessKeyId>', text)
    text = re.sub(r'<SecretAccessKey>[^<]+</SecretAccessKey>', '<SecretAccessKey>SCRUBBED</SecretAccessKey>', text)
    text = re.sub(r'<SessionToken>[^<]+</SessionToken>', '<SessionToken>SCRUBBED</SessionToken>', text)
    text = re.sub(r'<Expiration>[^<]+</Expiration>', '<Expiration>2099-01-01T00:00:00Z</Expiration>', text)
    text = re.sub(r'<AssumedRoleId>[^<]+</AssumedRoleId>', '<AssumedRoleId>SCRUBBED</AssumedRoleId>', text)
    text = re.sub(r'<Arn>[^<]+</Arn>', '<Arn>SCRUBBED</Arn>', text)
    return text.encode('utf-8')


def scrub_body(body: bytes | None, headers: dict[str, list[str]]) -> bytes | None:
    if not body:
        return body
    body = _decompress(body, headers)
    content_type = _content_type(headers)
    if content_type.startswith('application/json'):
        return scrub_json_body(body)
    if content_type.startswith('application/x-www-form-urlencoded'):
        return scrub_form_credentials(body)
    if content_type == 'text/xml':
        return scrub_xml_credentials(body)
    # Codex SSE responses can omit the content-type header.
    if body.startswith((b'event:', b'data:')):
        return _SAFETY_IDENTIFIER.sub(r'\1"scrubbed"', body.decode('utf-8')).encode('utf-8')
    return body


def before_record_request(request: RawRequest) -> RawRequest:
    parsed = urllib.parse.urlparse(request.uri)
    if (parsed.hostname, parsed.path) in SKIPPED_ENDPOINTS:
        raise SkipRecording
    request.uri = scrub_aws_account_id(request.uri)
    request.headers = filter_headers(request.headers)
    request.body = scrub_body(request.body, request.headers)
    return request


def before_record_response(response: RawResponse) -> RawResponse:
    response.headers = filter_headers(response.headers)
    response.body = scrub_body(response.body, response.headers)
    return response
