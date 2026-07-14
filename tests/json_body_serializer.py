# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
import gzip
import json
import re
import unicodedata
import urllib.parse
import zlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import brotli
import yaml

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
        return {k: normalize_body(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [normalize_body(item) for item in obj]
    return obj  # pragma: no cover


if TYPE_CHECKING:
    from yaml import Dumper, SafeLoader
else:
    try:
        from yaml import CDumper as Dumper, CSafeLoader as SafeLoader
    except ImportError:  # pragma: no cover
        from yaml import Dumper, SafeLoader

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
_OPENAI_OAUTH_JSON_CREDENTIAL_KEYS = frozenset(
    {
        'access_token',
        'account_id',
        'authorization_code',
        'chatgpt_account_id',
        'client_secret',
        'code',
        'code_verifier',
        'device_auth_id',
        'device_code',
        'id_token',
        'refresh_token',
        'token',
        'user_code',
    }
)
_CODEX_SSE_SENSITIVE_KEYS = frozenset({'prompt_cache_key', 'safety_identifier'})
_LEGACY_TOP_LEVEL_CREDENTIAL_KEYS = frozenset({'access_token', 'id_token'})

ALLOWED_HEADERS = {
    # required by huggingface_hub.file_download used by test_embeddings.py::TestSentenceTransformers
    'x-repo-commit',
    'x-linked-size',
    'x-linked-etag',
    # required for test_google_model_file_search_tool
    'x-goog-upload-url',
    'x-goog-upload-status',
}


class LiteralDumper(Dumper):
    """
    A custom dumper that will represent multi-line strings using literal style.
    """


def str_presenter(dumper: Dumper, data: str):
    """If the string contains newlines, represent it as a literal block."""
    if '\n' in data:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)


# Register the custom presenter on our dumper
LiteralDumper.add_representer(str, str_presenter)


def deserialize(cassette_string: str):
    cassette_dict = yaml.load(cassette_string, Loader=SafeLoader)
    for interaction in cassette_dict['interactions']:
        for kind, data in interaction.items():
            parsed_body = data.pop('parsed_body', None)
            if parsed_body is not None:
                dumped_body = json.dumps(parsed_body)
                data['body'] = {'string': dumped_body} if kind == 'response' else dumped_body
    return cassette_dict


def _content_type_startswith(content_type: Sequence[str | bytes], prefix: str) -> bool:
    return any(
        (h if isinstance(h, str) else h.decode('utf-8') if isinstance(h, bytes) else '').startswith(prefix)
        for h in content_type
    )


def scrub_form_credentials(
    data: dict[str, Any], content_type: list[str], *, openai_oauth: bool = False
) -> None:  # pragma: lax no cover
    """Redact credentials from application/x-www-form-urlencoded request bodies."""
    if not _content_type_startswith(content_type, 'application/x-www-form-urlencoded'):
        return
    query_params = urllib.parse.parse_qs(data['body'])
    keys = {
        'assertion',
        'client_id',
        'client_secret',
        'refresh_token',
        'RoleArn',
        'RoleSessionName',
    }
    if openai_oauth:
        keys.update({'code', 'code_verifier', 'device_code', 'token', 'user_code'})
    for key in keys:
        if key in query_params:
            query_params[key] = ['scrubbed']
            data['body'] = urllib.parse.urlencode(query_params, doseq=True)


def scrub_xml_credentials(
    data: dict[str, Any], headers: dict[str, list[str]], content_type: list[str]
) -> None:  # pragma: lax no cover
    """Redact AWS STS credentials from text/xml response bodies."""
    if content_type != ['text/xml']:
        return
    body = data.get('body', None)
    if isinstance(body, dict):
        body = body.get('string', '')
    if not isinstance(body, str) or '<Credentials>' not in body:
        return
    body = re.sub(r'<AccessKeyId>[^<]+</AccessKeyId>', '<AccessKeyId>SCRUBBED</AccessKeyId>', body)
    body = re.sub(r'<SecretAccessKey>[^<]+</SecretAccessKey>', '<SecretAccessKey>SCRUBBED</SecretAccessKey>', body)
    body = re.sub(r'<SessionToken>[^<]+</SessionToken>', '<SessionToken>SCRUBBED</SessionToken>', body)
    body = re.sub(r'<Expiration>[^<]+</Expiration>', '<Expiration>2099-01-01T00:00:00Z</Expiration>', body)
    body = re.sub(r'<AssumedRoleId>[^<]+</AssumedRoleId>', '<AssumedRoleId>SCRUBBED</AssumedRoleId>', body)
    body = re.sub(r'<Arn>[^<]+</Arn>', '<Arn>SCRUBBED</Arn>', body)
    data['body'] = {'string': body}
    if 'content-length' in headers:
        headers['content-length'] = [str(len(body.encode('utf-8')))]


def scrub_json_credentials(value: Any, sensitive_keys: frozenset[str]) -> Any:
    """Recursively redact selected credential fields from JSON-compatible data."""
    if isinstance(value, dict):
        return {
            key: 'scrubbed'
            if isinstance(key, str) and key.lower() in sensitive_keys
            else scrub_json_credentials(item, sensitive_keys)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [scrub_json_credentials(item, sensitive_keys) for item in value]
    return value


def scrub_sse_credentials(data: dict[str, Any], headers: dict[str, list[str]], sensitive_keys: frozenset[str]) -> None:
    """Redact selected sensitive values from JSON payloads in SSE response bodies."""
    body = data.get('body')
    wrapped = isinstance(body, dict)
    if wrapped:
        body = body.get('string')
    if isinstance(body, bytes):  # pragma: no cover - VCR currently provides decoded SSE text
        body = body.decode('utf-8')
    if (
        not sensitive_keys
        or not isinstance(body, str)
        or not any(line.startswith('data:') for line in body.splitlines())
    ):
        return

    lines: list[str] = []
    changed = False
    for line in body.splitlines(keepends=True):
        if line.startswith('data:'):
            payload = line.removeprefix('data:').strip()
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                pass
            else:
                scrubbed = scrub_json_credentials(parsed, sensitive_keys)
                if scrubbed != parsed:
                    ending = line[len(line.rstrip('\r\n')) :]
                    line = f'data: {json.dumps(scrubbed, separators=(",", ":"))}{ending}'
                    changed = True
        lines.append(line)

    if not changed:
        return
    scrubbed_body = ''.join(lines)
    data['body'] = {'string': scrubbed_body} if wrapped else scrubbed_body
    if 'content-length' in headers:
        headers['content-length'] = [str(len(scrubbed_body.encode('utf-8')))]


def scrub_uri_credentials(data: dict[str, Any]) -> None:
    """Redact OAuth callback and authorization query parameters from cassette URIs."""
    uri = data.get('uri')
    if not isinstance(uri, str):
        return
    parsed = urllib.parse.urlsplit(uri)
    if parsed.hostname not in {'127.0.0.1', 'auth.openai.com', 'localhost'}:
        return
    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    sensitive_keys = {'code', 'code_verifier', 'state', 'user_code'}
    if not sensitive_keys.intersection(query):
        return
    for key in sensitive_keys.intersection(query):
        query[key] = ['scrubbed']
    data['uri'] = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(query, doseq=True), parsed.fragment)
    )


def _store_json_body(
    kind: str,
    data: dict[str, Any],
    body: str,
    headers: dict[str, list[str]],
    *,
    openai_oauth: bool,
) -> None:  # pragma: lax no cover
    """Replace an `application/json` body with a normalized, scrubbed `parsed_body`.

    Some endpoints (e.g. resumable file uploads) send a non-JSON body under an `application/json`
    content-type; keep the raw body rather than crashing the serializer.
    """
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        data['body'] = {'string': body} if kind == 'response' else body
        if 'content-length' in headers:
            headers['content-length'] = [str(len(body.encode('utf-8')))]
        return
    normalized = normalize_body(parsed)
    if openai_oauth:
        normalized = scrub_json_credentials(normalized, _OPENAI_OAUTH_JSON_CREDENTIAL_KEYS)
    elif isinstance(normalized, dict):
        # Preserve the serializer's historical top-level access/id token redaction
        # without treating ordinary nested API fields such as `token` or `code` as credentials.
        normalized = {
            key: 'scrubbed' if key in _LEGACY_TOP_LEVEL_CREDENTIAL_KEYS else value for key, value in normalized.items()
        }
    data['parsed_body'] = normalized
    del data['body']
    # Update content-length to match the body that will be produced during deserialize.
    # This is necessary because decompression changes the body size, and botocore
    # verifies content-length against the actual body during cassette replay.
    if 'content-length' in headers:
        new_body = json.dumps(data['parsed_body'])
        headers['content-length'] = [str(len(new_body.encode('utf-8')))]


def serialize(cassette_dict: Any):  # pragma: lax no cover
    for interaction in cassette_dict['interactions']:
        request_uri = interaction.get('request', {}).get('uri', '')
        parsed_request_uri = urllib.parse.urlsplit(request_uri) if isinstance(request_uri, str) else None
        openai_oauth = parsed_request_uri is not None and parsed_request_uri.hostname == 'auth.openai.com'
        codex_responses = (
            parsed_request_uri is not None
            and parsed_request_uri.scheme == 'https'
            and parsed_request_uri.hostname == 'chatgpt.com'
            and parsed_request_uri.path.rstrip('/') == '/backend-api/codex/responses'
        )
        for kind, data in interaction.items():
            headers: dict[str, list[str]] = data.get('headers', {})
            # make headers lowercase
            headers = {k.lower(): v for k, v in headers.items()}
            # filter headers by name
            headers = {k: v for k, v in headers.items() if k not in FILTERED_HEADERS}
            # filter headers by prefix
            headers = {
                k: v
                for k, v in headers.items()
                if not any(k.startswith(prefix) for prefix in FILTERED_HEADER_PREFIXES)
                or k in ALLOWED_HEADERS
                or any(k.startswith(prefix) for prefix in ALLOWED_HEADER_PREFIXES)
            }
            # update headers on source object
            data['headers'] = headers

            content_type = headers.get('content-type', [])
            if any(isinstance(header, str) and header.startswith('application/json') for header in content_type):
                # Parse the body as JSON
                body = data.get('body', None)
                assert body is not None, data
                if isinstance(body, dict):
                    # Responses will have the body under a field called 'string'
                    body = body.get('string')
                if body:
                    if isinstance(body, bytes):
                        content_encoding = headers.get('content-encoding', [])
                        # Decompress the body and remove the content-encoding header.
                        # Otherwise httpx will try to decompress again on cassette replay.
                        if 'br' in content_encoding:
                            body = brotli.decompress(body)
                            headers.pop('content-encoding', None)
                        elif 'gzip' in content_encoding or (len(body) > 2 and body[:2] == b'\x1f\x8b'):
                            try:
                                body = gzip.decompress(body)
                                headers.pop('content-encoding', None)
                            except (gzip.BadGzipFile, zlib.error):
                                pass
                        body = body.decode('utf-8')
                    assert isinstance(body, str), data
                    _store_json_body(kind, data, body, headers, openai_oauth=openai_oauth)
            scrub_form_credentials(data, content_type, openai_oauth=openai_oauth)
            scrub_xml_credentials(data, headers, content_type)
            scrub_sse_credentials(data, headers, _CODEX_SSE_SENSITIVE_KEYS if codex_responses else frozenset())
            scrub_uri_credentials(data)

    # Use our custom dumper
    return yaml.dump(cassette_dict, Dumper=LiteralDumper, allow_unicode=True, width=120)
