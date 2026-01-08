# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
import gzip
import json
import unicodedata
import urllib.parse
import zlib
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
    elif isinstance(obj, list):  # pragma: no cover
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
FILTERED_HEADERS = {'authorization', 'date', 'request-id', 'server', 'user-agent', 'via', 'set-cookie', 'api-key'}
ALLOWED_HEADER_PREFIXES = {
    # required by huggingface_hub.file_download used by test_embeddings.py::TestSentenceTransformers
    'x-xet-',
}
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


def serialize(cassette_dict: Any):  # pragma: lax no cover
    for interaction in cassette_dict['interactions']:
        for _kind, data in interaction.items():
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
                    parsed = json.loads(body)  # pyright: ignore[reportUnknownArgumentType]
                    # Normalize smart quotes and special characters
                    data['parsed_body'] = normalize_body(parsed)
                    if 'access_token' in data['parsed_body']:
                        data['parsed_body']['access_token'] = 'scrubbed'
                    del data['body']
            if content_type == ['application/x-www-form-urlencoded']:
                query_params = urllib.parse.parse_qs(data['body'])
                for key in ['client_id', 'client_secret', 'refresh_token']:  # pragma: no cover
                    if key in query_params:
                        query_params[key] = ['scrubbed']
                        data['body'] = urllib.parse.urlencode(query_params)

    # Use our custom dumper
    return yaml.dump(cassette_dict, Dumper=LiteralDumper, allow_unicode=True, width=120)
