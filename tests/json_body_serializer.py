# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
import json
import unicodedata
import urllib.parse
from typing import TYPE_CHECKING, Any

import yaml

# Smart quote and special character normalization
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
    from yaml import Dumper
else:
    try:
        from yaml import CDumper as Dumper
    except ImportError:  # pragma: no cover
        from yaml import Dumper

FILTERED_HEADER_PREFIXES = ['anthropic-', 'cf-', 'x-']
FILTERED_HEADERS = {'authorization', 'date', 'request-id', 'server', 'user-agent', 'via', 'set-cookie', 'api-key'}
ALLOWED_HEADERS = {'x-goog-upload-url', 'x-goog-upload-status'}  # required for test_google_model_file_search_tool


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
    cassette_dict = yaml.safe_load(cassette_string)
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
                if not any(k.startswith(prefix) for prefix in FILTERED_HEADER_PREFIXES) or k in ALLOWED_HEADERS
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
                    # NOTE(Marcelo): This doesn't handle gzip compression.
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
