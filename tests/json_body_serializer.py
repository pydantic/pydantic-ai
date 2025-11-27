# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
import json
import urllib.parse
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from yaml import Dumper
else:
    try:
        from yaml import CDumper as Dumper
    except ImportError:  # pragma: no cover
        from yaml import Dumper

FILTERED_HEADER_PREFIXES = ['anthropic-', 'cf-', 'x-']
FILTERED_HEADERS = {'authorization', 'date', 'request-id', 'server', 'user-agent', 'via', 'set-cookie', 'api-key'}


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
                k: v for k, v in headers.items() if not any(k.startswith(prefix) for prefix in FILTERED_HEADER_PREFIXES)
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
                    data['parsed_body'] = json.loads(body)  # pyright: ignore[reportUnknownArgumentType]
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
