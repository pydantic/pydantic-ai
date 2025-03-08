# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
import json
import urllib.parse
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from yaml import Dumper, Loader
else:
    try:
        from yaml import CDumper as Dumper, CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

FILTERED_HEADER_PREFIXES = ['anthropic-', 'cf-', 'x-']
FILTERED_HEADERS = {'authorization', 'date', 'request-id', 'server', 'user-agent', 'via', 'set-cookie'}


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
    cassette_dict = yaml.load(cassette_string, Loader=Loader)
    for interaction in cassette_dict['interactions']:
        for kind, data in interaction.items():
            parsed_body = data.pop('parsed_body', None)
            if parsed_body is not None:
                dumped_body = json.dumps(parsed_body)
                data['body'] = {'string': dumped_body} if kind == 'response' else dumped_body
    return cassette_dict


def serialize(cassette_dict: Any):
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
                body: Any = data.get('body', None)
                assert body is not None, data
                if isinstance(body, dict):
                    # Responses will have the body under a field called 'string'
                    body = body.get('string')
                if body is not None:
                    data['parsed_body'] = json.loads(body)
                    if 'access_token' in data['parsed_body']:
                        data['parsed_body']['access_token'] = 'scrubbed'
                    del data['body']
            if content_type == ['application/x-www-form-urlencoded']:
                query_params = urllib.parse.parse_qs(data['body'])
                if 'client_secret' in query_params:
                    query_params['client_secret'] = ['scrubbed']
                    data['body'] = urllib.parse.urlencode(query_params)

    # Use our custom dumper
    return yaml.dump(cassette_dict, Dumper=LiteralDumper, allow_unicode=True, width=120)
