import json
from typing import Any

import pytest
import yaml

from ._inline_snapshot import snapshot
from .json_body_serializer import deserialize, scrub_xml_credentials, serialize


@pytest.fixture
def cassette_dict_base():
    """
    A fixture providing a base cassette dictionary with:
    - A request containing a multi-line JSON string in `body`.
    - A response containing a multi-line JSON string in `body['string']`.
    - Various headers to test filtering & lowercasing.
    """
    return {
        'interactions': [
            {
                'request': {
                    'headers': {
                        'Content-Type': ['application/json'],
                        'Authorization': ['some-token'],  # Should be filtered out
                        'X-Test': ['some-token'],  # Should be filtered out
                        'Other-Header': ['test-value'],  # Should remain, but become lowercase (other-header)
                    },
                    'body': '{"message": "line1\\nline2"}',  # multi-line JSON
                },
            },
            {
                'response': {
                    'headers': {
                        'Content-Type': ['application/json'],
                        'Date': ['some-date-string'],  # Should be filtered out
                    },
                    'body': {'string': '{"response": "line3\\nline4"}'},
                },
            },
        ]
    }


def test_filtered_headers_removed(cassette_dict_base: dict[str, Any]):
    """
    Ensure that headers in FILTERED_HEADERS (e.g. 'Authorization', 'Date', etc.)
    are removed from the serialized output.
    """
    output = serialize(cassette_dict_base).lower()

    assert 'authorization:' not in output, "Expected 'Authorization' to be filtered out."
    assert 'date:' not in output, "Expected 'Date' to be filtered out."
    assert 'x-test:' not in output, "Expected 'X-Test' to be filtered out."


def test_headers_are_lowercased(cassette_dict_base: dict[str, Any]):
    """
    Ensure that the remaining headers are written in all-lowercase form.
    """
    output = serialize(cassette_dict_base)

    # 'Other-Header' should appear as 'other-header:' in the YAML
    assert 'other-header:' in output.lower(), "Expected 'Other-Header' to become 'other-header' in the serialized YAML."
    # Ensure we don't see the uppercase form
    assert 'Other-Header:' not in output, (
        f'Found uppercase header name in the serialized YAML; expected it to be lowercase.\nOutput:\n{output}'
    )


def test_multiline_strings_are_literal(cassette_dict_base: dict[str, Any]):
    """
    Ensure that multi-line JSON strings are emitted as literal blocks in YAML.
    """
    output = serialize(cassette_dict_base)

    # The custom LiteralDumper uses style='|' when it finds '\n' in a string.
    # Often you'll see '|\n' or '|-' in the YAML depending on your exact presenter.
    assert '|\n' in output or '|-' in output, (
        "Expected multi-line string to be represented in literal style ('|' or '|-'), "
        "but didn't find it.\n\nSerialized output:\n" + output
    )


def test_serialization_includes_parsed_body_excludes_body(cassette_dict_base: dict[str, Any]):
    """
    After calling `serialize(...)`, the in-memory dict is transformed such that
    JSON content is moved into `parsed_body` (and `body` is removed) for readability.
    We then confirm the *YAML text* also contains `parsed_body`.
    """
    output_yaml = serialize(cassette_dict_base)

    # 1) Check the YAML text for "parsed_body" key
    assert 'parsed_body:' in output_yaml, (
        f"Expected the YAML output to contain 'parsed_body' for JSON content.\nGot:\n{output_yaml}"
    )

    # 2) Double-check by parsing the YAML text back into a dict (just to confirm structure)
    parsed = yaml.safe_load(output_yaml)
    interactions = parsed.get('interactions', [])
    assert len(interactions) == 2, 'Expected two interactions in the serialized output.'

    # request
    req = interactions[0]['request']
    assert 'parsed_body' in req, "Expected 'parsed_body' to be present for the request in the YAML dictionary."
    assert 'body' not in req, "Expected 'body' to be removed from the request after serialization."
    # response
    resp = interactions[1]['response']
    assert 'parsed_body' in resp, "Expected 'parsed_body' to be present for the response in the YAML dictionary."
    assert 'body' not in resp, "Expected 'body' to be removed from the response after serialization."


def test_deserialization_restores_body_removes_parsed_body(cassette_dict_base: dict[str, Any]):
    """
    After we deserialize the YAML, we want `body` to be restored for VCR usage,
    and `parsed_body` should be removed.
    """
    # First, get the YAML
    output_yaml = serialize(cassette_dict_base)

    # Now, deserialize it
    deserialized = deserialize(output_yaml)
    interactions = deserialized.get('interactions', [])

    assert len(interactions) == 2, 'Expected exactly two interactions after deserialization.'

    # Check request
    req = interactions[0]['request']
    assert 'parsed_body' not in req, "Expected 'parsed_body' to NOT be present in the request after deserialization."
    assert 'body' in req, "Expected 'body' to be restored in the request after deserialization."
    # The original code for a request sets `body` to a string
    assert req['body'] == '{"message": "line1\\nline2"}', (
        'Expected the request body to contain the JSON string from parsed_body.'
    )

    # Check response
    resp = interactions[1]['response']
    assert 'parsed_body' not in resp, "Expected 'parsed_body' to NOT be present in the response after deserialization."
    assert 'body' in resp, "Expected 'body' to be restored in the response after deserialization."
    # The original code for a response sets `body` to a dict with {'string': ...}
    assert resp['body'] == {'string': '{"response": "line3\\nline4"}'}, (
        "Expected the response body to be a dict with 'string' after deserialization."
    )


def test_round_trip_data_integrity(cassette_dict_base: dict[str, Any]):
    """
    Checks that going from an original cassette_dict -> serialize -> deserialize
    yields the data structure that VCR needs (i.e., final form has 'body',
    not 'parsed_body'), while still having the original JSON strings.
    """
    # Original dictionary:
    original_request_body = cassette_dict_base['interactions'][0]['request']['body']
    original_response_body = cassette_dict_base['interactions'][1]['response']['body']['string']

    # Serialize -> Deserialize
    output_yaml = serialize(cassette_dict_base)
    deserialized = deserialize(output_yaml)

    # Final data structure should have 'body' again
    interactions = deserialized.get('interactions', [])
    assert len(interactions) == 2

    req: Any = interactions[0]['request']
    resp: Any = interactions[1]['response']

    # Ensure the final request body matches the original
    assert req['body'] == original_request_body, (
        'The final request body after round-trip should match the original JSON string.'
    )

    # Ensure the final response body['string'] matches the original
    assert resp['body'] == {'string': original_response_body}, (
        'The final response body after round-trip should match the original JSON string.'
    )

    # No 'parsed_body' in the final dictionary
    assert 'parsed_body' not in req
    assert 'parsed_body' not in resp


def test_scrub_xml_credentials_redacts_sts_tokens():
    """Test that scrub_xml_credentials redacts AWS STS credentials from XML bodies."""
    xml_body = (
        '<AssumeRoleWithWebIdentityResponse>'
        '<Credentials>'
        '<AccessKeyId>ASIA1234</AccessKeyId>'
        '<SecretAccessKey>secret123</SecretAccessKey>'
        '<SessionToken>token456</SessionToken>'
        '<Expiration>2026-01-01T00:00:00Z</Expiration>'
        '</Credentials>'
        '</AssumeRoleWithWebIdentityResponse>'
    )
    data: dict[str, Any] = {'body': {'string': xml_body}}
    headers: dict[str, list[str]] = {'content-type': ['text/xml'], 'content-length': ['999']}

    scrub_xml_credentials(data, headers, ['text/xml'])

    body = data['body']['string']
    assert body == snapshot(
        '<AssumeRoleWithWebIdentityResponse><Credentials><AccessKeyId>SCRUBBED</AccessKeyId><SecretAccessKey>SCRUBBED</SecretAccessKey><SessionToken>SCRUBBED</SessionToken><Expiration>2099-01-01T00:00:00Z</Expiration></Credentials></AssumeRoleWithWebIdentityResponse>'
    )
    assert headers['content-length'] == [str(len(body.encode('utf-8')))]


def test_scrub_xml_credentials_string_body_no_content_length():
    """Test scrub_xml_credentials with a plain string body and no content-length header."""
    xml_body = '<Credentials><AccessKeyId>ASIA1234</AccessKeyId></Credentials>'
    data: dict[str, Any] = {'body': xml_body}
    headers: dict[str, list[str]] = {'content-type': ['text/xml']}

    scrub_xml_credentials(data, headers, ['text/xml'])

    assert data['body'] == snapshot({'string': '<Credentials><AccessKeyId>SCRUBBED</AccessKeyId></Credentials>'})
    assert 'content-length' not in headers


def test_scrub_xml_credentials_skips_non_xml():
    """Test that scrub_xml_credentials is a no-op for non-XML content types."""
    data: dict[str, Any] = {'body': {'string': '<Credentials>secret</Credentials>'}}
    headers: dict[str, list[str]] = {'content-type': ['application/json']}

    scrub_xml_credentials(data, headers, ['application/json'])

    assert data['body']['string'] == '<Credentials>secret</Credentials>'


def test_scrub_xml_credentials_skips_non_credentials_xml():
    """Test that scrub_xml_credentials is a no-op for XML without <Credentials>."""
    data: dict[str, Any] = {'body': {'string': '<Response>ok</Response>'}}
    headers: dict[str, list[str]] = {'content-type': ['text/xml']}

    scrub_xml_credentials(data, headers, ['text/xml'])

    assert data['body']['string'] == '<Response>ok</Response>'


def test_codex_oauth_credentials_are_recursively_scrubbed() -> None:
    secrets = {
        'access_token': 'access-secret',
        'refresh_token': 'refresh-secret',
        'token': 'revocation-secret',
        'id_token': 'id-secret',
        'account_id': 'account-secret',
        'authorization_code': 'authorization-secret',
        'code_verifier': 'verifier-secret',
        'device_auth_id': 'device-secret',
        'user_code': 'user-secret',
    }
    cassette: dict[str, Any] = {
        'interactions': [
            {
                'request': {
                    'uri': 'https://auth.openai.com/callback?code=callback-secret&state=state-secret',
                    'headers': {
                        'Content-Type': ['application/json'],
                        'ChatGPT-Account-ID': ['header-account-secret'],
                    },
                    'body': json.dumps({'credentials': secrets}),
                },
                'response': {
                    'headers': {'Content-Type': ['application/json']},
                    'body': {'string': json.dumps({'nested': [{'refresh_token': 'response-secret'}]})},
                },
            }
        ]
    }

    output = serialize(cassette)

    for secret in [*secrets.values(), 'callback-secret', 'state-secret', 'header-account-secret', 'response-secret']:
        assert secret not in output
    assert output.count('scrubbed') >= len(secrets) + 3


def test_codex_sse_identifiers_are_recursively_scrubbed() -> None:
    cassette: dict[str, Any] = {
        'interactions': [
            {
                'request': {
                    'uri': 'https://chatgpt.com/backend-api/codex/responses',
                    'headers': {},
                },
                'response': {
                    'headers': {'Content-Type': ['text/event-stream'], 'Content-Length': ['999']},
                    'body': {
                        'string': (
                            'event: response.created\n'
                            'data: {"response":{"safety_identifier":"account-secret",'
                            '"prompt_cache_key":"cache-secret","output":[]}}\n\n'
                            'event: response.output_text.delta\n'
                            'data: {"delta":"safe output"}\n\n'
                            'data: [DONE]\n\n'
                        )
                    },
                },
            }
        ]
    }

    output = serialize(cassette)
    deserialized = deserialize(output)
    body = deserialized['interactions'][0]['response']['body']['string']

    assert 'account-secret' not in output
    assert 'cache-secret' not in output
    assert body.count('"scrubbed"') == 2
    assert 'data: {"delta":"safe output"}' in body
    assert 'data: [DONE]' in body
    assert deserialized['interactions'][0]['response']['headers']['content-length'] == [str(len(body.encode('utf-8')))]


def test_codex_sse_scrubbing_handles_noops_and_missing_content_length() -> None:
    """Exercise serializer-only edge cases that VCR replay does not run."""
    cassette: dict[str, Any] = {
        'interactions': [
            {
                'request': {'uri': 'https://chatgpt.com/backend-api/codex/responses', 'headers': {}},
                'response': {
                    'headers': {'Content-Type': ['text/event-stream']},
                    'body': {'string': 'data: {"safety_identifier":"account-secret"}\n\n'},
                },
            },
            {
                'request': {'uri': 'https://chatgpt.com/backend-api/codex/responses', 'headers': {}},
                'response': {
                    'headers': {'Content-Type': ['text/event-stream']},
                    'body': {'string': 'data: {"output":[]}\n\n'},
                },
            },
            {
                'request': {'uri': 'https://auth.openai.com/oauth/token?client_id=public-client', 'headers': {}},
            },
        ]
    }

    deserialized = deserialize(serialize(cassette))

    scrubbed_response = deserialized['interactions'][0]['response']
    assert scrubbed_response['body']['string'] == 'data: {"safety_identifier":"scrubbed"}\n\n'
    assert 'content-length' not in scrubbed_response['headers']
    assert deserialized['interactions'][1]['response']['body']['string'] == 'data: {"output":[]}\n\n'
    assert deserialized['interactions'][2]['request']['uri'].endswith('client_id=public-client')


def test_ordinary_json_and_sse_fields_are_preserved() -> None:
    ordinary_fields = {'token': 'The', 'code': 'bad_request', 'account_id': 'public-account'}
    cassette: dict[str, Any] = {
        'interactions': [
            {
                'request': {
                    'uri': 'https://api.openai.com/v1/responses',
                    'headers': {'Content-Type': ['application/json']},
                    'body': json.dumps({'nested': ordinary_fields}),
                },
                'response': {
                    'headers': {'Content-Type': ['text/event-stream']},
                    'body': {'string': f'data: {json.dumps({"logprob": ordinary_fields})}\n\n'},
                },
            }
        ]
    }

    deserialized = deserialize(serialize(cassette))
    interaction = deserialized['interactions'][0]

    assert json.loads(interaction['request']['body'])['nested'] == ordinary_fields
    sse_payload = interaction['response']['body']['string'].removeprefix('data: ').strip()
    assert json.loads(sse_payload)['logprob'] == ordinary_fields
