from typing import Any

import pytest
import yaml

from .json_body_serializer import (
    deserialize,
    looks_like_base64,
    restore_truncated_binary,
    scrub_sensitive_fields,
    serialize,
    truncate_binary_data,
)


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


class TestScrubSensitiveFields:
    def test_scrubs_access_token(self):
        data = {'access_token': 'secret123', 'other': 'value'}
        result = scrub_sensitive_fields(data)
        assert result == {'access_token': 'scrubbed', 'other': 'value'}

    def test_scrubs_id_token(self):
        data = {'id_token': 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature', 'scope': 'openid'}
        result = scrub_sensitive_fields(data)
        assert result == {'id_token': 'scrubbed', 'scope': 'openid'}

    def test_scrubs_nested_fields(self):
        data = {'outer': {'access_token': 'nested_secret'}, 'list': [{'id_token': 'in_list'}]}
        result = scrub_sensitive_fields(data)
        assert result == {'outer': {'access_token': 'scrubbed'}, 'list': [{'id_token': 'scrubbed'}]}

    def test_preserves_non_sensitive_fields(self):
        data = {'model': 'gpt-4', 'messages': [{'role': 'user', 'content': 'hello'}]}
        result = scrub_sensitive_fields(data)
        assert result == data


class TestLooksLikeBase64:
    def test_short_string_returns_false(self):
        assert looks_like_base64('abc') is False
        assert looks_like_base64('a' * 99) is False

    def test_base64_image_returns_true(self):
        base64_sample = (
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
        )
        assert looks_like_base64(base64_sample * 10) is True

    def test_regular_text_returns_false(self):
        text = 'This is a normal text message that happens to be long enough. ' * 5
        assert looks_like_base64(text) is False


class TestTruncateBinaryData:
    def test_short_string_unchanged(self):
        data = {'image': 'short_base64'}
        result = truncate_binary_data(data)
        assert result == data

    def test_long_base64_truncated(self):
        long_base64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA' * 50
        data = {'image': long_base64}
        result = truncate_binary_data(data)
        assert result['image'].startswith('__BINARY_TRUNCATED__:len=')
        assert 'sha256=' in result['image']

    def test_nested_truncation(self):
        long_base64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/' * 50
        data = {'outer': {'data': long_base64}, 'list': [long_base64]}
        result = truncate_binary_data(data)
        assert result['outer']['data'].startswith('__BINARY_TRUNCATED__')
        assert result['list'][0].startswith('__BINARY_TRUNCATED__')

    def test_preserves_non_base64(self):
        data = {'text': 'Hello world! ' * 200, 'number': 42}
        result = truncate_binary_data(data)
        assert result == data


class TestRestoreTruncatedBinary:
    def test_restores_placeholder_to_valid_base64(self):
        placeholder = '__BINARY_TRUNCATED__:len=2550:sha256=abc123def456'
        result = restore_truncated_binary(placeholder)
        # Should be valid base64
        import base64

        decoded = base64.b64decode(result)
        assert decoded == placeholder.encode()

    def test_restores_nested_placeholders(self):
        data = {
            'outer': {'image': '__BINARY_TRUNCATED__:len=1000:sha256=abc'},
            'list': ['__BINARY_TRUNCATED__:len=2000:sha256=def'],
            'normal': 'keep this',
        }
        result = restore_truncated_binary(data)
        import base64

        assert base64.b64decode(result['outer']['image']) == b'__BINARY_TRUNCATED__:len=1000:sha256=abc'
        assert base64.b64decode(result['list'][0]) == b'__BINARY_TRUNCATED__:len=2000:sha256=def'
        assert result['normal'] == 'keep this'

    def test_leaves_non_placeholders_unchanged(self):
        data = {'text': 'normal text', 'number': 42, 'nested': {'key': 'value'}}
        result = restore_truncated_binary(data)
        assert result == data


class TestOpenAIHeaderFiltering:
    def test_openai_org_headers_filtered(self):
        cassette = {
            'interactions': [
                {
                    'response': {
                        'headers': {
                            'Content-Type': ['application/json'],
                            'openai-organization': ['pydantic-28gund'],
                            'openai-project': ['proj_wlzE3wrTAwGKSsoZUKNhfDgz'],
                        },
                        'body': {'string': '{"result": "ok"}'},
                    }
                }
            ]
        }
        output = serialize(cassette)
        assert 'openai-organization' not in output
        assert 'openai-project' not in output
        assert 'pydantic-28gund' not in output
