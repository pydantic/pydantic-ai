"""The recording hooks only run while a cassette is being recorded, so replay never reaches them."""

from __future__ import annotations

import gzip
import json
from typing import cast

import brotli
import pytest
from cassetter import RawRequest, RawResponse, SkipRecording

from ._inline_snapshot import snapshot
from .cassette_hooks import (
    before_record_request,
    before_record_response,
    filter_headers,
    normalize_uri,
    scrub_xml_credentials,
)


def _request(uri: str = 'https://api.example.com/v1', body: bytes | None = None, **headers: str) -> RawRequest:
    return RawRequest(method='POST', uri=uri, headers={k: [v] for k, v in headers.items()}, body=body)


def _response(body: bytes | None = None, **headers: str) -> RawResponse:
    return RawResponse(status=200, headers={k: [v] for k, v in headers.items()}, body=body)


def test_filtered_headers_removed_and_lowercased():
    headers = filter_headers(
        {
            'Content-Type': ['application/json'],
            'Authorization': ['some-token'],
            'Date': ['some-date-string'],
            'X-Test': ['some-token'],
            'cf-ray': ['abc'],
            'anthropic-ratelimit-requests-remaining': ['5'],
            'Other-Header': ['test-value'],
            'x-xet-hash': ['keep'],
            'X-Amzn-Bedrock-Input-Token-Count': ['keep'],
            'x-goog-upload-url': ['keep'],
        }
    )
    assert headers == snapshot(
        {
            'content-type': ['application/json'],
            'other-header': ['test-value'],
            'x-xet-hash': ['keep'],
            'x-amzn-bedrock-input-token-count': ['keep'],
            'x-goog-upload-url': ['keep'],
        }
    )


@pytest.mark.parametrize('url', ['https://oauth2.googleapis.com/token', 'https://auth.openai.com/oauth/token'])
def test_oauth_token_exchanges_are_not_recorded(url: str):
    with pytest.raises(SkipRecording):
        before_record_request(_request(url))


def test_oauth_credentials_are_scrubbed():
    request = before_record_request(
        _request(
            body=b'code=oauth-code&code_verifier=oauth-verifier&refresh_token=oauth-refresh',
            **{
                'Content-Type': 'application/x-www-form-urlencoded',
                'Authorization': 'Bearer oauth-access',
                'ChatGPT-Account-ID': 'oauth-account',
            },
        )
    )
    response = before_record_response(
        _response(
            body=b'{"access_token":"oauth-access","refresh_token":"oauth-refresh","id_token":"oauth-id"}',
            **{'Content-Type': 'application/json'},
        )
    )
    assert b'oauth-' not in (request.body or b'') + (response.body or b'')
    assert (request.headers, request.body) == snapshot(
        (
            {'content-type': ['application/x-www-form-urlencoded']},
            b'code=scrubbed&code_verifier=scrubbed&refresh_token=scrubbed',
        )
    )
    assert (response.headers, response.body) == snapshot(
        (
            {'content-type': ['application/json']},
            b'{"access_token": "scrubbed", "refresh_token": "scrubbed", "id_token": "scrubbed"}',
        )
    )


@pytest.mark.parametrize('content_type', ['application/json', 'text/event-stream', None])
def test_safety_identifier_is_scrubbed(content_type: str | None):
    body = '{"safety_identifier":"synthetic-user-id","output":[]}'
    if content_type != 'application/json':
        body = f'data: {{"response":{body}}}\n\ndata: [DONE]\n\n'
    headers = {'content-type': content_type} if content_type is not None else {}
    response = before_record_response(_response(body=body.encode(), **headers))
    assert response.body is not None
    assert b'synthetic-user-id' not in response.body
    assert b'scrubbed' in response.body


@pytest.mark.parametrize('encoding', ['gzip', 'br'])
def test_compressed_bodies_are_inflated_before_scrubbing(encoding: str):
    raw = b'{"access_token": "secret", "ok": true}'
    compressed = gzip.compress(raw) if encoding == 'gzip' else cast('bytes', brotli.compress(raw))  # pyright: ignore[reportUnknownMemberType]
    response = before_record_response(
        _response(body=compressed, **{'content-type': 'application/json', 'content-encoding': encoding})
    )
    assert 'content-encoding' not in response.headers
    assert response.body is not None
    assert json.loads(response.body) == {'access_token': 'scrubbed', 'ok': True}


def test_undecodable_body_keeps_its_content_encoding():
    response = before_record_response(
        _response(body=b'\x1f\x8bnot really gzip', **{'content-type': 'application/json', 'content-encoding': 'gzip'})
    )
    assert response.headers['content-encoding'] == ['gzip']
    assert response.body == b'\x1f\x8bnot really gzip'


def test_smart_characters_are_normalized():
    response = before_record_response(
        # Escaped so the `fix-smartquotes` pre-commit hook leaves the fixture alone.
        _response(
            body='{"text": "\u201cquoted\u201d \u2014 it\u2019s\u2026"}'.encode(),
            **{'content-type': 'application/json'},
        )
    )
    assert response.body == snapshot(b'{"text": "\\"quoted\\" -- it\'s..."}')


def test_json_array_body_is_normalized_without_scrubbing():
    response = before_record_response(
        _response(body='[{"access_token": "kept"}, "\u2019"]'.encode(), **{'content-type': 'application/json'})
    )
    assert response.body == b'[{"access_token": "kept"}, "\'"]'


def test_non_json_body_under_json_content_type_is_kept():
    response = before_record_response(_response(body=b'\x00\x01 not json', **{'content-type': 'application/json'}))
    assert response.body == b'\x00\x01 not json'


def test_empty_bodies_pass_through():
    assert before_record_request(_request(body=None)).body is None
    assert before_record_response(_response(body=b'')).body == b''
    assert before_record_response(_response(body=b'plain', **{'content-type': 'text/plain'})).body == b'plain'


def test_aws_account_id_is_scrubbed_from_request_uri():
    request = before_record_request(
        _request(
            'https://bedrock-runtime.us-east-1.amazonaws.com/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A111122223333%3Ainference-profile%2Fus.anthropic.claude/converse'
        )
    )
    assert request.uri == snapshot(
        'https://bedrock-runtime.us-east-1.amazonaws.com/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A123456789012%3Ainference-profile%2Fus.anthropic.claude/converse'
    )


def test_normalize_uri_erases_region_project_and_account():
    assert normalize_uri(
        'https://bedrock-runtime.eu-west-2.amazonaws.com/model/arn:aws:bedrock:eu-west-2:111122223333:inference-profile/x/converse'
    ) == snapshot(
        'https://bedrock-runtime.REGION.amazonaws.com/model/arn:aws:bedrock:eu-west-2:123456789012:inference-profile/x/converse'
    )
    assert normalize_uri(
        'https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini:generateContent'
    ) == snapshot(
        'https://aiplatform.googleapis.com/v1/projects/PROJECT/locations/REGION/publishers/google/models/gemini:generateContent'
    )
    assert normalize_uri('https://api.openai.com/v1/responses') == 'https://api.openai.com/v1/responses'


def test_scrub_xml_credentials_redacts_sts_tokens():
    xml_body = (
        b'<AssumeRoleWithWebIdentityResponse>'
        b'<Credentials>'
        b'<AccessKeyId>ASIA1234</AccessKeyId>'
        b'<SecretAccessKey>secret123</SecretAccessKey>'
        b'<SessionToken>token456</SessionToken>'
        b'<Expiration>2026-01-01T00:00:00Z</Expiration>'
        b'</Credentials>'
        b'</AssumeRoleWithWebIdentityResponse>'
    )
    assert scrub_xml_credentials(xml_body) == snapshot(
        b'<AssumeRoleWithWebIdentityResponse><Credentials><AccessKeyId>SCRUBBED</AccessKeyId><SecretAccessKey>SCRUBBED</SecretAccessKey><SessionToken>SCRUBBED</SessionToken><Expiration>2099-01-01T00:00:00Z</Expiration></Credentials></AssumeRoleWithWebIdentityResponse>'
    )


def test_scrub_xml_credentials_only_touches_credentials():
    assert scrub_xml_credentials(b'<Response>ok</Response>') == b'<Response>ok</Response>'
    # Dispatch is on the content type, so JSON carrying a `<Credentials>` tag is left to the JSON scrubber.
    response = before_record_response(
        _response(body=b'{"xml": "<Credentials>secret</Credentials>"}', **{'content-type': 'application/json'})
    )
    assert response.body == b'{"xml": "<Credentials>secret</Credentials>"}'
    response = before_record_response(
        _response(body=b'<Credentials>secret</Credentials>', **{'content-type': 'text/xml'})
    )
    assert response.body == snapshot(b'<Credentials>secret</Credentials>')
