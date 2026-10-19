"""The hand-written sandboxd messages, checked against protobuf's own encoding of the same schema."""

from __future__ import annotations

import pytest

from pydantic_ai_harness.kubernetes_sandbox._wire import (
    StartEvent,
    decode_start_response,
    encode_close_stdin,
    encode_start_request,
)

from .fake_sandboxd import MESSAGES


def test_start_request_decodes_as_protobuf() -> None:
    data = encode_start_request(['python', '-c', 'print("é")'], {'A': '1', 'EMPTY': ''})
    request = MESSAGES['StartRequest'].FromString(data)
    assert list(request.config.command) == ['python', '-c', 'print("é")']
    assert dict(request.config.env_vars) == {'A': '1', 'EMPTY': ''}
    assert request.config.cwd == ''


def test_close_stdin_decodes_as_protobuf() -> None:
    request = MESSAGES['WriteStdinRequest'].FromString(encode_close_stdin(300))
    assert (request.process_id, request.WhichOneof('payload')) == (300, 'eof')


@pytest.mark.parametrize(
    ('fields', 'expected'),
    [
        ({'init': {'process_id': 7}}, StartEvent('init', number=7)),
        ({'stdout': b'out'}, StartEvent('stdout', data=b'out')),
        ({'stderr': b''}, StartEvent('stderr', data=b'')),
        ({'exit': {'exit_code': 137}}, StartEvent('exit', number=137)),
        # proto3 omits zero values: an empty `ExitEvent` is exit code 0.
        ({'exit': {}}, StartEvent('exit', number=0)),
        ({'exit': {'exit_code': -1}}, StartEvent('exit', number=-1)),
        ({}, StartEvent('unknown')),
    ],
)
def test_start_response_decodes_protobuf(fields: dict[str, object], expected: StartEvent) -> None:
    assert decode_start_response(MESSAGES['StartResponse'](**fields).SerializeToString()) == expected


def test_fields_a_newer_sandboxd_adds_are_skipped() -> None:
    stdout = MESSAGES['StartResponse'](stdout=b'out').SerializeToString()
    # Field 9 as varint, fixed64 and fixed32, then field 10 as a message, before the known event.
    unknown = bytes([9 << 3 | 0, 1, 9 << 3 | 1, *b'\0' * 8, 9 << 3 | 5, *b'\0' * 4, 10 << 3 | 2, 0])
    assert decode_start_response(unknown + stdout) == StartEvent('stdout', data=b'out')
    # An `ExitEvent` whose exit code arrives with an unexpected wire type reads as 0.
    assert decode_start_response(bytes([4 << 3 | 2, 3, 1 << 3 | 2, 1, 0])) == StartEvent('exit', number=0)


@pytest.mark.parametrize(
    'data',
    [
        bytes([0x80]),  # a varint that never ends
        bytes([0x80] * 11),  # a varint longer than 64 bits
        bytes([2 << 3 | 2, 5, 1]),  # a field longer than the message
        bytes([2 << 3 | 3]),  # a group, which proto3 never sends
    ],
)
def test_malformed_messages_raise(data: bytes) -> None:
    with pytest.raises(ValueError):
        decode_start_response(data)
