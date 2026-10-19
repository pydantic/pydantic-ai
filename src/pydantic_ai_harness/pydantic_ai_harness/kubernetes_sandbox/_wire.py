"""Protobuf wire encoding for the few sandboxd `process.v1` messages the backend exchanges.

Workaround: the SDK's generated stubs (`k8s_agent_sandbox._proto`) need protobuf 7 at runtime, which
`xai-sdk` still excludes, and they load only after the SDK puts its private `_proto` directory on
`sys.path`. Encoding these four messages by hand needs no protobuf at all. Drop this module once the
SDK exposes `Start` itself (https://github.com/kubernetes-sigs/agent-sandbox/issues/1603) and its
stubs work with protobuf 6 (not tracked upstream yet).

Field numbers from `packages/sandboxd/spec/process/v1/process.proto`:

    message ProcessConfig { repeated string command = 1; map<string, string> env_vars = 2; optional string cwd = 3; }
    message StartRequest { ProcessConfig config = 1; optional PTY pty = 2; }
    message StartResponse { oneof event { InitEvent init = 1; bytes stdout = 2; bytes stderr = 3; ExitEvent exit = 4; } }
    message InitEvent { int32 process_id = 1; }
    message ExitEvent { int32 exit_code = 1; }
    message WriteStdinRequest { int32 process_id = 1; oneof payload { bytes input = 2; google.protobuf.Empty eof = 3; } }
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

_VARINT = 0
_I64 = 1
_LEN = 2
_I32 = 5


def _varint(value: int) -> bytes:
    # Negative int32 values are sign-extended to 64 bits, as protobuf does.
    value &= (1 << 64) - 1
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _field(number: int, payload: bytes) -> bytes:
    return _varint(number << 3 | _LEN) + _varint(len(payload)) + payload


def encode_start_request(command: Sequence[str], env: Mapping[str, str]) -> bytes:
    """Encode a `StartRequest` with no `cwd` and no PTY."""
    config = b''.join(_field(1, argument.encode()) for argument in command)
    for key, value in env.items():
        config += _field(2, _field(1, key.encode()) + _field(2, value.encode()))
    return _field(1, config)


def encode_close_stdin(process_id: int) -> bytes:
    """Encode a `WriteStdinRequest` that closes the process's stdin (`eof`)."""
    return _varint(1 << 3 | _VARINT) + _varint(process_id) + _field(3, b'')


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if pos >= len(data) or shift > 63:
            raise ValueError('truncated or oversized varint in a sandboxd message')
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _fields(data: bytes) -> Iterator[tuple[int, int | bytes]]:
    pos = 0
    while pos < len(data):
        key, pos = _read_varint(data, pos)
        number, wire_type = key >> 3, key & 7
        if wire_type == _VARINT:
            value, pos = _read_varint(data, pos)
            yield number, value
        elif wire_type == _LEN:
            length, pos = _read_varint(data, pos)
            if pos + length > len(data):
                raise ValueError('truncated field in a sandboxd message')
            yield number, data[pos : pos + length]
            pos += length
        elif wire_type in (_I64, _I32):
            pos += 8 if wire_type == _I64 else 4  # Unknown fixed-width field from a newer sandboxd.
        else:
            raise ValueError(f'unsupported wire type {wire_type} in a sandboxd message')


def _int32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value & 0x80000000 else value


def _int32_field(message: bytes, number: int) -> int:
    # proto3 omits a zero value, so an empty `ExitEvent` means exit code 0.
    for field_number, value in _fields(message):
        if field_number == number and isinstance(value, int):
            return _int32(value)
    return 0


@dataclass(frozen=True)
class StartEvent:
    """One `StartResponse`: the process id, an output chunk, or the exit code."""

    kind: Literal['init', 'stdout', 'stderr', 'exit', 'unknown']
    data: bytes = b''
    number: int = 0


def decode_start_response(data: bytes) -> StartEvent:
    """Decode a `StartResponse`, ignoring fields a newer sandboxd may add."""
    for number, value in _fields(data):
        if not isinstance(value, bytes):
            continue
        if number == 1:
            return StartEvent('init', number=_int32_field(value, 1))
        if number == 2:
            return StartEvent('stdout', data=value)
        if number == 3:
            return StartEvent('stderr', data=value)
        if number == 4:
            return StartEvent('exit', number=_int32_field(value, 1))
    return StartEvent('unknown')
