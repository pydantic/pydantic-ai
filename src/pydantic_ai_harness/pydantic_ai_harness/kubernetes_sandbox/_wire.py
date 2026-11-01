"""Protobuf wire encoding for the few sandboxd `process.v1` messages the backend exchanges.

Workaround: the SDK's generated stubs (`k8s_agent_sandbox._proto`) need protobuf 7 at runtime, which
`xai-sdk` still excludes, and they load only after the SDK puts its private `_proto` directory on
`sys.path`. Encoding these few messages by hand needs no protobuf at all. Drop this module once the
SDK exposes `Start` itself (https://github.com/kubernetes-sigs/agent-sandbox/issues/1603) and its
stubs work with protobuf 6 (not tracked upstream yet).

Field numbers from `packages/sandboxd/spec/process/v1/process.proto`:

    message ProcessConfig { repeated string command = 1; map<string, string> env_vars = 2; optional string cwd = 3; }
    message StartRequest { ProcessConfig config = 1; optional PTY pty = 2; }
    message StartResponse { oneof event { InitEvent init = 1; bytes stdout = 2; bytes stderr = 3; ExitEvent exit = 4; } }
    message ExitEvent { int32 exit_code = 1; }

These messages use only varint and length-delimited fields, so those are the only wire types read.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

_VARINT = 0
_LEN = 2

_EVENTS: dict[int, Literal['stdout', 'stderr', 'exit']] = {2: 'stdout', 3: 'stderr', 4: 'exit'}


def _varint(value: int) -> bytes:
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


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
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
            yield number, data[pos : pos + length]
            pos += length
        else:  # pragma: no cover - not sent by sandboxd's `process.v1` (see above)
            raise ValueError(f'unsupported wire type {wire_type} in a sandboxd message')


def _int32_field(message: bytes, number: int) -> int:
    # proto3 omits a zero value, so an empty `ExitEvent` means exit code 0. A negative int32 is
    # sent sign-extended to 64 bits.
    value = next((value for n, value in _fields(message) if n == number and isinstance(value, int)), 0)
    return value - (1 << 64) if value >= 1 << 63 else value


@dataclass(frozen=True)
class StartEvent:
    """One `StartResponse`: an output chunk, the exit code, or another event (`init` included), which the backend ignores."""

    kind: Literal['stdout', 'stderr', 'exit', 'unknown']
    data: bytes = b''
    number: int = 0


def decode_start_response(data: bytes) -> StartEvent:
    """Decode a `StartResponse`; `init` and any event a newer sandboxd adds decode as `unknown`."""
    for number, value in _fields(data):
        kind = _EVENTS.get(number)
        if kind is not None and isinstance(value, bytes):
            if kind == 'exit':
                return StartEvent(kind, number=_int32_field(value, 1))
            return StartEvent(kind, data=value)
    return StartEvent('unknown')
