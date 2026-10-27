"""Runs the `network=False` seccomp filter in a small classic-BPF interpreter, for both architectures it covers."""

from __future__ import annotations

import base64
import struct

import pytest

from pydantic_ai_harness.bubblewrap_sandbox._seccomp import NETWORK_FILTER_BASE64

ALLOW, DENY, KILL = 'allow', 'deny', 'kill'
_ACTIONS = {0x7FFF0000: ALLOW, 0x00050001: DENY, 0x80000000: KILL}

X86_64, AARCH64, I386 = 0xC000003E, 0xC00000B7, 0x40000003
AF_UNIX, AF_INET, AF_VSOCK = 1, 2, 40
SOCK_STREAM, SOCK_DGRAM, SOCK_SEQPACKET, SOCK_CLOEXEC = 1, 2, 5, 0o2000000


def run_filter(arch: int, number: int, arg0: int = 0, arg1: int = SOCK_STREAM) -> str:
    """What the filter does with a call: `struct seccomp_data` is nr, arch, instruction pointer, then args."""
    data = struct.pack('<iIQ6Q', number, arch, 0, arg0, arg1, 0, 0, 0, 0)
    program = base64.b64decode(NETWORK_FILTER_BASE64)
    accumulator, pc = 0, 0
    while True:
        code, true, false, k = struct.unpack_from('<HBBI', program, pc * 8)
        pc += 1
        if code == 0x20:
            (accumulator,) = struct.unpack_from('<I', data, k)
        elif code == 0x54:
            accumulator &= k
        elif code == 0x06:
            return _ACTIONS[k]
        else:
            taken = accumulator == k if code == 0x15 else accumulator >= k
            assert code in (0x15, 0x35)
            pc += true if taken else false


# (name, x86_64 number, aarch64 number): the calls Codex's restricted network mode denies.
DENIED = [
    ('ptrace', 101, 117),
    ('process_vm_readv', 310, 270),
    ('process_vm_writev', 311, 271),
    ('io_uring_setup', 425, 425),
    ('io_uring_enter', 426, 426),
    ('io_uring_register', 427, 427),
    ('connect', 42, 203),
    ('accept', 43, 202),
    ('accept4', 288, 242),
    ('bind', 49, 200),
    ('listen', 50, 201),
    ('getpeername', 52, 205),
    ('getsockname', 51, 204),
    ('shutdown', 48, 210),
    ('sendto', 44, 206),
    ('sendmmsg', 307, 269),
    ('recvmmsg', 299, 243),
    ('getsockopt', 55, 209),
    ('setsockopt', 54, 208),
]
ALLOWED = [('read', 0, 63), ('write', 1, 64), ('execve', 59, 221), ('recvfrom', 45, 207), ('sendmsg', 46, 211)]


@pytest.mark.parametrize(('name', 'x86_64', 'aarch64'), DENIED)
def test_network_and_process_calls_are_denied(name: str, x86_64: int, aarch64: int) -> None:
    assert (run_filter(X86_64, x86_64), run_filter(AARCH64, aarch64)) == (DENY, DENY)


@pytest.mark.parametrize(('name', 'x86_64', 'aarch64'), ALLOWED)
def test_other_calls_are_allowed(name: str, x86_64: int, aarch64: int) -> None:
    assert (run_filter(X86_64, x86_64), run_filter(AARCH64, aarch64)) == (ALLOW, ALLOW)


@pytest.mark.parametrize(('arch', 'socket', 'socketpair'), [(X86_64, 41, 53), (AARCH64, 198, 199)])
def test_only_unix_sockets_can_be_created(arch: int, socket: int, socketpair: int) -> None:
    assert [run_filter(arch, call, AF_UNIX) for call in (socket, socketpair)] == [ALLOW, ALLOW]
    for family in (AF_INET, AF_VSOCK):
        assert [run_filter(arch, call, family) for call in (socket, socketpair)] == [DENY, DENY]
    # Only the domain's low 32 bits count, as the kernel reads an `int`.
    assert run_filter(arch, socket, AF_UNIX | 1 << 32) == ALLOW


@pytest.mark.parametrize(('arch', 'socket', 'socketpair'), [(X86_64, 41, 53), (AARCH64, 198, 199)])
def test_unix_datagram_sockets_are_denied(arch: int, socket: int, socketpair: int) -> None:
    """`sendmsg` can address a datagram to any path without `connect`; stream and seqpacket sockets can't."""
    for kind in (SOCK_DGRAM, SOCK_DGRAM | SOCK_CLOEXEC):
        assert [run_filter(arch, call, AF_UNIX, kind) for call in (socket, socketpair)] == [DENY, DENY]
    for kind in (SOCK_STREAM | SOCK_CLOEXEC, SOCK_SEQPACKET):
        assert [run_filter(arch, call, AF_UNIX, kind) for call in (socket, socketpair)] == [ALLOW, ALLOW]


def test_other_abis_are_killed_rather_than_let_through() -> None:
    assert run_filter(I386, 102, AF_INET) == KILL  # socketcall
    assert run_filter(X86_64, 0x40000000 | 42) == KILL  # x32 connect
