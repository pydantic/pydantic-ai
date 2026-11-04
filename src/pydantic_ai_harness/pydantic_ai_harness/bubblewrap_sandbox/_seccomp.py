"""The seccomp filter `BubblewrapWorkspace` loads with `network=False`, as a classic BPF program for `bwrap --seccomp`.

It denies what the Codex CLI's Linux sandbox denies without network access
(`codex-rs/linux-sandbox/src/landlock.rs`, `NetworkSeccompMode::Restricted`): every socket operation that could
reach another process, including over Unix sockets, which a private network namespace doesn't cover. Unix
`socketpair()` stays allowed, since runtimes use it between their own processes. Unlike Codex, Unix datagram
sockets are denied too: `sendmsg` (which runtimes need to pass descriptors) can address one to any path without
`connect`. Denied calls fail with `EPERM`.

The program is built for the host that runs `bwrap`, which may not be this one, so it covers x86_64 and aarch64 and
kills a process calling in with any other ABI (32-bit, or x32 on x86_64) rather than let it skip the rules.
"""

from __future__ import annotations

import base64
import struct

__all__ = ('NETWORK_FILTER_BASE64',)

# Linux syscall numbers, from `arch/x86/entry/syscalls/syscall_64.tbl` and, for aarch64, `scripts/syscall.tbl`.
_DENIED = {
    'x86_64': (
        101,  # ptrace
        310,  # process_vm_readv
        311,  # process_vm_writev
        425,  # io_uring_setup: it can create sockets without `socket()`
        426,  # io_uring_enter
        427,  # io_uring_register
        42,  # connect
        43,  # accept
        288,  # accept4
        49,  # bind
        50,  # listen
        52,  # getpeername
        51,  # getsockname
        48,  # shutdown
        44,  # sendto
        307,  # sendmmsg
        299,  # recvmmsg
        55,  # getsockopt
        54,  # setsockopt
    ),
    'aarch64': (117, 270, 271, 425, 426, 427, 203, 202, 242, 200, 201, 205, 204, 210, 206, 269, 243, 209, 208),
}
_SOCKET = {'x86_64': (41, 53), 'aarch64': (198, 199)}
"""`socket` and `socketpair`: denied unless the domain is `AF_UNIX`, as in Codex, and the type isn't `SOCK_DGRAM`."""

_AUDIT_ARCH = {'x86_64': 0xC000003E, 'aarch64': 0xC00000B7}
_X32_SYSCALL_BIT = 0x40000000
_AF_UNIX = 1
_SOCK_DGRAM = 2
_SOCK_TYPE_MASK = 0xF
"""The type argument also carries flags such as `SOCK_CLOEXEC`."""

# `struct seccomp_data`: int nr, u32 arch, u64 instruction_pointer, u64 args[6] (little-endian on both arches).
_NR, _ARCH, _ARG0_LOW, _ARG1_LOW = 0, 4, 16, 24

_LOAD = 0x20  # BPF_LD | BPF_W | BPF_ABS
_JEQ = 0x15  # BPF_JMP | BPF_JEQ | BPF_K
_JGE = 0x35  # BPF_JMP | BPF_JGE | BPF_K
_AND = 0x54  # BPF_ALU | BPF_AND | BPF_K
_RET = 0x06  # BPF_RET | BPF_K
_ALLOW = 0x7FFF0000
_DENY = 0x00050000 | 1  # SECCOMP_RET_ERRNO | EPERM
_KILL = 0x80000000  # SECCOMP_RET_KILL_PROCESS

_Instruction = tuple[int, str | int, str | int, int]
"""`(code, jump if true, jump if false, k)`; a jump is a label, or 0 for the next instruction."""


def _program() -> list[_Instruction | str]:
    """The program with symbolic jump targets; a bare string marks where a label points."""
    program: list[_Instruction | str] = [(_LOAD, 0, 0, _ARCH)]
    program += [(_JEQ, f'{arch}', 0, audit) for arch, audit in _AUDIT_ARCH.items()]
    program.append((_RET, 0, 0, _KILL))
    for arch, denied in _DENIED.items():
        program += [arch, (_LOAD, 0, 0, _NR)]
        if arch == 'x86_64':
            program.append((_JGE, 'kill', 0, _X32_SYSCALL_BIT))
        program += [(_JEQ, 'deny', 0, number) for number in denied]
        program += [(_JEQ, 'socket', 0, number) for number in _SOCKET[arch]]
        program.append((_RET, 0, 0, _ALLOW))
    program += [
        'socket',
        (_LOAD, 0, 0, _ARG0_LOW),
        (_JEQ, 0, 'deny', _AF_UNIX),
        (_LOAD, 0, 0, _ARG1_LOW),
        (_AND, 0, 0, _SOCK_TYPE_MASK),
        (_JEQ, 'deny', 'allow', _SOCK_DGRAM),
        'allow',
        (_RET, 0, 0, _ALLOW),
        'deny',
        (_RET, 0, 0, _DENY),
        'kill',
        (_RET, 0, 0, _KILL),
    ]
    return program


def _assemble(program: list[_Instruction | str]) -> bytes:
    labels: dict[str, int] = {}
    instructions: list[_Instruction] = []
    for item in program:
        if isinstance(item, str):
            labels[item] = len(instructions)
        else:
            instructions.append(item)

    def offset(target: str | int, index: int) -> int:
        # Classic BPF jumps only forward, relative to the next instruction.
        return target if isinstance(target, int) else labels[target] - index - 1

    return b''.join(
        struct.pack('<HBBI', code, offset(true, index), offset(false, index), k)
        for index, (code, true, false, k) in enumerate(instructions)
    )


NETWORK_FILTER_BASE64 = base64.b64encode(_assemble(_program())).decode()
"""The filter, base64-encoded so it can travel in a command line to the host that runs `bwrap`."""
