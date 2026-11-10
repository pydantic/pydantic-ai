"""Tests for `SpritesSandbox(mounts=...)`, against the fake Sprites transport with a `sudo` that answers for the helper.

The fake runs commands on this machine, so a `sudo` script first on `PATH` stands in for the Sprite's: it records the
root command and the line it read from stdin, and prints the helper's reply. The helper itself runs only in a real
Sprite: see `test_mounts_live.py`.
"""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from pydantic_ai_harness import MountError
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox
from tests.harness.mounts.sandbox_fixtures import bucket, probing

from .fake_sprites import SpriteTransport

_SUDO = """#!/bin/sh
printf '%s\\n' "$@" > "$0.argv"
head -n 1 > "$0.stdin"
cat "$0.reply"
exit "$(cat "$0.exit")"
"""


@dataclass
class FakeSudo:
    path: Path

    def reply(self, stdout: str = '{"recovered": []}\n', exit_code: int = 0) -> None:
        Path(f'{self.path}.reply').write_text(stdout, encoding='utf-8')
        Path(f'{self.path}.exit').write_text(str(exit_code), encoding='utf-8')

    @property
    def argv(self) -> list[str]:
        return Path(f'{self.path}.argv').read_text(encoding='utf-8').splitlines()

    @property
    def request(self) -> dict[str, Any]:
        return json.loads(Path(f'{self.path}.stdin').read_text(encoding='utf-8'))


@pytest.fixture
def sudo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeSudo:
    """A `sudo` first on the `PATH` the fake's commands inherit, answering as the helper does."""
    bin_dir = tmp_path / 'fake-bin'
    bin_dir.mkdir()
    script = bin_dir / 'sudo'
    script.write_text(_SUDO)
    script.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    fake = FakeSudo(script)
    fake.reply()
    return fake


async def test_a_run_mounts_through_sudo_with_the_keys_only_on_stdin(
    transport: SpriteTransport, sudo: FakeSudo
) -> None:
    result = await probing(SpritesSandbox[str](mounts=[bucket()])).run('go', deps='')

    assert result.output == '{"probe":"contained\\n"}'
    assert sudo.argv[:4] == ['-n', '--', 'sh', '-c'] and '/dev/fuse' in sudo.argv[4]
    assert sudo.argv[5:8] == ['sh', 'python3', '-c']
    assert sudo.request['working_dir'] == str(transport.root.resolve())
    assert sudo.request['mounts'][0]['keys']['secret_access_key'] == 'the-secret-key'
    (root_exec,) = [socket for socket in transport.execs if 'sudo' in socket.query['cmd']]
    assert root_exec.query['dir'] == ['/']
    assert any(b'the-secret-key' in frame for frame in transport.stdin_sent)
    assert not any('the-secret-key' in socket.query_url for socket in transport.execs)


async def test_mounts_can_be_a_function_of_the_run(transport: SpriteTransport, sudo: FakeSudo) -> None:
    await probing(SpritesSandbox[str](mounts=lambda ctx: [bucket(key=ctx.deps)])).run('go', deps='customer-a-key')

    assert sudo.request['mounts'][0]['keys']['access_key_id'] == 'customer-a-key'


async def test_a_sprite_without_fuse_is_refused(transport: SpriteTransport, sudo: FakeSudo) -> None:
    sudo.reply('', 86)

    with pytest.raises(MountError) as raised:
        await probing(SpritesSandbox[str](mounts=[bucket()])).run('go', deps='')

    (name,) = transport.created
    assert str(raised.value) == (f'Sprite {name} has no FUSE device (`/dev/fuse`), which mounts need.')


async def test_the_helpers_refusal_fails_the_operation(transport: SpriteTransport, sudo: FakeSudo) -> None:
    sudo.reply('{"error": "conflict", "message": "data already shows another bucket in this sandbox."}\n')

    with pytest.raises(MountError, match='data already shows another bucket in this sandbox'):
        await probing(SpritesSandbox[str](mounts=[bucket()])).run('go', deps='')


def test_a_spec_cannot_set_mounts() -> None:
    assert 'mounts' not in inspect.signature(SpritesSandbox.from_spec).parameters
