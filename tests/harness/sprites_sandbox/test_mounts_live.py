"""`SpritesSandbox(mounts=...)` against a real Sprite, with MinIO running inside it.

Opt in with `PYDANTIC_AI_HARNESS_SPRITES_LIVE=1` and a `SPRITE_TOKEN`:
`PYDANTIC_AI_HARNESS_SPRITES_LIVE=1 uv run pytest -m sprites_live tests/harness/sprites_sandbox/test_mounts_live.py`.
One Sprite serves every scenario, each at a folder of its own, as the helper's install and download happen once per
Sprite. The scenarios are in `tests/harness/mounts/_scenarios_live.py`.
"""

from __future__ import annotations

import io
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

import pytest
from sprites import AsyncSprite, AsyncSpritesClient
from sprites.exceptions import ExecError, NotFoundError

from pydantic_ai.workspaces import CommandResult, Workspace
from pydantic_ai_harness import S3Mount
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox, SpritesSandboxBackend

from ..mounts._scenarios_live import SCENARIOS, SECRET, STORE, LiveSandbox, in_run, s3, start_store

pytestmark = pytest.mark.sprites_live

_SOCKET = '/dev/shm/pydantic-ai-mounts/rc.sock'


_MARKER = 'pai-live-stderr-follows'
_SEPARATED = f'err=$(mktemp); sh -c "$1" 2> "$err"; status=$?; printf "\\n%s\\n" {_MARKER}; cat "$err"; rm -f "$err"; exit $status'
"""Runs a script with its stderr printed on stdout after a marker line, as the Sprite's stderr stream is not dependable."""


async def _run_as_root(sprite: AsyncSprite, script: str, stdin: bytes) -> CommandResult:
    """Through `sudo`, as the Sprite's own user."""
    # A script that reads no input waits for its end instead, as output printed before the client attaches is lost.
    gated = script if stdin else f'cat > /dev/null; {script}'
    command = sprite.command('sudo', '-n', 'sh', '-c', _SEPARATED, 'sh', gated, cwd='/', stdin=io.BytesIO(stdin))
    try:
        output, exit_code = await command.output(), 0
    except ExecError as error:
        output, exit_code = error.stdout, error.exit_code()
    stdout, _, stderr = output.decode().partition(f'\n{_MARKER}\n')
    return CommandResult(exit_code=exit_code, stdout=stdout, stderr=stderr)


@pytest.fixture(scope='module')
async def live(sprites_token: str) -> AsyncIterator[LiveSandbox]:
    async with AsyncSpritesClient(token=sprites_token) as client:
        backend = SpritesSandboxBackend(client=client)
        sprite = await backend.get_sandbox()
        try:
            assert backend.ref is not None
            live = LiveSandbox(
                ref=backend.ref,
                working_dir=await backend.working_dir(),
                capability=lambda mounts: SpritesSandbox(client=client, mounts=mounts),
                run_as_root=lambda script, stdin: _run_as_root(sprite, script, stdin),
            )
            await start_store(live)
            yield live
        finally:
            try:
                await sprite.delete()
            except NotFoundError:
                pass


@pytest.mark.parametrize('scenario', [pytest.param(scenario, id=scenario.__name__) for scenario in SCENARIOS])
async def test_mounts(live: LiveSandbox, scenario: Callable[[LiveSandbox], Awaitable[None]]) -> None:
    await scenario(live)


async def test_the_sprites_own_user_can_read_the_daemons_keys(live: LiveSandbox) -> None:
    """The Sprite's user holds `CAP_DAC_OVERRIDE` and `CAP_SETUID` (seen 2026-11-10), so the root-only folder around the
    daemon's socket does not stop it, and the socket hands out every mount's keys. The docs say so; this pins it.
    """
    mounts: Sequence[S3Mount] = [s3('daemon')]

    async def dump(workspace: Workspace) -> CommandResult:
        return await workspace.run(
            f'id -un; {STORE}/rclone rc --unix-socket {_SOCKET} --url http://localhost/ config/dump', shell=True
        )

    result = await in_run(live, mounts, dump)

    assert result.exit_code == 0, result.stderr
    assert result.stdout.splitlines()[0] == 'sprite'
    assert SECRET in result.stdout
