"""Integration tests that boot real BoxLite VMs.

The fake-backed suites cover the backend's own logic. This tier admits only what a correctly
written fake could get wrong: real shells in real images, a box's process table, the runtime's
lock on its home directory, its network policy and mounts, and every example on the docs page.

Admission rule:
  A test belongs here only when its docstring can name the fake-encoded assumption it validates
  against real BoxLite behavior.

Gating:
  * `boxlite_live` marker separates this tier from fake-backed tests.
  * skipped unless `PYDANTIC_AI_HARNESS_BOXLITE_LIVE=1`, on a host that can run BoxLite VMs. The gate
    is `pytest_runtest_setup` in `conftest.py`.

Run locally:
`PYDANTIC_AI_HARNESS_BOXLITE_LIVE=1 uv run pytest -m boxlite_live tests/harness/boxlite_sandbox`
"""

from __future__ import annotations

import asyncio
import gc
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import anyio
import boxlite
import pytest
from pytest_examples import CodeExample

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import WorkspaceOutputLimitError, WorkspaceTimeoutError, WorkspaceUnavailableError
from pydantic_ai_harness.boxlite_sandbox import BoxLiteSandbox, BoxLiteSandboxBackend

from .._docs_examples import documented_cleanup, python_blocks, run_block

pytestmark = pytest.mark.boxlite_live

_IMAGES = ['alpine:3.20', 'python:3.12-slim']
"""BusyBox `sh` and `setsid`, and dash with util-linux `setsid`: the shells the script must work in."""


@pytest.fixture
async def owned() -> AsyncIterator[list[BoxLiteSandboxBackend]]:
    """Backends a test creates, each deleted afterwards."""
    backends: list[BoxLiteSandboxBackend] = []
    yield backends
    for backend in backends:
        if backend.ref is not None:
            await BoxLiteSandbox[None]().destroy(backend.ref)


def _status(name: str) -> str | None:
    async def status() -> str | None:
        info = await cast(Any, boxlite).Boxlite.default().get_info(name)
        return None if info is None else info.state.status

    return anyio.run(status)


@pytest.mark.parametrize('image', _IMAGES)
async def test_a_timeout_stops_the_command_and_what_it_started(owned: list[BoxLiteSandboxBackend], image: str) -> None:
    """Validates the fake's assumption that `setsid` puts the command in a session the stop kills whole.

    The fake runs host processes; only a real box shows the image's `setsid`, and that PID 1 never
    reaps, so the script must reap the command itself.
    """
    backend = BoxLiteSandboxBackend(image=image)
    owned.append(backend)
    with pytest.raises(WorkspaceTimeoutError) as caught:
        await backend.run('echo started; sleep 300 & echo $! > /tmp/child; exec sleep 301', shell=True, timeout=2)
    assert caught.value.stdout == 'started\n'
    child = (await backend.run(['cat', '/tmp/child'])).stdout.strip()
    # Killed; an orphaned child stays a zombie of PID 1, which doesn't reap.
    state = await backend.run(['sh', '-c', 'cut -d" " -f3 "/proc/$1/stat" 2>/dev/null || echo gone', 'sh', child])
    assert state.stdout.strip() in {'Z', 'gone'}


@pytest.mark.parametrize('image', _IMAGES)
async def test_the_script_works_with_any_command_env_and_program_name(
    owned: list[BoxLiteSandboxBackend], image: str
) -> None:
    """Validates the fake's assumption that the script uses only builtins before `setsid` takes over.

    The command's `env` applies to the whole script, so a `PATH` or `IFS` that breaks the image's own
    tools must not break the backend; a program named like an option must not reach `setsid` as one.
    """
    backend = BoxLiteSandboxBackend(image=image, env={'IFS': ':'})
    owned.append(backend)
    assert (await backend.run(['/bin/echo', 'a b'])).stdout == 'a b\n'
    assert (await backend.run(['ls'], env={'PATH': '/nonexistent'})).exit_code == 127
    assert (await backend.run(['-n'])).exit_code == 127
    killed = await backend.run(['sh', '-c', 'kill -9 $$'])
    assert killed.exit_code == 137


async def test_output_past_the_limit_stops_the_command(owned: list[BoxLiteSandboxBackend]) -> None:
    """Validates the fake's assumption that BoxLite streams output as it is written, without dropping any.

    A dropped chunk would raise `WorkspaceError` instead, and output held back until the command ends would
    leave `yes` running.
    """
    backend = BoxLiteSandboxBackend()
    owned.append(backend)
    with anyio.fail_after(60), pytest.raises(WorkspaceOutputLimitError) as caught:
        await backend.run(['yes', 'x'])
    assert caught.value.stdout == 'x\n' * 32 * 1024


async def test_a_stopped_box_ends_its_command_and_boots_again(owned: list[BoxLiteSandboxBackend]) -> None:
    """Validates the fake's assumption that stopping a box ends its commands and the next exec boots it again."""
    backend = BoxLiteSandboxBackend()
    owned.append(backend)
    await backend.run(['sh', '-c', 'echo kept > kept.txt'])
    box: Any = await backend.get_sandbox()
    async with anyio.create_task_group() as group:

        async def command() -> None:
            with pytest.raises(WorkspaceUnavailableError):
                await backend.run(['sleep', '60'])

        group.start_soon(command)
        await anyio.sleep(1)
        await box.stop()
    assert (await backend.run(['cat', 'kept.txt'])).stdout == 'kept\n'


async def test_a_released_box_stops_and_its_ref_boots_it_again(owned: list[BoxLiteSandboxBackend]) -> None:
    """Validates the fake's assumption that a box nothing holds stops, keeps its disk, and attaches by ref."""
    backend = BoxLiteSandboxBackend()
    owned.append(backend)
    await backend.run(['sh', '-c', 'echo kept > kept.txt'])
    ref = backend.ref
    assert ref is not None
    attached = BoxLiteSandboxBackend(ref=ref)
    owned[0] = attached
    del backend
    gc.collect()
    with anyio.fail_after(30):
        while await anyio.to_thread.run_sync(_status, ref.id) != 'stopped':
            await anyio.sleep(0.2)
    assert (await attached.run(['cat', 'kept.txt'])).stdout == 'kept\n'


async def test_a_creation_cancelled_mid_boot_leaves_no_box() -> None:
    """Validates the fake's assumptions that a cancelled `start()` stops the boot, and that the box a create
    made can be deleted once the create returns, so a cancelled first use leaves nothing behind."""
    runtime = cast(Any, boxlite).Boxlite.default()
    backend = BoxLiteSandboxBackend()
    caller = asyncio.create_task(backend.get_sandbox())
    with anyio.fail_after(30):
        while backend.ref is None or await runtime.get_info(backend.ref.id) is None:
            await anyio.sleep(0.01)
    name = backend.ref.id
    caller.cancel()  # A native cancel, as core sends to a tool call.
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert backend.ref is None
    assert await runtime.get_info(name) is None


@pytest.mark.subprocess(reason='opens a second BoxLite runtime on the same home directory')
async def test_a_second_process_on_the_same_home_is_unavailable(owned: list[BoxLiteSandboxBackend]) -> None:
    """Validates the fake's assumption that the runtime panics when another process holds its home directory."""
    backend = BoxLiteSandboxBackend()
    owned.append(backend)
    await backend.run(['true'])
    script = (
        'import asyncio\n'
        'from pydantic_ai.workspaces import WorkspaceUnavailableError\n'
        'from pydantic_ai_harness.boxlite_sandbox import BoxLiteSandboxBackend\n'
        'try:\n'
        "    asyncio.run(BoxLiteSandboxBackend().run(['true']))\n"
        'except WorkspaceUnavailableError as error:\n'
        '    print(error)\n'
    )
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=120, check=True)
    assert 'Could not open the BoxLite runtime' in result.stdout


async def test_allowed_hosts_limit_the_network(owned: list[BoxLiteSandboxBackend]) -> None:
    """Validates the fake's assumption that `allowed_hosts` becomes BoxLite's enforced outbound allow-list."""
    backend = BoxLiteSandboxBackend(image='python:3.12-slim', allow_internet_access=True, allowed_hosts=['example.com'])
    owned.append(backend)
    fetch = 'import sys, urllib.request; print(urllib.request.urlopen(sys.argv[1], timeout=15).status)'
    assert (await backend.run(['python', '-c', fetch, 'https://example.com'], timeout=60)).stdout == '200\n'
    assert (await backend.run(['python', '-c', fetch, 'https://pypi.org'], timeout=60)).exit_code != 0
    offline = BoxLiteSandboxBackend(image='python:3.12-slim')
    owned.append(offline)
    assert (await offline.run(['python', '-c', fetch, 'https://example.com'], timeout=60)).exit_code != 0


async def test_box_options_reach_the_box(owned: list[BoxLiteSandboxBackend], tmp_path: Path) -> None:
    """Validates the fake's assumption that `box_options` arguments reach `boxlite.BoxOptions` as given."""
    (tmp_path / 'data.txt').write_text('from host')
    backend = BoxLiteSandboxBackend(box_options={'volumes': [(str(tmp_path), '/data', True)]})
    owned.append(backend)
    assert (await backend.run(['cat', '/data/data.txt'])).stdout == 'from host'
    assert (await backend.run(['touch', '/data/new.txt'])).exit_code != 0


async def test_a_box_passed_in_is_found_again_after_a_stop() -> None:
    """Validates the fake's assumptions that a box created without a name has `name=None` and is found by
    its id, and that stopping it invalidates the handle the application passed in."""
    runtime = cast(Any, boxlite).Boxlite.default()
    box = await runtime.create(cast(Any, boxlite).BoxOptions(image='alpine:3.20', auto_delete=0))
    try:
        assert box.name is None
        backend = BoxLiteSandboxBackend(sandbox=box)
        assert backend.ref is not None and backend.ref.id == box.id
        assert (await backend.run(['pwd'])).stdout == '/workspace\n'
        await box.stop()
        assert (await backend.run(['echo', 'again'])).stdout == 'again\n'
    finally:
        await runtime.remove(box.id, force=True)


async def test_a_box_of_another_home_is_refused(tmp_path: Path) -> None:
    """Validates the fake's assumption that the default runtime can't see a box another BoxLite home made,
    so the backend must refuse it rather than look its name up in the default home."""
    other = cast(Any, boxlite).Boxlite(cast(Any, boxlite).Options(home_dir=str(tmp_path)))
    box = await other.create(cast(Any, boxlite).BoxOptions(image='alpine:3.20', auto_delete=0))
    try:
        with pytest.raises(UserError, match='not in this process'):
            await BoxLiteSandboxBackend(sandbox=box).run(['true'])
    finally:
        await other.remove(box.id, force=True)


# The README's Python blocks are the same as this page's.
_DOCS_BLOCKS = python_blocks('docs/harness/boxlite-sandbox.md')


@pytest.mark.parametrize('example', [pytest.param(block, id=f'line {block.start_line}') for block in _DOCS_BLOCKS])
def test_docs_example(example: CodeExample) -> None:
    """Every example on the docs page runs as written, and its agent's tools do their work in a real box.

    The fake stands in for BoxLite, so only this shows the page's code, its default settings, and its
    cleanup work against the real runtime. A follow-up run, from the message history or a stored ref,
    works in the first run's box.
    """
    _, runs = run_block(example, cleanup=documented_cleanup(_DOCS_BLOCKS, 'delete_box'))
    assert all(run.used_sandbox for run in runs), runs
    assert len({run.ref for run in runs}) <= 1, runs
