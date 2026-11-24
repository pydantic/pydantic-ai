"""Integration tests that require a real Daytona sandbox.

The fake-backed suites cover the harness-owned logic: deadline handling, protocol translation,
and exception mapping. This live tier admits only what a correctly written fake could not catch:
real process execution, a client-owned deadline over a real process, one filesystem shared by
Daytona's file API and its process sessions, and real lifecycle state in Daytona's control plane.

Admission rule:
  A test belongs here only when its docstring can name the fake-encoded assumption it
  validates against real Daytona behavior.

Gating:
  * `daytona_live` marker separates this tier from fake-backed tests.
  * skipped unless `PYDANTIC_AI_HARNESS_DAYTONA_LIVE=1` opts in explicitly.
  * also requires a non-empty `DAYTONA_API_KEY`; without one it skips, or fails when
    `DAYTONA_REQUIRE_LIVE=1` (CI's live job). The gate is `pytest_runtest_setup` in `conftest.py`.
  * the async `client` fixture opens a Daytona client per test and closes it when the test ends.

Run locally:
`PYDANTIC_AI_HARNESS_DAYTONA_LIVE=1 uv run pytest -m daytona_live tests/harness/daytona_sandbox/test_daytona_live.py`
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

import anyio
import daytona
import pytest
from pytest_examples import CodeExample

from pydantic_ai.computer_use import Click, PressKeys, Scroll, TypeText
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceTimeoutError, WorkspaceUnavailableError
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox, DaytonaSandboxBackend

from .._docs_examples import documented_cleanup, python_blocks, run_block
from .._tool_calls import call_tools
from ..computer_use.display_contract_live import check_display_contract, png_size
from ..computer_use.display_fakes import full_display
from .conftest import LIVE_AUTO_STOP_INTERVAL

pytestmark = pytest.mark.daytona_live


def _unique(prefix: str) -> str:
    return f'{prefix}-{uuid.uuid4().hex}'


@pytest.fixture
async def client() -> AsyncIterator[daytona.AsyncDaytona]:
    """A caller-owned API client for one test's backends, closed when the test ends.

    Not module-scoped: the client's event socket must not stay open through the synchronous docs
    tests, which block the loop for minutes, or socket.io reconnects it while the client closes.
    """
    client = daytona.AsyncDaytona()
    try:
        yield client
    finally:
        await client.close()
    # aiohttp closes SSL transports on the next loop ticks after ClientSession.close().
    # Let them finish before pytest tears down this module's event loop.
    await asyncio.sleep(0.25)


@asynccontextmanager
async def _owned(
    client: daytona.AsyncDaytona, desktop: bool | tuple[int, int] = False
) -> AsyncGenerator[DaytonaSandboxBackend]:
    """Create a sandbox and delete it on the way out, even when the test deleted it already."""
    backend = DaytonaSandboxBackend(client=client, auto_stop_interval=LIVE_AUTO_STOP_INTERVAL, desktop=desktop)
    native = await backend.get_sandbox()
    try:
        yield backend
    finally:
        try:
            await native.refresh_data()
            if native.state not in (daytona.SandboxState.DESTROYING, daytona.SandboxState.DESTROYED):
                await native.delete()
        except daytona.DaytonaNotFoundError:
            pass
        # Daytona accepts deletion asynchronously; confirm this test's ID has disappeared.
        for _ in range(60):
            try:
                await client.get(native.id)
            except daytona.DaytonaNotFoundError:
                break
            await asyncio.sleep(1)
        else:
            raise AssertionError(f'Daytona sandbox {native.id} was not deleted')


async def test_destroy_by_ref_does_not_start_sandbox(
    client: daytona.AsyncDaytona, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validates the fake-encoded assumptions that SDK get + delete does not resume a stopped sandbox,
    and that a sandbox being deleted reports the `destroying` state, so destroying it again returns quietly.

    `delete()` does not wait, so Daytona may already have dropped the sandbox's record, which the
    harness also treats as deleted."""

    async def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError('destroy started the sandbox')

    async with _owned(client) as backend:
        assert backend.ref is not None
        await (await client.get(backend.ref.id)).stop()
        monkeypatch.setattr(daytona.AsyncSandbox, 'start', refuse)
        await DaytonaSandbox(client=client).destroy(backend.ref)
        try:
            state = (await client.get(backend.ref.id)).state
        except daytona.DaytonaNotFoundError:
            state = None
        assert state in (None, daytona.SandboxState.DESTROYING, daytona.SandboxState.DESTROYED)
        await DaytonaSandbox(client=client).destroy(backend.ref)


async def test_creates_a_fresh_sandbox_and_runs_a_command(client: daytona.AsyncDaytona) -> None:
    """Validates the fake-encoded assumption that a session command reports real output and exit code."""
    async with _owned(client) as backend:
        result = await backend.run('echo out; echo err 1>&2; exit 3', shell=True, timeout=60)

        assert backend.ref is not None
        assert (result.stdout.strip(), result.stderr.strip(), result.exit_code) == ('out', 'err', 3)


async def test_large_output_is_exact(client: daytona.AsyncDaytona) -> None:
    """Validates the fake-encoded assumption that a finished command's stored logs are its exact output.

    The SDK's log stream corrupts output above a few KB (it misreads a stream prefix split across
    websocket frames), so the result must not be built from it.
    """
    async with _owned(client) as backend:
        result = await backend.run(['seq', '1', '50000'], timeout=120)

        assert result.stdout == ''.join(f'{i}\n' for i in range(1, 50001))


async def test_reattach_by_ref_reads_a_file_the_first_backend_wrote(client: daytona.AsyncDaytona) -> None:
    """Validates the fake-encoded assumption that `client.get` plus `start` reaches the same sandbox."""
    path = f'/tmp/{_unique("reattach")}.txt'
    async with _owned(client) as owner:
        await owner.write_bytes(path, b'shared')
        assert owner.ref is not None

        attached = DaytonaSandboxBackend(client=client, ref=owner.ref)

        assert await attached.read_bytes(path) == b'shared'
        assert attached.ref == owner.ref


async def test_command_and_file_api_share_one_filesystem(client: daytona.AsyncDaytona) -> None:
    """Validates the protocol's one-environment contract: process sessions and `sandbox.fs` see one tree."""
    root = f'/tmp/{_unique("roundtrip")}'
    async with _owned(client) as backend:
        await backend.write_bytes(f'{root}/nested/in.txt', b'from-file-api\n')
        result = await backend.run(
            ['sh', '-c', 'cat "$1/nested/in.txt" && printf from-shell > "$1/out.txt"', 'sh', root], timeout=60
        )

        assert (result.exit_code, result.stdout) == (0, 'from-file-api\n')
        assert await backend.read_bytes(f'{root}/out.txt') == b'from-shell'


async def test_timeout_raises_with_the_partial_output(client: daytona.AsyncDaytona) -> None:
    """Validates the fake-encoded assumption that a command's output files hold what it printed before it ends.

    The deadline is enforced client-side, so the output printed before it expired must reach
    the `WorkspaceTimeoutError`, and deleting the session must stop the command.
    """
    marker = f'/tmp/{_unique("after-deadline")}'
    async with _owned(client) as backend:
        with pytest.raises(WorkspaceTimeoutError) as exc_info:
            await backend.run(f'echo DIAGNOSTIC; sleep 20; touch {marker}', shell=True, timeout=5)

        assert 'DIAGNOSTIC' in exc_info.value.stdout
        assert '5' in str(exc_info.value)
        assert (await backend.run(['sleep', '20'], timeout=60)).exit_code == 0
        assert await backend.exists(marker) is False


async def test_timeout_stops_only_its_command(client: daytona.AsyncDaytona) -> None:
    """Checks against the real SDK that deleting a session stops its process, not the shared sandbox."""
    async with _owned(client) as backend:
        marker = f'/tmp/{_unique("daytona-stopped")}'
        with pytest.raises(WorkspaceTimeoutError):
            await backend.run(f'sleep 5; touch {marker}', shell=True, timeout=1)
        # Outlasts the stopped command's `sleep 5`, so a command left running would have written the marker.
        assert (await backend.run(['sleep', '6'], timeout=30)).exit_code == 0
        assert not await backend.exists(marker)


async def test_a_cancelled_run_stops_the_command(client: daytona.AsyncDaytona) -> None:
    """Validates the protocol's cancellation contract against real Daytona.

    A cancelled `run()` deletes the command's session; the marker written after the cancellation
    must never appear.
    """
    marker = f'/tmp/{_unique("cancelled")}'
    pid_file = f'{marker}.pid'
    async with _owned(client) as backend:
        with anyio.move_on_after(5):
            await backend.run(f'echo $$ > {pid_file}; sleep 15; touch {marker}', shell=True, timeout=60)

        # The session is deleted by the time `run()` returns, so a command that had not started yet never will.
        if await backend.exists(pid_file):
            with anyio.fail_after(60):
                pid = (await backend.read_bytes(pid_file)).decode().strip()
                assert pid.isdigit()
                while (await backend.run(['kill', '-0', pid], timeout=30)).exit_code == 0:
                    await anyio.sleep(0.2)
        else:
            assert (await backend.run(['sleep', '16'], timeout=30)).exit_code == 0
            assert await backend.exists(pid_file) is False
        assert await backend.exists(marker) is False


async def test_a_new_sandbox_gets_its_working_dir(client: daytona.AsyncDaytona) -> None:
    """Validates the fake-encoded assumption that `mkdir -p` through `process.exec` makes `working_dir`
    on a new sandbox as the non-root default user, so the first command can start there."""
    directory = f'/home/daytona/{_unique("project")}/nested'
    backend = DaytonaSandboxBackend(client=client, auto_stop_interval=LIVE_AUTO_STOP_INTERVAL, working_dir=directory)
    try:
        result = await backend.run(['pwd', '-P'], timeout=60)
        assert (result.exit_code, result.stdout.strip()) == (0, directory)
    finally:
        if backend.ref is not None:
            await DaytonaSandbox(client=client).destroy(backend.ref)


async def test_detached_child_does_not_hold_command_result(client: daytona.AsyncDaytona) -> None:
    """Validates the docs claim that `run()` returns when the command exits, while a background child it started runs on."""
    async with _owned(client) as backend:
        result = await backend.run('sleep 4 & echo ready', shell=True, timeout=3)
        assert result.exit_code == 0
        assert result.stdout.strip() == 'ready'


async def test_reattach_to_a_deleted_sandbox_is_unavailable(client: daytona.AsyncDaytona) -> None:
    """Validates the fake-encoded assumption about deletion: `delete()` returns while the sandbox is
    still `destroying`, and a later attach or a call on the held handle is reported as gone.

    If Daytona answers either differently, a deleted sandbox would surface as a retryable error
    or, worse, as a missing file.
    """
    path = f'/tmp/{_unique("deleted")}.txt'
    async with _owned(client) as owner:
        await owner.write_bytes(path, b'before-delete')
        assert owner.ref is not None
        await (await owner.get_sandbox()).delete()

        with pytest.raises(WorkspaceUnavailableError):
            await DaytonaSandboxBackend(client=client, ref=owner.ref).working_dir()
        with pytest.raises(WorkspaceUnavailableError):
            await owner.read_bytes(path)
        with pytest.raises(WorkspaceUnavailableError):
            await owner.run(['true'], timeout=30)


async def test_coder_tools_run_in_the_sandbox(client: daytona.AsyncDaytona) -> None:
    """Validates the fake-encoded assumption that a relative path the file API writes is where a
    session command started in the working directory finds it, so `Coder`'s tools agree."""
    async with _owned(client) as backend:
        results = await call_tools(
            [Coder()],
            [
                ('write_file', {'path': 'hello.py', 'content': "print('hi from daytona')\n"}),
                ('shell', {'command': 'python3 hello.py'}),
            ],
            workspace=backend,
        )

        assert 'hi from daytona' in results[1]


async def test_commands_get_a_usable_environment(client: daytona.AsyncDaytona) -> None:
    """Validates the fake-encoded assumption that a command sees the snapshot's `PATH` and `HOME`, and
    what the docs say the default snapshot has: the non-root `daytona` user in `/home/daytona`, git and
    Python and ripgrep. A per-call `env` adds to the environment rather than replacing it."""
    async with _owned(client) as backend:
        result = await backend.run(
            'echo "$PATH"; echo "$HOME"; whoami; pwd -P; git --version; python3 --version; command -v rg',
            shell=True,
            timeout=60,
        )
        path, home, user, cwd, git, python, rg = result.stdout.splitlines()
        assert (home, user, cwd) == ('/home/daytona', 'daytona', '/home/daytona')
        # Only being on `PATH` matters; where the snapshot installs ripgrep is Daytona's choice.
        assert rg.endswith('/rg')
        assert path and git.startswith('git version') and python.startswith('Python 3')
        assert await backend.working_dir() == '/home/daytona'

        result = await backend.run('echo "$FOO"; echo "$PATH"', shell=True, env={'FOO': '1'}, timeout=60)
        assert result.stdout.splitlines() == ['1', path]


async def test_a_new_desktop_sandbox_below_1024x768_gets_that_size(client: daytona.AsyncDaytona) -> None:
    """Creates a new sandbox. Validates the fake-encoded assumption that `VNC_RESOLUTION` gives a new sandbox a
    display below Daytona's 1024x768 default, which the display and its screenshots then report."""
    async with _owned(client, desktop=(800, 600)) as backend:
        display = backend.display
        assert display is not None
        assert await display.display_size() == (800, 600)
        assert png_size(await display.screenshot()) == (800, 600)


_EVENT_LOGGER = """
import tkinter as tk
root = tk.Tk()
root.geometry('400x300+0+0')
log = open('events.txt', 'w')
def write(line):
    log.write(line + '\\n')
    log.flush()
root.bind_all('<KeyPress>', lambda event: write(f'key {event.keysym}'))
root.bind('<ButtonPress>', lambda event: write(f'button {event.num} state {event.state}'))
root.after(500, write, 'ready')
root.mainloop()
"""
"""A window at the top left that logs to `events.txt` each key it gets and each mouse button with X's modifier state."""


class TestLiveDaytonaDesktop:
    """Runs the display against a sandbox started with a desktop."""

    @pytest.fixture(scope='class')
    @classmethod
    async def backend(cls) -> AsyncIterator[WorkspaceBackend]:
        backend = DaytonaSandboxBackend(auto_stop_interval=LIVE_AUTO_STOP_INTERVAL, desktop=True)
        try:
            yield backend
        finally:
            try:
                if backend.ref is not None:
                    await DaytonaSandbox().destroy(backend.ref)
            finally:
                await backend.aclose()

    async def test_display_contract(self, backend: WorkspaceBackend) -> None:
        """The desktop is 1280x800, and its screenshots, zoom and pointer agree with that size."""
        display = full_display(backend)
        assert await display.display_size() == (1280, 800)
        await check_display_contract(display)

    async def test_commands_see_the_desktop(self, backend: WorkspaceBackend) -> None:
        """Shell commands get `DISPLAY`, so programs they start open on the desktop."""
        assert isinstance(backend, DaytonaSandboxBackend)
        assert (await backend.run('echo "$DISPLAY"', shell=True)).stdout == ':0\n'

    async def test_modifiers_are_held_around_clicks_and_scrolls(self, backend: WorkspaceBackend) -> None:
        """Validates the mapping the fake assumes: Daytona holds a click's or scroll's modifiers around it, and `cmd`,
        which `super` maps to, is `Super_L`."""
        assert isinstance(backend, DaytonaSandboxBackend)
        display = full_display(backend)
        await display.display_size()  # starts the desktop, so the window opens on it
        await backend.write_bytes('log_events.py', _EVENT_LOGGER.encode())
        await backend.run('rm -f events.txt; setsid python3 log_events.py >/dev/null 2>&1 &', shell=True)
        await backend.run('until grep -q ready events.txt 2>/dev/null; do sleep 0.1; done', shell=True, timeout=30)

        await display.act(Click(x=200, y=150))  # focuses the window, so it gets the key events
        await display.act(Click(x=200, y=150, modifiers=('shift',)))
        await display.act(Scroll(x=200, y=150, direction='down', amount=1, modifiers=('ctrl',)))
        await display.act(Click(x=200, y=150, modifiers=('super',)))

        # Up to 10 seconds for the window to log all four presses.
        wait = 'i=0; until [ "$(grep -c ^button events.txt)" -ge 4 ] || [ $i -ge 100 ]; do sleep 0.1; i=$((i+1)); done'
        logged = (await backend.run(f'{wait}; cat events.txt', shell=True)).stdout.splitlines()
        # X's state bits: 1 is shift, 4 is control, 64 is Mod4, which Super_L is on.
        assert [line for line in logged if line.startswith('button')] == [
            'button 1 state 0',
            'button 1 state 1',
            'button 5 state 4',
            'button 1 state 64',
        ]
        assert 'key Super_L' in logged

    async def test_keys_and_text_reach_the_focused_window(self, backend: WorkspaceBackend) -> None:
        """Validates the key mapping the fake assumes: a shifted symbol is shift and its key on Daytona's display."""
        assert isinstance(backend, DaytonaSandboxBackend)
        display = backend.display
        assert display is not None
        await display.display_size()  # starts the desktop, so the terminal opens on it
        # Its own process, so a terminal already running in the session doesn't take the window and its directory.
        # The terminal's shell writes `ready` once the window is up; Daytona's desktop has no xdotool to wait with.
        await backend.run(
            'rm -f ready typed.txt; setsid xfce4-terminal --disable-server --geometry 60x10+100+100 '
            '-x sh -c "touch ready; cat > typed.txt" >/dev/null 2>&1 &',
            shell=True,
        )
        waited = await backend.run('until [ -e ready ]; do sleep 0.1; done', shell=True, timeout=30)
        assert waited.exit_code == 0

        await display.act(Click(x=300, y=180))
        await display.act(TypeText(text='hello,\tdesktop'))
        await display.act(PressKeys(keys=('!',)))
        await display.act(PressKeys(keys=('enter',)))

        typed = await backend.run('until [ -s typed.txt ]; do sleep 0.1; done; cat typed.txt', shell=True, timeout=30)
        assert typed.stdout == 'hello,\tdesktop!\n'


# The README's Python blocks are the same as this page's.
_DOCS_BLOCKS = python_blocks('docs/harness/daytona-sandbox.md')


@pytest.mark.parametrize('example', [pytest.param(block, id=f'line {block.start_line}') for block in _DOCS_BLOCKS])
def test_docs_example(example: CodeExample) -> None:
    """Every example on the docs page runs as written, and its agent's tools do their work in a real sandbox.

    The fake stands in for Daytona, so only this shows the page's code, its default settings, and its
    cleanup work against the real service. A follow-up run, from the message history or a stored ref,
    works in the first run's sandbox.
    """
    _, runs = run_block(example, cleanup=documented_cleanup(_DOCS_BLOCKS, 'delete_sandbox'))
    assert all(run.used_sandbox for run in runs), runs
    assert len({run.ref for run in runs}) <= 1, runs
