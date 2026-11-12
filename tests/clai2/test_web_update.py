"""`/update` from the web UI: the page follows the install, and the new build serves the same page again."""

import io
import re
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anyio
import httpx
import pytest
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_clai2 import chat
from pydantic_clai2.builtin_plugins.web.app import HEADER
from pydantic_clai2.builtin_plugins.web.protocol import (
    PROTOCOL,
    DialogOpened,
    Notice,
    Snapshot,
    UpdateChanged,
    UpdateInfo,
)
from pydantic_clai2.cli.self_update import Relaunch, Updates, WebHandoff, installed
from pydantic_clai2.cli.web import run_web
from pydantic_clai2.config import PluginSettings, Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.runtime import session_runtime
from pydantic_clai2.runtime.session_runtime import create_shell
from pydantic_clai2.ui.menus.update_menu import VERSION
from tests.clai2.test_app_edges import inputs
from tests.clai2.test_self_update import WEB_TOKEN, _updates  # pyright: ignore[reportPrivateUsage]
from tests.clai2.test_web import TOKEN, WIRE, MakeWeb, make_web as make_web, receive_until
from tests.clai2.test_web_parts import _Output  # pyright: ignore[reportPrivateUsage]

WEB = PluginSettings(id='web', factory='pydantic_clai2.builtin_plugins.web', settings={'open_browser': False})
UV_LINES = ['Resolved 3 packages', 'Installed 3 packages']


def _use(monkeypatch: pytest.MonkeyPatch, updates: Updates) -> None:
    def build(*, channel: object) -> Updates:
        return updates

    monkeypatch.setattr(session_runtime, 'Updates', build)


@pytest.mark.parametrize('code', [0, 1])
def test_the_page_follows_an_update(code: int, make_web: MakeWeb, monkeypatch: pytest.MonkeyPatch) -> None:
    """The page shows uv's lines as they come, then all of them, then a restart, or the failure and nothing more."""
    updates, _ = _updates(['stable'], codes=[code], lines=UV_LINES)
    _use(monkeypatch, updates)
    web_ = make_web(Agent(TestModel()))
    # The test client serves the page without `/web`, which is what tells CLAI a page watches.
    updates.capture = lambda: True
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket_:
        snapshot = WIRE.validate_json(socket_.receive_text())
        assert isinstance(snapshot, Snapshot)
        assert (snapshot.version, snapshot.update) == (installed().label, None)
        running = web_.run()
        # Typed in the page, `/update` opens Settings > About, whose version row installs once confirmed.
        assert web_.post('/api/input', {'text': '/update'}).json() == {'menu': 'update'}
        with ThreadPoolExecutor(1) as pool:
            action = pool.submit(web_.post, f'/api/commands/update/fields/{VERSION}/action')
            confirm = receive_until(socket_, DialogOpened)[1].dialog
            assert web_.post(f'/api/dialogs/{confirm.id}', {'value': 'yes'}).status_code == 204
            assert action.result(timeout=10).status_code == 200
        received, output = receive_until(socket_, Notice)
        assert [message.update for message in received if isinstance(message, UpdateChanged)] == [
            UpdateInfo(target='0.53.0'),
            UpdateInfo(target='0.53.0', line='Resolved 3 packages'),
            UpdateInfo(target='0.53.0', line='Installed 3 packages'),
        ]
        assert (output.text, output.level) == ('\n'.join(UV_LINES), 'error' if code else 'info')
        _, changed = receive_until(socket_, UpdateChanged)
        _, result = receive_until(socket_, Notice)
        if code:
            assert changed.update is None
            assert result.text == 'uv exited with status 1; see its output above.'
            web_.post('/api/input', {'text': '/exit'})
        else:
            assert changed.update == UpdateInfo(target='0.53.0', restarting=True)
            assert result.text == 'Updated CLAI to 0.53.0 (stable). Restarting...'
            # A page that connects, or reloads, before CLAI exits still shows the restart.
            assert web_.get('/api/state').json()['update'] == {'target': '0.53.0', 'line': '', 'restarting': True}
        assert running.result(timeout=10) == 'exit'


async def test_uv_draws_on_the_terminal_until_a_page_watches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With `web` loaded but not serving, nobody but the terminal watches uv, so its progress bars stay."""
    updates, _ = _updates(['stable'])
    _use(monkeypatch, updates)
    shell = create_shell(
        Agent(TestModel()),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=io.StringIO()),
        settings=None,
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=[WEB],
        project=ProjectSettings(),
        headless=True,
    )
    await shell.loader.load_all()
    assert not updates.capture()
    assert updates.serving() is None
    await shell.commands.execute_async('/web')
    assert updates.capture()
    assert updates.serving() is not None
    await shell.loader.close('exit')


async def test_run_web_restarts_on_the_same_port_and_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`/update` under `clai2 --web` raises `Relaunch` with the web UI's port and token, for the new build to serve."""
    monkeypatch.chdir(tmp_path)
    opened: list[str] = []
    monkeypatch.setattr('pydantic_clai2.builtin_plugins.web.open_browser', opened.append)
    updates, _ = _updates(['stable'])
    _use(monkeypatch, updates)
    output = _Output()

    async def update() -> None:
        await output.ready.wait()
        found = re.search(r'(http://\S+)/\?token=(\S+)', output.getvalue())
        assert found is not None
        async with httpx.AsyncClient(base_url=found[1], headers={HEADER: found[2]}) as client:
            assert (await client.post('/api/input', json={'text': '/update'})).json() == {'menu': 'update'}
            async with anyio.create_task_group() as confirming:
                confirming.start_soon(client.post, f'/api/commands/update/fields/{VERSION}/action')
                dialogs: list[dict[str, str]] = []
                while not dialogs:
                    await anyio.sleep(0.01)
                    dialogs = (await client.get('/api/state')).json()['dialogs']
                await client.post(f'/api/dialogs/{dialogs[0]["id"]}', json={'value': 'yes'})

    relaunched: list[Relaunch] = []
    with anyio.fail_after(30):
        async with anyio.create_task_group() as group:
            group.start_soon(update)
            try:
                await run_web(
                    settings_for=lambda project: Settings(model='test'),
                    store=SettingsStore(tmp_path / 'config.db'),
                    project=ProjectSettings(),
                    console=Console(file=output, force_terminal=False, width=500),
                )
            except Relaunch as relaunch:
                relaunched.append(relaunch)
    found = re.search(r'http://127\.0\.0\.1:(\d+)/\?token=(\S+)', output.getvalue())
    assert found is not None
    [relaunch] = relaunched
    assert relaunch.executable == '/tools/clai2'
    assert relaunch.web == WebHandoff(port=int(found[1]), token=found[2])


@pytest.mark.parametrize('launcher', [False, True])
async def test_run_web_serves_the_handed_over_web_ui(
    launcher: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The new build serves the page's token again and opens no browser: the page is open already."""
    opened: list[str] = []
    monkeypatch.setattr('pydantic_clai2.builtin_plugins.web.open_browser', opened.append)
    output = _Output()

    async def leave() -> None:
        await output.ready.wait()
        found = re.search(r'Web UI running at (http://127\.0\.0\.1:\d+)/(\S*)', output.getvalue())
        assert found is not None
        # A launcher keeps the token itself, so the address it reads leaves it out.
        assert found[2] == ('' if launcher else f'?token={WEB_TOKEN}')
        async with httpx.AsyncClient(base_url=found[1], headers={HEADER: WEB_TOKEN}) as client:
            assert (await client.post('/api/input', json={'text': '/exit'})).json() == {'menu': None}

    with anyio.fail_after(30):
        async with anyio.create_task_group() as group:
            group.start_soon(leave)
            await run_web(
                settings_for=lambda project: Settings(model='test'),
                store=SettingsStore(tmp_path / 'config.db'),
                project=ProjectSettings(),
                console=Console(file=output, force_terminal=False, width=500),
                port=0,
                token=WEB_TOKEN if launcher else None,
                restart=WebHandoff(port=0, token=WEB_TOKEN),
            )
    assert opened == []


async def _chat(tmp_path: Path, output: io.StringIO, *, web: WebHandoff | None = None, plugins: bool = True) -> None:
    await chat(
        Agent(TestModel()),
        deps=None,
        settings=Settings(model='test', update_channel='stable'),
        console=Console(file=output, width=500),
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=[WEB] if plugins else [],
        web=web,
    )


async def test_a_terminal_session_hands_its_web_ui_over(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After `/web`, `/update` in a terminal session hands the web UI's port and token to the new build."""
    updates, _ = _updates(['stable'])
    _use(monkeypatch, updates)
    inputs(monkeypatch, ['/web', '/update now', '/exit'])
    output = io.StringIO()
    with pytest.raises(Relaunch) as raised:
        await _chat(tmp_path, output)
    found = re.search(r'http://127\.0\.0\.1:(\d+)/\?token=(\S+)', output.getvalue())
    assert found is not None
    assert raised.value.web == WebHandoff(port=int(found[1]), token=found[2])


@pytest.mark.parametrize('case', ['served', 'off', 'taken'])
async def test_a_terminal_session_serves_its_web_ui_again(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The new build serves the web UI it was handed, or says why it could not and how to start one."""
    inputs(monkeypatch, ['/exit'])
    output = io.StringIO()
    with socket.socket() as taken:
        taken.bind(('127.0.0.1', 0))
        taken.listen()
        port = taken.getsockname()[1] if case == 'taken' else 0
        await _chat(tmp_path, output, web=WebHandoff(port=port, token=WEB_TOKEN), plugins=case != 'off')
    text = output.getvalue()
    if case == 'served':
        assert re.search(rf'Web UI running at http://127\.0\.0\.1:\d+/\?token={WEB_TOKEN}', text)
    elif case == 'off':
        assert 'The web UI did not start again: the web plugin is off.' in text
    else:
        assert f'The web UI did not start again on port {port}: ' in text
