"""`/update`: release lookup, install commands, the footer notice, and the shell exit after an install."""

import argparse
import base64
import json
import os
import shlex
import sys
import threading
from collections.abc import Callable, Sequence
from io import StringIO
from pathlib import Path

import httpx
import pytest
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from pydantic_clai2 import _app, chat
from pydantic_clai2.cli import _cli, self_update
from pydantic_clai2.cli._cli import relaunch_argv
from pydantic_clai2.cli.self_update import (
    COMMITS_URL,
    PYPI_URL,
    Installed,
    Relaunch,
    Update,
    Updates,
    _in_thread,  # pyright: ignore[reportPrivateUsage]
    _run_uv,  # pyright: ignore[reportPrivateUsage]
    after_exit_script,
    find_update,
    find_uv,
    install_after_exit,
    installed,
    latest,
    powershell,
    tool_executable,
)
from pydantic_clai2.commands import set_completions
from pydantic_clai2.config import Settings, UpdateChannel
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.ui.menus.field_menu import FieldMenu
from pydantic_clai2.ui.menus.set_menu import SettingsSource
from tests.clai2.menu_script import make_context
from tests.clai2.test_app_edges import inputs

SHA = '4bd401a5539aa96b9b67753092fd7f8b90e4dc8a'
NEWER = '9e08a34ee0000000000000000000000000000000'
TOOL = Installed(version='0.52.1.dev54+4bd401a55', commit=SHA, tool=True)


def test_channel_setting(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    assert context.settings.update_channel == 'stable'
    assert set_completions(['updates.channel', '']) == ('stable', 'bleeding')
    row = next(row for row in FieldMenu(SettingsSource(context)).rows if row.key == 'updates.channel')
    assert row.choices == ('stable', 'bleeding')
    assert context.set_setting(['updates.channel', 'bleeding']) == 'Saved updates.channel. Applied.'
    assert SettingsStore(context.store.path).load().update_channel == 'bleeding'
    with pytest.raises(ValueError, match='stable'):
        context.set_setting(['updates.channel', 'nightly'])
    assert SettingsStore(context.store.path).load().update_channel == 'bleeding'


class _Distribution:
    def __init__(self, direct_url: str | None) -> None:
        self.version = '0.52.0'
        self._direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        assert filename == 'direct_url.json'
        return self._direct_url


@pytest.mark.parametrize(
    ('direct_url', 'commit'),
    [
        (None, None),
        ('{"url":"file:///src","dir_info":{"editable":true}}', None),
        (
            json.dumps(
                {'url': 'https://github.com/pydantic/pydantic-ai', 'vcs_info': {'vcs': 'git', 'commit_id': SHA}}
            ),
            SHA,
        ),
        (
            json.dumps(
                {
                    'url': f'https://github.com/pydantic/pydantic-ai/archive/{SHA}.tar.gz',
                    'archive_info': {},
                    'subdirectory': 'src/pydantic_clai2',
                }
            ),
            SHA,
        ),
        ('{"url":"https://example.com/archive/x.tar.gz","archive_info":{}}', None),
    ],
)
def test_installed_reads_metadata_and_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, direct_url: str | None, commit: str | None
) -> None:
    def distribution(name: str) -> _Distribution:
        assert name == 'pydantic-clai2'
        return _Distribution(direct_url)

    monkeypatch.setattr(self_update.metadata, 'distribution', distribution)
    monkeypatch.setattr(sys, 'prefix', str(tmp_path))
    assert installed() == Installed(version='0.52.0', commit=commit, tool=False)
    (tmp_path / 'uv-receipt.toml').write_text('[tool]\n')
    assert installed().tool


def test_labels() -> None:
    assert TOOL.label == '4bd401a55'
    assert Installed(version='0.52.0').label == '0.52.0'
    assert Update(channel='bleeding', target=SHA).label == '4bd401a55'
    assert Update(channel='stable', target='0.53.0').label == '0.53.0'


def test_latest_reads_pypi_and_github() -> None:
    seen: list[httpx.URL] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        if str(request.url) == PYPI_URL:
            return httpx.Response(200, json={'info': {'version': '0.53.0', 'summary': 'ignored'}})
        return httpx.Response(200, json=[{'sha': NEWER, 'commit': {}}])

    transport = httpx.MockTransport(respond)
    assert latest('stable', transport=transport) == '0.53.0'
    assert latest('bleeding', transport=transport) == NEWER
    bleeding = seen[1]
    assert str(bleeding.copy_with(query=None)) == COMMITS_URL
    assert dict(bleeding.params) == {'sha': 'main', 'path': 'src/pydantic_clai2', 'per_page': '1'}


def test_latest_reports_failures() -> None:
    with pytest.raises(ValueError, match='no CLAI commits'):
        latest('bleeding', transport=httpx.MockTransport(lambda request: httpx.Response(200, json=[])))
    with pytest.raises(httpx.HTTPStatusError):
        latest('stable', transport=httpx.MockTransport(lambda request: httpx.Response(403)))


def test_find_update() -> None:
    release = Installed(version='0.52.0')
    assert find_update('stable', release, '0.52.0') is None
    assert find_update('stable', release, '0.53.0') == Update(channel='stable', target='0.53.0')
    # A bleeding install switching back to stable is offered the release, even at the same base version.
    assert find_update('stable', Installed(version='0.52.0', commit=SHA), '0.52.0') is not None
    assert find_update('bleeding', TOOL, SHA) is None
    assert find_update('bleeding', release, SHA) == Update(channel='bleeding', target=SHA)


def test_install_commands(tmp_path: Path) -> None:
    stable = Update(channel='stable', target='0.53.0')
    assert stable.command(uv='uv', overrides=None) == ['uv', 'tool', 'install', '--force', 'pydantic-clai2==0.53.0']
    assert stable.environment() == {}
    archive = f'https://github.com/pydantic/pydantic-ai/archive/{NEWER}.tar.gz#subdirectory='
    bleeding = Update(channel='bleeding', target=NEWER)
    overrides = tmp_path / 'overrides.txt'
    assert bleeding.command(uv='/bin/uv', overrides=overrides) == [
        '/bin/uv',
        'tool',
        'install',
        '--force',
        '--overrides',
        str(overrides),
        f'pydantic-clai2 @ {archive}src/pydantic_clai2',
    ]
    assert bleeding.overrides() == (
        f'pydantic-ai-harness @ {archive}src/pydantic_ai_harness\n'
        f'pydantic-ai-slim @ {archive}pydantic_ai_slim\n'
        f'pydantic-graph @ {archive}pydantic_graph\n'
    )
    assert bleeding.environment() == {'UV_DYNAMIC_VERSIONING_BYPASS': '0.0.0+9e08a34ee'}


def _updates(
    channel: list[UpdateChannel],
    *,
    current: Installed = TOOL,
    fetch: Callable[[UpdateChannel], str] = lambda channel: NEWER if channel == 'bleeding' else '0.53.0',
    codes: list[int] | None = None,
    uv: str | None = '/bin/uv',
    executable: str | None = '/tools/clai2',
) -> tuple[Updates, list[Sequence[str]]]:
    ran: list[Sequence[str]] = []

    async def run(command: Sequence[str], environment: dict[str, str]) -> int:
        ran.append(command)
        if '--overrides' in command:
            # The overrides file exists while uv runs and names the other packages.
            text = Path(command[command.index('--overrides') + 1]).read_text(encoding='utf-8')
            assert 'pydantic-ai-harness @ ' in text
            assert environment == {'UV_DYNAMIC_VERSIONING_BYPASS': '0.0.0+9e08a34ee'}
        return (codes or [0]).pop(0)

    async def locate(uv: str) -> str | None:
        assert uv == '/bin/uv'
        return executable

    updates = Updates(
        channel=lambda: channel[0],
        current=current,
        fetch=fetch,
        spawn=lambda work: work(),
        run=run,
        find_uv=lambda: uv,
        windows=False,
        hand_off=lambda script: pytest.fail('only Windows hands the install off'),
        locate=locate,
    )
    return updates, ran


def test_segment_checks_once_per_channel() -> None:
    fetched: list[UpdateChannel] = []

    def fetch(channel: UpdateChannel) -> str:
        fetched.append(channel)
        return NEWER if channel == 'bleeding' else '0.52.1.dev54+4bd401a55'

    channel: list[UpdateChannel] = ['bleeding']
    updates, _ = _updates(channel, current=TOOL, fetch=fetch)
    assert updates.segment() == 'update 9e08a34ee: /update'
    assert updates.segment() == 'update 9e08a34ee: /update'
    channel[0] = 'stable'
    assert updates.segment() == 'update 0.52.1.dev54+4bd401a55: /update'
    assert fetched == ['bleeding', 'stable']


def test_segment_is_quiet_offline_and_outside_uv_tool() -> None:
    def offline(channel: UpdateChannel) -> str:
        raise httpx.ConnectError('offline')

    updates, _ = _updates(['bleeding'], fetch=offline)
    assert updates.segment() == ''
    source, _ = _updates(['bleeding'], current=Installed(version='0.52.0'), fetch=offline)
    assert source.segment() == ''


def test_segment_ignores_a_result_for_another_channel() -> None:
    pending: list[Callable[[], None]] = []
    channel: list[UpdateChannel] = ['bleeding']
    updates = Updates(channel=lambda: channel[0], current=TOOL, fetch=lambda channel: NEWER, spawn=pending.append)
    assert updates.segment() == ''
    channel[0] = 'stable'
    assert updates.segment() == ''
    pending[0]()  # The bleeding check lands after the switch to stable.
    assert updates.segment() == ''


async def test_command_installs_and_restarts() -> None:
    updates, ran = _updates(['bleeding'])
    assert await updates.command([]) == 'Updated CLAI to 9e08a34ee (bleeding). Restarting...'
    assert updates.restart_required
    assert updates.relaunch == '/tools/clai2'
    [command] = ran
    overrides = Path(command[5])
    assert command == Update(channel='bleeding', target=NEWER).command(uv='/bin/uv', overrides=overrides)
    assert not overrides.exists()


async def test_command_reports_current_failure_and_manual_installs() -> None:
    current, ran = _updates(['bleeding'], fetch=lambda channel: SHA)
    assert await current.command([]) == 'CLAI 4bd401a55 is the newest on the bleeding channel.'
    failed, _ = _updates(['stable'], codes=[2])
    assert await failed.command([]) == 'uv exited with status 2; see its output above.'
    assert not failed.restart_required
    no_uv, _ = _updates(['stable'], uv=None)
    assert (await no_uv.command([])).endswith('To update by hand, run:\nuv tool install --force pydantic-clai2==0.53.0')
    # A source checkout installs too, as a `uv tool` CLAI.
    source, installs = _updates(['bleeding'], current=Installed(version='0.52.0'))
    assert await source.command([]) == 'Updated CLAI to 9e08a34ee (bleeding). Restarting...'
    assert len(installs) == 1
    lost, _ = _updates(['stable'], executable=None)
    assert await lost.command([]) == 'Updated CLAI to 0.53.0 (stable). Exiting; start clai2 again to use it.'
    assert lost.restart_required
    assert lost.relaunch is None
    manual, _ = _updates(['bleeding'], uv=None)
    printed = (await manual.command([])).splitlines()[-1]
    assert printed.startswith('UV_DYNAMIC_VERSIONING_BYPASS=0.0.0+9e08a34ee uv tool install --force --overrides ')
    # The printed command still needs its overrides file.
    overrides = Path(shlex.split(printed)[6])
    assert 'pydantic-graph @ ' in overrides.read_text(encoding='utf-8')
    overrides.unlink()
    assert ran == []
    with pytest.raises(ValueError, match='Usage: /update'):
        await current.command(['now'])


async def test_run_uv_passes_the_environment_and_returns_the_exit_status() -> None:
    script = 'import os; raise SystemExit(int(os.environ["CLAI_TEST_CODE"]))'
    assert await _run_uv([sys.executable, '-c', script], {'CLAI_TEST_CODE': '3'}) == 3


async def test_failed_install_removes_the_overrides_file() -> None:
    seen: list[Path] = []

    async def run(command: Sequence[str], environment: dict[str, str]) -> int:
        seen.append(Path(command[5]))
        raise OSError('uv vanished')

    updates, _ = _updates(['bleeding'])
    updates.run = run
    with pytest.raises(OSError):
        await updates.command([])
    assert not seen[0].exists()


def test_find_uv_skips_relative_path_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    planted = tmp_path / 'repo'
    trusted = tmp_path / 'bin'
    for directory in (planted, trusted):
        directory.mkdir()
        for name in ('uv', 'uv.EXE'):
            (directory / name).write_text('')
            (directory / name).chmod(0o755)
    (tmp_path / 'empty').mkdir()
    (tmp_path / 'empty' / 'uv').write_text('')  # Not executable.
    (tmp_path / 'empty' / 'uv').chmod(0o644)
    monkeypatch.chdir(planted)
    path = os.pathsep.join(['', '.', 'repo', str(tmp_path / 'empty'), str(trusted)])
    assert find_uv(environ={'PATH': path}, windows=False) == str(trusted / 'uv')
    windows = {'PATH': path, 'PATHEXT': os.pathsep.join(['.COM', '.EXE'])}
    assert find_uv(environ=windows, windows=True) == str(trusted / 'uv.EXE')
    assert find_uv(environ={'PATH': 'repo'}, windows=False) is None
    assert find_uv(environ={}, windows=False) is None


def test_powershell_quotes_every_part() -> None:
    line = powershell([r'C:\uv\uv.exe', 'tool', "it's"], {'NAME': "o'k"})
    assert line == "$env:NAME = 'o''k'; & 'C:\\uv\\uv.exe' 'tool' 'it''s'"
    assert powershell(['uv'], {}) == "& 'uv'"


def test_after_exit_script(tmp_path: Path) -> None:
    overrides = tmp_path / "o'verrides.txt"
    script = after_exit_script(['uv', 'tool'], {'NAME': 'v'}, pid=42, overrides=overrides)
    lines = script.splitlines()
    assert lines[0] == 'Wait-Process -Id 42 -ErrorAction SilentlyContinue'
    assert lines[2] == "$env:NAME = 'v'; & 'uv' 'tool'"
    assert f"Remove-Item -LiteralPath '{str(overrides).replace(chr(39), chr(39) * 2)}'" in lines
    assert lines[-1] == "Read-Host 'Press Enter to close'"
    assert 'Remove-Item' not in after_exit_script(['uv'], {}, pid=42, overrides=None)


def test_install_after_exit_starts_windows_powershell() -> None:
    started: list[Sequence[str]] = []
    install_after_exit('Write-Host é', environ={'SystemRoot': r'D:\Win'}, start=started.append)
    [argv] = started
    assert argv[:5] == [
        r'D:\Win\System32\WindowsPowerShell\v1.0\powershell.exe',
        '-NoProfile',
        '-ExecutionPolicy',
        'Bypass',
        '-EncodedCommand',
    ]
    assert base64.b64decode(argv[5]).decode('utf-16-le') == 'Write-Host é'
    install_after_exit('', environ={}, start=started.append)
    assert started[1][0] == r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe'


async def test_windows_hands_the_install_off_and_exits() -> None:
    updates, ran = _updates(['bleeding'])
    scripts: list[str] = []
    updates.windows = True
    updates.hand_off = scripts.append
    message = await updates.command([])
    assert message == (
        'Installing CLAI 9e08a34ee (bleeding) in a new window once CLAI exits. '
        'Exiting; start clai2 again when it finishes.'
    )
    assert updates.restart_required
    assert ran == []
    [script] = scripts
    assert f'Wait-Process -Id {os.getpid()} ' in script
    assert "$env:UV_DYNAMIC_VERSIONING_BYPASS = '0.0.0+9e08a34ee'; & '/bin/uv' 'tool' 'install'" in script
    # The overrides file outlives CLAI; the script removes it after uv reads it.
    overrides = Path(script.split("Remove-Item -LiteralPath '")[1].split("'")[0])
    assert 'pydantic-graph @ ' in overrides.read_text(encoding='utf-8')
    overrides.unlink()


async def test_windows_manual_command_is_powershell() -> None:
    updates, _ = _updates(['stable'], uv=None)
    updates.windows = True
    printed = (await updates.command([])).splitlines()[-1]
    assert printed == "& 'uv' 'tool' 'install' '--force' 'pydantic-clai2==0.53.0'"


def test_in_thread_runs_the_work() -> None:
    done = threading.Event()
    _in_thread(done.set)
    assert done.wait(5)


async def test_tool_executable_asks_uv_for_its_bin_directory(tmp_path: Path) -> None:
    uv = tmp_path / 'uv'
    uv.write_text(f'#!/bin/sh\n[ "$*" = "tool dir --bin" ] && echo {tmp_path}\n')
    uv.chmod(0o755)
    assert await tool_executable(str(uv)) is None
    (tmp_path / 'clai2').write_text('')
    assert await tool_executable(str(uv)) == str(tmp_path / 'clai2')
    uv.write_text('#!/bin/sh\nexit 2\n')
    assert await tool_executable(str(uv)) is None


async def _relaunch_after(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, prompts: list[str]) -> Relaunch:
    updates, _ = _updates(['bleeding'])

    def build(*, channel: Callable[[], UpdateChannel]) -> Updates:
        return updates

    monkeypatch.setattr(_app, 'Updates', build)
    inputs(monkeypatch, [*prompts, '/update', '/exit'])
    with pytest.raises(Relaunch) as raised:
        await chat(
            Agent(TestModel()),
            deps=None,
            settings=Settings(model='test', update_channel='bleeding'),
            console=Console(file=StringIO(), width=200),
            store=SettingsStore(tmp_path / 'config.db'),
        )
    assert raised.value.code == 0
    assert raised.value.executable == '/tools/clai2'
    return raised.value


async def test_shell_relaunches_after_an_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert (await _relaunch_after(monkeypatch, tmp_path, [])).session_id is None
    resumed = await _relaunch_after(monkeypatch, tmp_path, ['hello'])
    assert resumed.session_id is not None
    saved = await SqliteConversationStore(database=tmp_path / 'sessions.db').get(conversation_id=resumed.session_id)
    assert saved.messages


def test_relaunch_argv(tmp_path: Path) -> None:
    bare = argparse.Namespace(agent=None, model=None, request_limit=None, database=None)
    assert relaunch_argv(bare, executable='/b/clai2', session_id=None) == ['/b/clai2']
    full = argparse.Namespace(agent='m:a', model='test', request_limit=5, database=tmp_path / 'c.db')
    assert relaunch_argv(full, executable='/b/clai2', session_id='abc') == [
        '/b/clai2',
        '--agent',
        'm:a',
        '--model',
        'test',
        '--request-limit',
        '5',
        '--database',
        str(tmp_path / 'c.db'),
        '--resume',
        'abc',
    ]


def test_cli_execs_the_new_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('sys.argv', ['clai2', '--database', 'config.db', '--resume', 'old', '-m', 'test'])

    async def chat(*args: object, **kwargs: object) -> None:
        raise Relaunch(executable='/b/clai2', session_id='new')

    execs: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(_app, 'chat', chat)

    def execv(path: str, argv: list[str]) -> None:
        execs.append((path, argv))

    monkeypatch.setattr(os, 'execv', execv)
    _cli.run()
    assert execs == [
        ('/b/clai2', ['/b/clai2', '--model', 'test', '--database', str(tmp_path / 'config.db'), '--resume', 'new'])
    ]


async def test_shell_exits_after_an_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    updates, ran = _updates(['stable'], executable=None)
    channels: list[UpdateChannel] = []

    def build(*, channel: Callable[[], UpdateChannel]) -> Updates:
        channels.append(channel())
        return updates

    monkeypatch.setattr(_app, 'Updates', build)
    inputs(monkeypatch, ['/update', '/exit'])
    output = StringIO()
    await chat(
        Agent(TestModel()),
        deps=None,
        settings=Settings(model='test', update_channel='bleeding'),
        console=Console(file=output, width=200),
        store=SettingsStore(tmp_path / 'config.db'),
    )
    text = output.getvalue()
    assert channels == ['bleeding']
    assert 'Updated CLAI to 0.53.0 (stable). Exiting; start clai2 again to use it.' in text
    assert 'Goodbye.' not in text
    assert len(ran) == 1
