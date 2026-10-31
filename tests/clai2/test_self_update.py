"""`/update`: release lookup, install commands, the footer notice, and the shell exit after an install."""

import argparse
import base64
import json
import os
import shlex
import sys
import threading
from collections.abc import Awaitable, Callable, Sequence
from importlib import metadata
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from io import StringIO
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from packaging.requirements import Requirement
from rich.console import Console
from termflow.tui.menu import MenuResult

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from pydantic_clai2 import app, chat, cli
from pydantic_clai2.cli import cli as cli_module, self_update
from pydantic_clai2.cli.cli import relaunch_argv
from pydantic_clai2.cli.self_update import (
    BLEEDING_URL,
    PACKAGES,
    PYPI_URL,
    Installed,
    Output,
    Relaunch,
    Update,
    UpdateCheck,
    Updates,
    WebHandoff,
    _in_thread,  # pyright: ignore[reportPrivateUsage]
    _run_uv,  # pyright: ignore[reportPrivateUsage]
    after_exit_script,
    bleeding_url,
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
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import InstallProgress
from pydantic_clai2.plugins.menus import CloseMenu
from pydantic_clai2.runtime import session_runtime
from pydantic_clai2.ui.menus import dialogs, update_menu as update_menu_module
from pydantic_clai2.ui.menus.field_menu import FieldMenu
from pydantic_clai2.ui.menus.set_menu import SettingsSource
from pydantic_clai2.ui.menus.update_menu import VERSION, AboutSource, update_menu
from tests.clai2.dialog_script import Back, Confirm, ScriptedDialogs
from tests.clai2.menu_script import Script, make_context, pick
from tests.clai2.test_app_edges import inputs

SHA = '4bd401a5539aa96b9b67753092fd7f8b90e4dc8a'
NEWER = '9e08a34ee0000000000000000000000000000000'
TOOL = Installed(version='0.52.1.dev54+4bd401a55', commit=SHA, tool=True)


def test_channel_setting(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    assert context.settings.update_channel == 'stable'
    assert set_completions(['updates.channel', '']) == ('stable', 'main')
    row = next(row for row in FieldMenu(SettingsSource(context)).rows if row.key == 'updates.channel')
    assert row.choices == ('stable', 'main')
    assert context.set_setting(['updates.channel', 'main']) == 'Saved updates.channel. Applied.'
    assert SettingsStore(context.store.path).load().update_channel == 'main'
    # Saved under its former name, which older builds still read.
    assert context.store.overrides()['updates.channel'] == 'bleeding'
    context.set_setting(['updates.channel', 'stable'])
    assert context.set_setting(['updates.channel', 'bleeding']) == 'Saved updates.channel. Applied.'
    assert context.settings.update_channel == 'main'
    assert SettingsStore(context.store.path).load().update_channel == 'main'
    with pytest.raises(ValueError, match='stable'):
        context.set_setting(['updates.channel', 'nightly'])
    assert SettingsStore(context.store.path).load().update_channel == 'main'


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
        (json.dumps({'url': f'{BLEEDING_URL}/pydantic_clai2-{SHA}.tar.gz', 'archive_info': {}}), SHA),
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
    assert Update(channel='main', target=SHA).label == '4bd401a55'
    assert Update(channel='stable', target='0.53.0').label == '0.53.0'


def test_latest_reads_pypi_and_the_bleeding_release(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if str(request.url) == PYPI_URL:
            return httpx.Response(200, json={'info': {'version': '0.53.0', 'summary': 'ignored'}})
        return httpx.Response(200, json={'commit': NEWER})

    transport = httpx.MockTransport(respond)
    monkeypatch.delenv('CLAI_BLEEDING_URL', raising=False)
    assert latest('stable', transport=transport) == '0.53.0'
    assert latest('main', transport=transport) == NEWER
    monkeypatch.setenv('CLAI_BLEEDING_URL', 'http://localhost:8000/')
    assert latest('main', transport=transport) == NEWER
    # A release download, not a GitHub API request.
    assert seen[1:] == [
        'https://github.com/pydantic/pydantic-ai/releases/download/clai2-bleeding/clai2-bleeding.json',
        'http://localhost:8000/clai2-bleeding.json',
    ]


def test_bleeding_url() -> None:
    assert bleeding_url(environ={}) == BLEEDING_URL
    assert bleeding_url(environ={'CLAI_BLEEDING_URL': ''}) == BLEEDING_URL
    fork = 'https://github.com/me/pydantic-ai/releases/download/clai2-bleeding'
    assert bleeding_url(environ={'CLAI_BLEEDING_URL': f'{fork}/'}) == fork


def test_latest_reports_failures() -> None:
    with pytest.raises(ValueError, match='commit'):
        latest('main', transport=httpx.MockTransport(lambda request: httpx.Response(200, json={'commit': 'main'})))
    with pytest.raises(httpx.HTTPStatusError):
        latest('main', transport=httpx.MockTransport(lambda request: httpx.Response(404)))
    with pytest.raises(httpx.HTTPStatusError):
        latest('stable', transport=httpx.MockTransport(lambda request: httpx.Response(403)))


def test_find_update() -> None:
    release = Installed(version='0.52.0')
    assert find_update('stable', release, '0.52.0') is None
    assert find_update('stable', release, '0.53.0') == Update(channel='stable', target='0.53.0')
    # A `main` install switching back to stable is offered the release, even at the same base version.
    assert find_update('stable', Installed(version='0.52.0', commit=SHA), '0.52.0') is not None
    assert find_update('main', TOOL, SHA) is None
    assert find_update('main', release, SHA) == Update(channel='main', target=SHA)


def test_install_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stable = Update(channel='stable', target='0.53.0')
    # Compiling bytecode at install time spares the restart after `/update` from compiling it on import.
    assert stable.command(uv='uv', overrides=None) == [
        'uv',
        'tool',
        'install',
        '--force',
        '--compile-bytecode',
        'pydantic-clai2==0.53.0',
    ]
    monkeypatch.delenv('CLAI_BLEEDING_URL', raising=False)
    release = 'https://github.com/pydantic/pydantic-ai/releases/download/clai2-bleeding'
    update = Update(channel='main', target=NEWER)
    overrides = tmp_path / 'overrides.txt'
    assert update.command(uv='/bin/uv', overrides=overrides) == [
        '/bin/uv',
        'tool',
        'install',
        '--force',
        '--compile-bytecode',
        '--overrides',
        str(overrides),
        'pydantic-clai2',
    ]
    assert update.overrides() == (
        f'pydantic-clai2 @ {release}/pydantic_clai2-{NEWER}.tar.gz\n'
        f'pydantic-ai-harness[coder,skills] @ {release}/pydantic_ai_harness-{NEWER}.tar.gz\n'
        f'pydantic-ai-slim[anthropic,mcp,openai] @ {release}/pydantic_ai_slim-{NEWER}.tar.gz\n'
        f'pydantic-graph @ {release}/pydantic_graph-{NEWER}.tar.gz\n'
    )
    monkeypatch.setenv('CLAI_BLEEDING_URL', 'http://localhost:8000')
    assert (
        update.requirement('pydantic-graph') == f'pydantic-graph @ http://localhost:8000/pydantic_graph-{NEWER}.tar.gz'
    )


def test_build_script_builds_every_package() -> None:
    script = Path(__file__).parents[2] / 'src' / 'pydantic_clai2' / 'scripts' / 'build_bleeding.sh'
    [line] = [line for line in script.read_text(encoding='utf-8').splitlines() if line.startswith('for package in ')]
    assert line.removeprefix('for package in ').removesuffix('; do').split() == [
        requirement.partition('[')[0] for requirement in PACKAGES
    ]


def test_archive_overrides_preserve_declared_extras() -> None:
    overrides = {
        requirement.name: requirement.extras
        for requirement in map(Requirement, Update(channel='main', target=NEWER).overrides().splitlines())
    }
    # Graph is a transitive dependency, so read each overridden package's requirements.
    declared_extras: dict[str, set[str]] = {}
    for name in overrides:
        for requirement in map(Requirement, metadata.requires(name) or []):
            if requirement.name in overrides and (requirement.marker is None or requirement.marker.evaluate()):
                declared_extras.setdefault(requirement.name, set()).update(requirement.extras)
    assert declared_extras == {name: extras for name, extras in overrides.items() if name != 'pydantic-clai2'}


def test_skills_dependency_survives_overrides_from_older_builds() -> None:
    """An older build's `/update` overrides the harness without its `skills` extra, so CLAI requires PyYAML itself."""

    def pyyaml(name: str) -> list[Requirement]:
        return [
            requirement
            for requirement in map(Requirement, metadata.requires(name) or [])
            if requirement.name == 'pyyaml'
        ]

    [direct] = pyyaml('pydantic-clai2')
    assert direct.marker is None
    skills = [
        requirement.specifier
        for requirement in pyyaml('pydantic-ai-harness')
        if requirement.marker and requirement.marker.evaluate({'extra': 'skills'})
    ]
    assert skills == [direct.specifier]


def _updates(
    channel: list[UpdateChannel],
    *,
    current: Installed = TOOL,
    fetch: Callable[[UpdateChannel], str] = lambda channel: NEWER if channel == 'main' else '0.53.0',
    codes: list[int] | None = None,
    uv: str | None = '/bin/uv',
    executable: str | None = '/tools/clai2',
    lines: Sequence[str] = (),
) -> tuple[Updates, list[Sequence[str]]]:
    ran: list[Sequence[str]] = []

    async def run(command: Sequence[str], output: Output | None) -> int:
        ran.append(command)
        for line in lines:
            assert output is not None, 'uv prints to the terminal while nobody listens'
            await output(line)
        if '--overrides' in command:
            # The overrides file exists while uv runs and preserves the packages' required extras.
            text = Path(command[command.index('--overrides') + 1]).read_text(encoding='utf-8')
            assert 'pydantic-ai-harness[coder,skills] @ ' in text
            assert 'pydantic-ai-slim[anthropic,mcp,openai] @ ' in text
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
        return NEWER if channel == 'main' else '0.52.1.dev54+4bd401a55'

    channel: list[UpdateChannel] = ['main']
    updates, _ = _updates(channel, current=TOOL, fetch=fetch)
    assert updates.segment() == 'update 9e08a34ee: /update'
    assert updates.segment() == 'update 9e08a34ee: /update'
    channel[0] = 'stable'
    assert updates.segment() == 'update 0.52.1.dev54+4bd401a55: /update'
    assert fetched == ['main', 'stable']


def test_segment_is_quiet_offline_and_outside_uv_tool() -> None:
    def offline(channel: UpdateChannel) -> str:
        raise httpx.ConnectError('offline')

    updates, _ = _updates(['main'], fetch=offline)
    assert updates.segment() == ''
    source, _ = _updates(['main'], current=Installed(version='0.52.0'), fetch=offline)
    assert source.segment() == ''


def test_segment_ignores_a_result_for_another_channel() -> None:
    pending: list[Callable[[], None]] = []
    channel: list[UpdateChannel] = ['main']
    updates = Updates(channel=lambda: channel[0], current=TOOL, fetch=lambda channel: NEWER, spawn=pending.append)
    assert updates.segment() == ''
    channel[0] = 'stable'
    assert updates.segment() == ''
    pending[0]()  # The `main` check lands after the switch to stable.
    assert updates.segment() == ''


async def test_command_installs_and_restarts() -> None:
    updates, ran = _updates(['main'])
    assert await updates.command([]) == 'Updated CLAI to 9e08a34ee (main). Restarting...'
    assert updates.restart_required
    assert updates.relaunch == '/tools/clai2'
    [command] = ran
    overrides = Path(command[command.index('--overrides') + 1])
    assert command == Update(channel='main', target=NEWER).command(uv='/bin/uv', overrides=overrides)
    assert not overrides.exists()


async def test_command_reports_current_failure_and_manual_installs() -> None:
    current, ran = _updates(['main'], fetch=lambda channel: SHA)
    assert await current.command([]) == 'CLAI 4bd401a55 is the newest on the main channel.'
    failed, _ = _updates(['stable'], codes=[2])
    assert await failed.command([]) == 'uv exited with status 2; see its output above.'
    assert not failed.restart_required
    no_uv, _ = _updates(['stable'], uv=None)
    assert (await no_uv.command([])).endswith(
        'To update by hand, run:\nuv tool install --force --compile-bytecode pydantic-clai2==0.53.0'
    )
    # A source checkout installs too, as a `uv tool` CLAI.
    source, installs = _updates(['main'], current=Installed(version='0.52.0'))
    assert await source.command([]) == 'Updated CLAI to 9e08a34ee (main). Restarting...'
    assert len(installs) == 1
    lost, _ = _updates(['stable'], executable=None)
    assert await lost.command([]) == 'Updated CLAI to 0.53.0 (stable). Exiting; start clai2 again to use it.'
    assert lost.restart_required
    assert lost.relaunch is None
    manual, _ = _updates(['main'], uv=None)
    printed = (await manual.command([])).splitlines()[-1]
    assert printed.startswith('uv tool install --force --compile-bytecode --overrides ')
    assert printed.endswith(' pydantic-clai2')
    # The printed command still needs its overrides file.
    argv = shlex.split(printed)
    overrides = Path(argv[argv.index('--overrides') + 1])
    assert 'pydantic-graph @ ' in overrides.read_text(encoding='utf-8')
    overrides.unlink()
    assert ran == []
    assert (current.failed, failed.failed, no_uv.failed, manual.failed, lost.failed) == (False, True, True, True, False)
    assert await current.command(['now']) == 'CLAI 4bd401a55 is the newest on the main channel.'
    with pytest.raises(ValueError, match='Usage: /update opens the update menu; /update now installs'):
        await current.command(['later'])


@pytest.mark.subprocess(
    reason='runs Python one-liners in place of `uv` to check its exit status and lines are returned'
)
async def test_run_uv_returns_the_exit_status(
    capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert await _run_uv([sys.executable, '-c', 'raise SystemExit(3)']) == 3
    lines: list[str] = []

    async def output(line: str) -> None:
        lines.append(line)

    script = "import sys; print('Resolved'); sys.stderr.write('Prepared\\r\\nfailed'); raise SystemExit(2)"
    assert await _run_uv([sys.executable, '-u', '-c', script], output) == 2
    assert lines == ['Resolved', 'Prepared', 'failed']
    assert capfd.readouterr().err == 'Resolved\nPrepared\nfailed\n', 'the terminal still shows them'
    # While the shell relays standard error, the lines go straight to the terminal, before the editor is back.
    source, terminal = os.pipe()
    monkeypatch.setattr(self_update, 'terminal_stderr', lambda: terminal)
    assert await _run_uv([sys.executable, '-c', "print('Installed')"], output) == 0
    os.close(terminal)
    with os.fdopen(source) as relayed:
        assert relayed.read() == 'Installed\n'
    assert lines[-1] == 'Installed'


async def test_command_reports_its_progress() -> None:
    """While a plugin listens, it hears the install start, each line uv prints, and how the install ended."""
    heard: list[InstallProgress] = []

    async def report(event: InstallProgress) -> None:
        heard.append(event)

    restarted, _ = _updates(['stable'], lines=['Resolved 3 packages', 'Installed 3 packages'])
    restarted.report = lambda: report
    restarted.capture = lambda: True
    served = WebHandoff(port=4321, token=WEB_TOKEN)
    restarted.serving = lambda: served
    assert await restarted.command([]) == 'Updated CLAI to 0.53.0 (stable). Restarting...'
    assert restarted.web == served, 'the new build serves the page again'
    assert heard == [
        InstallProgress(stage='installing', target='0.53.0'),
        InstallProgress(stage='output', target='0.53.0', text='Resolved 3 packages'),
        InstallProgress(stage='output', target='0.53.0', text='Installed 3 packages'),
        InstallProgress(stage='restarting', target='0.53.0'),
    ]
    heard.clear()
    # Without a page to show them, uv's lines stay on the terminal, where uv draws its progress itself.
    quiet, _ = _updates(['stable'])
    quiet.report = lambda: report
    await quiet.command([])
    assert [event.stage for event in heard] == ['installing', 'restarting']
    heard.clear()
    lost, _ = _updates(['stable'], executable=None)
    failed, _ = _updates(['stable'], codes=[1])
    vanished, _ = _updates(['main'])

    async def missing(command: Sequence[str], output: Output | None) -> int:
        raise OSError('uv vanished')

    vanished.run = missing
    for updates in (lost, failed, vanished):
        updates.report = lambda: report
        updates.serving = lambda: served
    await lost.command([])
    await failed.command([])
    with pytest.raises(OSError):
        await vanished.command([])
    assert (lost.web, failed.web) == (None, None), 'nothing restarts to serve it'
    assert [event.stage for event in heard] == [
        'installing',
        'installed',
        'installing',
        'failed',
        'installing',
        'failed',
    ]


async def test_failed_install_removes_the_overrides_file() -> None:
    seen: list[Path] = []

    async def run(command: Sequence[str], output: Output | None) -> int:
        seen.append(Path(command[command.index('--overrides') + 1]))
        raise OSError('uv vanished')

    updates, _ = _updates(['main'])
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
    line = powershell([r'C:\uv\uv.exe', 'tool', "it's"])
    assert line == "& 'C:\\uv\\uv.exe' 'tool' 'it''s'"
    assert powershell(['uv']) == "& 'uv'"


def test_after_exit_script(tmp_path: Path) -> None:
    overrides = tmp_path / "o'verrides.txt"
    script = after_exit_script(['uv', 'tool'], pid=42, overrides=overrides)
    lines = script.splitlines()
    assert lines[0] == 'Wait-Process -Id 42 -ErrorAction SilentlyContinue'
    assert lines[2] == "& 'uv' 'tool'"
    assert f"Remove-Item -LiteralPath '{str(overrides).replace(chr(39), chr(39) * 2)}'" in lines
    assert lines[-1] == "Read-Host 'Press Enter to close'"
    assert 'Remove-Item' not in after_exit_script(['uv'], pid=42, overrides=None)


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
    updates, ran = _updates(['main'])
    scripts: list[str] = []
    updates.windows = True
    updates.hand_off = scripts.append
    message = await updates.command([])
    assert message == (
        'Installing CLAI 9e08a34ee (main) in a new window once CLAI exits. Exiting; start clai2 again when it finishes.'
    )
    assert updates.restart_required
    assert ran == []
    [script] = scripts
    assert f'Wait-Process -Id {os.getpid()} ' in script
    assert "& '/bin/uv' 'tool' 'install' '--force' '--compile-bytecode' '--overrides' " in script
    # The overrides file outlives CLAI; the script removes it after uv reads it.
    overrides = Path(script.split("Remove-Item -LiteralPath '")[1].split("'")[0])
    assert overrides.read_text(encoding='utf-8') == Update(channel='main', target=NEWER).overrides()
    assert script.splitlines()[2].endswith(" 'pydantic-clai2'")
    overrides.unlink()


@pytest.mark.parametrize('channel', ['stable', 'main'])
async def test_windows_manual_command_is_powershell(channel: UpdateChannel) -> None:
    updates, _ = _updates([channel], uv=None)
    updates.windows = True
    printed = (await updates.command([])).splitlines()[-1]
    if channel == 'stable':
        assert printed == "& 'uv' 'tool' 'install' '--force' '--compile-bytecode' 'pydantic-clai2==0.53.0'"
    else:
        assert printed.startswith("& 'uv' 'tool' 'install' '--force' '--compile-bytecode' '--overrides' ")
        assert printed.endswith(" 'pydantic-clai2'")
        overrides = Path(printed.split("'--overrides' '")[1].split("'")[0])
        assert overrides.read_text(encoding='utf-8') == Update(channel='main', target=NEWER).overrides()
        overrides.unlink()


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


async def _relaunch_after(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    prompts: list[str],
    *,
    session_id: str | None = None,
    updates: Updates | None = None,
) -> Relaunch:
    if updates is None:
        updates, _ = _updates(['main'])

    def build(*, channel: Callable[[], UpdateChannel]) -> Updates:
        return updates

    monkeypatch.setattr(session_runtime, 'Updates', build)
    inputs(monkeypatch, [*prompts, '/update now', '/exit'])
    with pytest.raises(Relaunch) as raised:
        await chat(
            Agent(TestModel()),
            deps=None,
            settings=Settings(model='test', update_channel='main'),
            console=Console(file=StringIO(), width=200),
            store=SettingsStore(tmp_path / 'config.db'),
            session_id=session_id,
        )
    assert raised.value.code == 0
    assert raised.value.executable == '/tools/clai2'
    return raised.value


async def test_shell_relaunches_after_an_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fresh = await _relaunch_after(monkeypatch, tmp_path, [])
    assert (fresh.session_id, fresh.new_session_id) == (None, None)
    named = await _relaunch_after(monkeypatch, tmp_path, [], session_id='0b4f5d8e-3c1a-4e6b-9f2d-7a8c9b0d1e2f')
    assert (named.session_id, named.new_session_id) == (None, '0b4f5d8e-3c1a-4e6b-9f2d-7a8c9b0d1e2f')

    def rebuild(factory: Callable[[], object]) -> object:
        return factory()

    # `/reload` rebuilds the shell, which keeps the conversation the launch options chose.
    monkeypatch.setattr(app, 'reload_clai', rebuild)
    reloaded = await _relaunch_after(
        monkeypatch, tmp_path, ['/reload'], session_id='0b4f5d8e-3c1a-4e6b-9f2d-7a8c9b0d1e2f'
    )
    assert (reloaded.session_id, reloaded.new_session_id) == (None, '0b4f5d8e-3c1a-4e6b-9f2d-7a8c9b0d1e2f')
    resumed = await _relaunch_after(monkeypatch, tmp_path, ['hello'])
    assert resumed.session_id is not None
    saved = await SqliteConversationStore(database=tmp_path / 'sessions.db').get(conversation_id=resumed.session_id)
    assert saved.messages


class _NewBuild(MetaPathFinder):
    """The code uv installed: once it has run, a CLAI module imported for the first time is the new build's."""

    def __init__(self) -> None:
        self.installed = False

    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        if self.installed and fullname.partition('.')[0] in {'pydantic_ai', 'pydantic_ai_harness', 'pydantic_clai2'}:
            raise ImportError(f'{fullname} would come from the new build')  # pragma: no cover -- the defect
        return None


async def test_restart_imports_nothing_after_the_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The new build's modules may need names the old build's, already in memory, lack: restarting imports none."""
    new_build = _NewBuild()
    monkeypatch.setattr(sys, 'meta_path', [new_build, *sys.meta_path])
    # A session that never served the web UI has not imported its module.
    monkeypatch.delitem(sys.modules, 'pydantic_clai2.cli.web', raising=False)
    monkeypatch.delattr(cli, 'web', raising=False)
    updates, _ = _updates(['main'])
    install = updates.run

    async def run(command: Sequence[str], output: Output | None) -> int:
        code = await install(command, output)
        new_build.installed = True
        return code

    updates.run = run
    relaunched = await _relaunch_after(monkeypatch, tmp_path, [], updates=updates)
    assert new_build.installed
    assert relaunched.web is None


def _args(**options: object) -> argparse.Namespace:
    defaults: dict[str, object] = dict(
        agent=None, model=None, request_limit=None, database=None, web=False, port=None, token_file=None
    )
    return argparse.Namespace(**(defaults | options))


def test_relaunch_argv(tmp_path: Path) -> None:
    bare = _args()
    assert relaunch_argv(bare, executable='/b/clai2', session_id=None) == ['/b/clai2']
    assert relaunch_argv(bare, executable='/b/clai2', session_id=None, new_session_id='u') == [
        '/b/clai2',
        '--session-id',
        'u',
    ]
    assert relaunch_argv(bare, executable='/b/clai2', session_id='s', new_session_id='u') == [
        '/b/clai2',
        '--resume',
        's',
    ]
    # `--web` stays a web launch; the token stays in its file, off the command line.
    web = _args(web=True)
    assert relaunch_argv(web, executable='/b/clai2', session_id='s') == ['/b/clai2', '--web', '--resume', 's']
    launcher = _args(web=True, port=0, token_file=tmp_path / 'token')
    assert relaunch_argv(launcher, executable='/b/clai2', session_id=None) == [
        '/b/clai2',
        '--web',
        '--port',
        '0',
        '--token-file',
        str(tmp_path / 'token'),
    ]
    full = _args(agent='m:a', model='test', request_limit=5, database=tmp_path / 'c.db')
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
    monkeypatch.setattr(app, 'chat', chat)

    def execv(path: str, argv: list[str]) -> None:
        execs.append((path, argv))

    monkeypatch.setattr(os, 'execv', execv)
    cli_module.run()
    assert execs == [
        ('/b/clai2', ['/b/clai2', '--model', 'test', '--database', str(tmp_path / 'config.db'), '--resume', 'new'])
    ]
    assert 'CLAI_RELAUNCH_WEB' not in os.environ, 'no web UI to hand over'


WEB_TOKEN = 'k' * 43


def test_cli_hands_the_web_ui_to_the_new_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`clai2 --web` restarts as `clai2 --web`, with its port and token in the environment, never in `ps`."""
    token = tmp_path / 'token'
    token.write_text(WEB_TOKEN)
    token.chmod(0o600)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('sys.argv', ['clai2', '--web', '--port', '0', '--token-file', 'token', '--database', 'c.db'])
    calls: list[dict[str, object]] = []

    async def run_web(**kwargs: object) -> None:
        calls.append(kwargs)
        raise Relaunch(executable='/b/clai2', session_id='s', web=WebHandoff(port=4321, token=WEB_TOKEN))

    execs: list[list[str]] = []

    def execv(path: str, argv: list[str]) -> None:
        execs.append(argv)

    monkeypatch.setattr('pydantic_clai2.cli.web.run_web', run_web)
    monkeypatch.setattr(os, 'execv', execv)
    monkeypatch.setenv('CLAI_RELAUNCH_WEB', f'4321 {WEB_TOKEN}')
    cli_module.run()
    [call] = calls
    assert (call['port'], call['token'], call['restart']) == (0, WEB_TOKEN, WebHandoff(port=4321, token=WEB_TOKEN))
    [argv] = execs
    database = str(tmp_path / 'c.db')
    assert argv == [
        '/b/clai2',
        '--web',
        '--port',
        '0',
        '--token-file',
        str(token),
        '--database',
        database,
        '--resume',
        's',
    ]
    assert WEB_TOKEN not in ' '.join(argv)
    assert os.environ['CLAI_RELAUNCH_WEB'] == f'4321 {WEB_TOKEN}'


@pytest.mark.parametrize(
    ('value', 'handoff'),
    [
        (f'4321 {WEB_TOKEN}', WebHandoff(port=4321, token=WEB_TOKEN)),
        (f'99999 {WEB_TOKEN}', None),
        (f'port {WEB_TOKEN}', None),
        ('4321 short', None),
        ('', None),
    ],
)
def test_a_terminal_session_serves_the_handed_over_web_ui(
    value: str, handoff: WebHandoff | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The handed-over port and token reach the shell, and leave the environment before any tool could read them."""
    monkeypatch.setattr('sys.argv', ['clai2', '--database', str(tmp_path / 'c.db'), '-m', 'test'])
    monkeypatch.setenv('CLAI_RELAUNCH_WEB', value)
    calls: list[dict[str, object]] = []

    async def chat(*args: object, **kwargs: object) -> None:
        calls.append(kwargs)
        assert 'CLAI_RELAUNCH_WEB' not in os.environ

    monkeypatch.setattr(app, 'chat', chat)
    cli_module.run()
    assert calls[0]['web'] == handoff


async def test_shell_exits_after_an_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    updates, ran = _updates(['stable'], executable=None)
    channels: list[UpdateChannel] = []

    def build(*, channel: Callable[[], UpdateChannel]) -> Updates:
        channels.append(channel())
        return updates

    monkeypatch.setattr(session_runtime, 'Updates', build)
    inputs(monkeypatch, ['/update now', '/exit'])
    output = StringIO()
    await chat(
        Agent(TestModel()),
        deps=None,
        settings=Settings(model='test', update_channel='main'),
        console=Console(file=output, width=200),
        store=SettingsStore(tmp_path / 'config.db'),
    )
    text = output.getvalue()
    assert channels == ['main']
    assert 'Updated CLAI to 0.53.0 (stable). Exiting; start clai2 again to use it.' in text
    assert 'Goodbye.' not in text
    assert len(ran) == 1


async def test_the_about_menu_keeps_the_version_read_only_and_updates_only_when_confirmed(tmp_path: Path) -> None:
    """`/update`'s menu, which bare `/update` opens, in the terminal and in the web UI as Settings > About."""
    context, _ = make_context(tmp_path)
    entered: list[str] = []
    updates, _ = _updates(['stable'], current=Installed(version='1.2.3', commit='a' * 40))
    menu = update_menu(context, updates, run=entered.append)
    source = menu.source
    version, channel = source.rows()
    assert (version.key, source.current(version), version.default) == (
        VERSION,
        f'1.2.3 ({"a" * 9})',
        source.current(version),
    )
    assert source.problem(version, '2.0') == 'The version is read-only; Update installs a newer one.'
    with pytest.raises(ValueError, match='read-only'):
        source.apply(version, '2.0')
    assert source.reset(version) == 'The version has nothing to reset.'
    assert source.problem(channel, 'nightly') is not None
    assert source.apply(channel, 'main').startswith('Saved updates.channel')
    assert source.current(channel) == 'main'
    assert source.reset(channel).startswith('Reset updates.channel')
    plain, _ = _updates(['stable'], current=Installed(version='1.2.3'))
    plain_menu = update_menu(context, plain, run=entered.append)
    assert plain_menu.source.current(plain_menu.source.rows()[0]) == '1.2.3'

    update = menu.actions[VERSION]
    assert await update(ScriptedDialogs([Back()])) == [] and entered == []
    asked = ScriptedDialogs([Confirm(yes=True)])
    closing = await update(asked)
    assert isinstance(closing, CloseMenu) and closing == ('Running /update now on the stable channel.',)
    assert entered == ['/update now'], 'entered as typing it would, so it queues behind a running turn'
    assert 'stable channel' in asked.asked[0].description
    refusals = iter([None, 'A session started a turn.'])
    racing = update_menu(context, updates, run=entered.append, refusal=lambda: next(refusals))
    assert await racing.actions[VERSION](ScriptedDialogs([Confirm(yes=True)])) == ['A session started a turn.']
    assert entered == ['/update now'], 'refused after the dialog: nothing entered'


def test_the_about_menu_says_whether_a_newer_clai_is_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The version row's note answers the update check, which the menu starts even for a source checkout."""
    context, _ = make_context(tmp_path)

    def note(updates: Updates) -> str:
        return AboutSource(context, updates).rows()[0].note

    def offline(channel: UpdateChannel) -> str:
        raise httpx.ConnectError('offline')

    newest, _ = _updates(['stable'], current=Installed(version='0.53.0', tool=True))
    assert note(newest) == 'clai2 is already on the latest version 0.53.0 on the stable channel.'
    newer, _ = _updates(['main'])
    assert note(newer) == 'a new version 9e08a34ee of clai2 is available on the main channel.'
    source, _ = _updates(['stable'], current=Installed(version='0.52.0'))
    assert note(source) == 'a new version 0.53.0 of clai2 is available on the stable channel.'
    assert source.segment() == '', 'the footer still checks only `uv tool` installs'
    failing, _ = _updates(['stable'], fetch=offline)
    assert note(failing) == 'Could not check for a newer CLAI: offline'
    assert failing.segment() == ''
    # The menu cannot redraw when the answer comes, so it waits a moment for it before it shows.
    monkeypatch.setattr(update_menu_module, 'CHECK_WAIT', 0.0)
    checks: list[Callable[[], None]] = []
    pending = Updates(channel=lambda: 'stable', current=TOOL, fetch=lambda channel: '0.53.0', spawn=checks.append)
    assert note(pending) == 'Checking for a newer CLAI…'
    assert note(pending) == 'Checking for a newer CLAI…' and len(checks) == 1, 'one check per channel'
    checks[0]()
    assert note(pending) == 'a new version 0.53.0 of clai2 is available on the stable channel.'


def test_check_waits_for_the_answer_until_its_deadline() -> None:
    answer = threading.Event()

    def fetch(channel: UpdateChannel) -> str:
        answer.wait(10)
        return '0.53.0'

    updates = Updates(channel=lambda: 'stable', current=TOOL, fetch=fetch)
    assert updates.check(wait=0.01) is None, 'the deadline passed while the check runs'
    answer.set()
    assert updates.check(wait=10) == UpdateCheck(channel='stable', update=Update(channel='stable', target='0.53.0'))


async def test_bare_update_without_a_terminal_points_to_the_web_menu(tmp_path: Path) -> None:
    """A session with no terminal cannot open a terminal menu; its web UI handles bare `/update` separately."""
    shell = session_runtime.create_shell(
        Agent(TestModel()),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=StringIO()),
        settings=None,
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )
    with pytest.raises(ValueError, match='update menu needs a terminal; open About in the web UI'):
        await shell.commands.execute_async('/update')
    await shell.close('exit')


@pytest.mark.parametrize('confirm', [True, False])
async def test_bare_update_opens_the_menu_and_installs_once_it_closes(
    confirm: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In the terminal, bare `/update` shows the menu; its Update action closes it and installs at once."""
    updates, ran = _updates(['stable'], executable=None)

    def build(*, channel: Callable[[], UpdateChannel]) -> Updates:
        return updates

    monkeypatch.setattr(session_runtime, 'Updates', build)
    shown = Script(lists=[pick(VERSION), MenuResult(cancelled=True)], choices=[pick(confirm)], texts=[])
    monkeypatch.setattr(dialogs, 'RUNNERS', shown.runners)
    inputs(monkeypatch, ['/update', '/exit'])
    output = StringIO()
    await chat(
        Agent(TestModel()),
        deps=None,
        settings=Settings(model='test', update_channel='stable'),
        console=Console(file=output, width=200),
        store=SettingsStore(tmp_path / 'config.db'),
    )
    text = output.getvalue()
    if confirm:
        # No second list: the action closed the menu, and the install followed without `/update now` typed.
        assert shown.opened == ['list', 'choice']
        assert text.index('Running /update now on the stable channel.') < text.index('Updated CLAI to 0.53.0')
        assert len(ran) == 1
    else:
        assert shown.opened == ['list', 'choice', 'list'], 'Cancel returns to the menu'
        assert 'Updated CLAI' not in text
        assert ran == []


@pytest.mark.parametrize(
    ('code', 'exit_code', 'printed'),
    [
        (0, 0, 'Updated CLAI to 9e08a34ee (main). Exiting; start clai2 again to use it.'),
        (2, 1, 'uv exited with status 2; see its output above.'),
    ],
)
def test_update_flag_installs_and_exits_without_restarting(
    code: int,
    exit_code: int,
    printed: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`clai2 --update` installs from the saved channel, prints how it went, and exits: no shell, no relaunch."""
    context, _ = make_context(tmp_path)
    context.set_setting(['updates.channel', 'main'])
    ran: list[list[Sequence[str]]] = []

    def build(*, channel: Callable[[], UpdateChannel], locate: Callable[[str], Awaitable[str | None]]) -> Updates:
        updates, installs = _updates([channel()], codes=[code])
        updates.locate = locate
        ran.append(installs)
        return updates

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(self_update, 'Updates', build)

    monkeypatch.setattr(app, 'chat', Mock(side_effect=AssertionError('--update starts no shell')))
    monkeypatch.setattr('sys.argv', ['clai2', '--update', '--database', str(context.store.path)])
    with pytest.raises(SystemExit) as exited:
        cli_module.run()
    assert exited.value.code == exit_code
    assert capsys.readouterr().out == f'{printed}\n'
    [installs] = ran
    assert len(installs) == 1


def test_update_flag_refuses_windows_before_scheduling_an_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A detached PowerShell install has no result when the CLI exits; never report success before uv runs."""
    fetch = Mock(side_effect=AssertionError('the Windows flag must not check for updates'))

    def build(*, channel: Callable[[], UpdateChannel], locate: Callable[[str], Awaitable[str | None]]) -> Updates:
        updates, installs = _updates([channel()], fetch=fetch)
        updates.windows = True
        updates.locate = locate
        updates.hand_off = lambda script: pytest.fail('the install must not be scheduled')
        assert installs == []
        return updates

    monkeypatch.setattr(self_update, 'Updates', build)
    monkeypatch.setattr('sys.argv', ['clai2', '--update', '--database', str(tmp_path / 'c.db')])
    with pytest.raises(SystemExit) as exited:
        cli_module.run()
    assert exited.value.code == 1
    output = capsys.readouterr()
    assert output.out == ''
    assert (
        output.err == '`clai2 --update` cannot install on Windows. Use `/update now` in an interactive clai2 session.\n'
    )
    fetch.assert_not_called()


def test_update_flag_reports_a_failed_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def offline(channel: UpdateChannel) -> str:
        raise httpx.ConnectError('offline')

    def build(*, channel: Callable[[], UpdateChannel], locate: Callable[[str], Awaitable[str | None]]) -> Updates:
        return _updates([channel()], fetch=offline)[0]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(self_update, 'Updates', build)
    monkeypatch.setattr('sys.argv', ['clai2', '--update', '--database', str(tmp_path / 'c.db')])
    with pytest.raises(SystemExit) as exited:
        cli_module.run()
    assert exited.value.code == 1
    assert capsys.readouterr().err == 'Could not check for a newer CLAI: offline\n'


@pytest.mark.parametrize(
    ('argv', 'error'),
    [
        (['--model', 'test'], '--update cannot be combined'),
        (['-m', 'test'], '--update cannot be combined'),
        (['--request-limit', '7'], '--update cannot be combined'),
        (['--fork-session'], '`fork_session` (`--fork-session`) requires `resume` (`--resume`)'),
    ],
)
def test_update_flag_rejects_session_options_before_installing(
    argv: list[str], error: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install = AsyncMock(return_value=0)
    monkeypatch.setattr(cli_module, '_update', install)
    monkeypatch.setattr('sys.argv', ['clai2', '--update', '--database', str(tmp_path / 'c.db'), *argv])
    with pytest.raises(SystemExit) as exited:
        cli_module.run()
    assert exited.value.code == 2
    assert error in capsys.readouterr().err
    install.assert_not_called()


@pytest.mark.parametrize(
    'argv',
    [
        ['-p', 'hi'],
        ['--web'],
        ['--resume', 'abc'],
        ['--resume-codex'],
        ['--continue'],
        ['--session-id', '0b4f5d8e-3c1a-4e6b-9f2d-7a8c9b0d1e2f'],
        ['--agent', 'a:b'],
        ['--worktree'],
        ['config'],
        ['plugins'],
    ],
)
def test_update_flag_rejects_other_modes(
    argv: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr('sys.argv', ['clai2', '--update', *argv])
    with pytest.raises(SystemExit) as exited:
        cli_module.run()
    assert exited.value.code == 2
    assert '--update cannot be combined with config, plugins, --prompt, --web, --resume' in capsys.readouterr().err
