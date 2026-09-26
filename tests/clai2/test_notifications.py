"""Desktop notifications stay optional and do not expose conversation content."""

import io
import subprocess
from collections.abc import Sequence
from pathlib import Path

import anyio
import pytest
from pydantic import JsonValue
from pydantic_ai import Agent, ToolDefinition
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.ask_user import AskUser, AskUserRequest, AskUserResponse
from rich.console import Console

from pydantic_clai2 import notifications
from pydantic_clai2._app import DEFAULT_PLUGINS, create_shell
from pydantic_clai2.plugins import PluginHost, TurnEnd, TurnOutcome, TurnStart
from pydantic_clai2.project_settings import ProjectSettings
from pydantic_clai2.settings_store import SettingsStore


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    messages: list[str] = []

    async def send(message: str) -> None:
        messages.append(message)

    monkeypatch.setattr(notifications, 'notify', send)
    monkeypatch.delenv('SSH_CONNECTION', raising=False)
    monkeypatch.delenv('SSH_TTY', raising=False)
    return messages


def host(*, terminal: bool = True) -> PluginHost[None]:
    return PluginHost(name='notifications', console=Console(file=io.StringIO(), force_terminal=terminal), settings={})


@pytest.mark.parametrize(
    ('outcome', 'expected'),
    [
        ('completed', ['Task finished.']),
        ('failed', ['Task failed. Check the terminal for details.']),
        ('cancelled', []),
    ],
)
async def test_turn_outcomes(outcome: TurnOutcome, expected: list[str], sent: list[str]) -> None:
    plugin = host()
    notifications.activate(plugin)
    for handler in plugin.handlers:
        await handler(TurnEnd(text='secret prompt', outcome=outcome, error=ValueError('secret error')))
    assert sent == expected


@pytest.mark.parametrize('environment', ['pipe', 'SSH_CONNECTION', 'SSH_TTY'])
def test_nonlocal_or_noninteractive_is_quiet(
    environment: str, sent: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    if environment != 'pipe':
        monkeypatch.setenv(environment, 'remote')
    plugin = host(terminal=environment != 'pipe')
    notifications.activate(plugin)
    assert plugin.handlers == []
    assert plugin.capabilities == []
    assert sent == []


async def test_question_notifies_before_answerer(sent: list[str]) -> None:
    plugin = host()
    notifications.activate(plugin)

    async def answer(request: AskUserRequest) -> AskUserResponse:
        assert sent == ['Your input is needed.']
        return AskUserResponse(cancelled=True)

    class QuestionModel(TestModel):
        def gen_tool_args(self, tool_def: ToolDefinition) -> JsonValue:
            return {
                'questions': [
                    {'question': 'secret question', 'header': 'Secret', 'options': [{'label': 'Yes'}, {'label': 'No'}]}
                ]
            }

    agent = Agent(
        QuestionModel(call_tools=['ask_user_question']),
        deps_type=type(None),
        capabilities=[AskUser(answerer=answer), *plugin.capabilities],
    )
    await agent.run('private prompt')
    assert sent == ['Your input is needed.']


async def test_default_plugin_disable_enable(tmp_path: Path, sent: list[str]) -> None:
    shell = create_shell(
        Agent(TestModel()),
        deps=None,
        plugins=(),
        usage_limits=None,
        settings=None,
        project=ProjectSettings(),
        console=Console(file=io.StringIO(), force_terminal=True),
        store=SettingsStore(tmp_path / 'settings.db'),
        builtin_plugins=[plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'notifications'],
    )
    await shell.loader.load_all()
    await shell.loader.fire(await shell.run_turn(TurnStart(text='private prompt'), headless=True))
    assert sent == ['Task finished.']
    await shell.loader.command(['disable', 'notifications'])
    await shell.loader.fire(await shell.run_turn(TurnStart(text='another private prompt'), headless=True))
    assert sent == ['Task finished.']
    await shell.loader.command(['enable', 'notifications'])
    await shell.loader.fire(await shell.run_turn(TurnStart(text='third private prompt'), headless=True))
    assert sent == ['Task finished.', 'Task finished.']
    await shell.loader.close('exit')


@pytest.mark.parametrize('platform', ['darwin', 'linux', 'win32'])
async def test_native_command(platform: str, monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    async def run(command: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert kwargs == {'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL, 'check': False}
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(notifications.sys, 'platform', platform)
    monkeypatch.setattr(notifications.anyio, 'run_process', run)
    message = 'Quotes " and \\ and newlines\n are data'
    await notifications.notify(message)
    if platform == 'darwin':
        assert commands == [
            [
                '/usr/bin/osascript',
                '-e',
                'on run argv\n display notification (item 1 of argv) with title "CLAI2"\nend run',
                message,
            ]
        ]
    elif platform == 'linux':
        assert commands == [['/usr/bin/notify-send', '--app-name=CLAI2', '--', 'CLAI2', message]]
    else:
        assert commands == []


@pytest.mark.parametrize('error', [FileNotFoundError(), PermissionError(), TimeoutError()])
async def test_notification_service_failure(error: Exception, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run(command: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise error

    monkeypatch.setattr(notifications.sys, 'platform', 'darwin')
    monkeypatch.setattr(notifications.anyio, 'run_process', run)
    await notifications.notify('Task finished.')


async def test_cancellation_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    started = anyio.Event()
    cleaned = anyio.Event()

    async def run(command: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        started.set()
        try:
            await anyio.sleep_forever()
        finally:
            cleaned.set()
        raise AssertionError('unreachable')  # pragma: no cover

    monkeypatch.setattr(notifications.sys, 'platform', 'darwin')
    monkeypatch.setattr(notifications.anyio, 'run_process', run)
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(notifications.notify, 'Task finished.')
        await started.wait()
        tasks.cancel_scope.cancel()
    assert cleaned.is_set()
