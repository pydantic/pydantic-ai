"""Headless output, lifecycle, persistence, and CLI validation."""

from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore

from pydantic_clai2 import _cli, headless
from pydantic_clai2.config import PluginSettings, Settings
from pydantic_clai2.project_settings import ProjectSettings
from pydantic_clai2.settings_store import SettingsStore


async def test_answer_resume_and_no_ask_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    model = TestModel(call_tools=[], custom_output_text='[literal] ' + 'long ' * 100)
    agent = Agent(model)
    monkeypatch.setattr(headless, 'create_agent', lambda: agent)
    store = SettingsStore(tmp_path / 'config.db')
    # Even a user override must not activate in headless mode.
    store.save_plugin(PluginSettings(id='ask_user', factory='missing_module:fail'))
    with agent.override(model=model):
        assert (
            await headless.run_headless(
                text='/literal prompt', settings=Settings(model='test'), store=store, project=ProjectSettings()
            )
            == 0
        )
        saved = SqliteConversationStore(database=tmp_path / 'sessions.db')
        entries = await saved.listing()
        assert len(entries) == 1
        first = await saved.get(conversation_id=entries[0].id)
        assert first.summary.outcome == 'completed'
        assert (
            await headless.run_headless(
                text='follow up',
                settings=Settings(model='test'),
                store=store,
                project=ProjectSettings(),
                resume=entries[0].id,
            )
            == 0
        )
        second = await saved.get(conversation_id=entries[0].id)
    assert len(second.messages) > len(first.messages)
    assert model.last_model_request_parameters is not None
    assert 'ask_user_question' not in [tool.name for tool in model.last_model_request_parameters.function_tools]
    assert store.plugins()[0].enabled
    captured = capsys.readouterr()
    assert model.custom_output_text is not None
    assert captured.out == (model.custom_output_text + '\n') * 2
    assert captured.err == ''


@pytest.mark.parametrize('mode', ['cancel', 'raise', 'screen', 'load', 'model'])
async def test_failure_is_nonzero_and_silent_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], mode: str
) -> None:
    plugin = tmp_path / 'guard.py'
    plugin.write_text(
        'def activate(host):\n'
        + (
            '    raise RuntimeError("load failed")\n'
            if mode == 'load'
            else '    @host.on("turn_start")\n'
            '    async def guard(event):\n'
            + {
                'cancel': '        event.cancel("declined")\n',
                'raise': '        raise RuntimeError("guard failed")\n',
                'screen': '        async with host.full_screen():\n            pass\n',
                'model': '        pass\n',
            }[mode]
        )
    )
    store = SettingsStore(tmp_path / 'config.db')
    monkeypatch.setattr(headless, 'DEFAULT_PLUGINS', (PluginSettings(id='guard', factory='guard', path=str(plugin)),))
    assert (
        await headless.run_headless(
            text='hello',
            settings=Settings(model='unknown:missing' if mode == 'model' else 'test'),
            store=store,
            project=ProjectSettings(),
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err


async def test_missing_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='Choose a model'):
        await headless.run_headless(
            text='hello',
            settings=Settings(model=None),
            store=SettingsStore(tmp_path / 'config.db'),
            project=ProjectSettings(),
        )


@pytest.mark.parametrize('args', [['-p'], ['-p', ''], ['-p', 'x', '--resume'], ['-p', 'x', 'config']])
def test_cli_invalid_prompt(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    monkeypatch.setattr('sys.argv', ['clai2', *args])
    with pytest.raises(SystemExit) as error:
        _cli.run()
    assert error.value.code == 2


@pytest.mark.parametrize('interrupt', [False, True])
def test_cli_alias_and_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: bool) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    store.set('model', 'test')
    monkeypatch.setenv('CLAI_MODEL', 'env:model')
    monkeypatch.setattr('sys.argv', ['clai2', '--database', str(store.path), '-m', 'explicit:model', '-p', 'hello'])

    async def run_headless(
        *, text: str, settings: Settings, store: SettingsStore, project: ProjectSettings, resume: str | None
    ) -> int:
        assert text == 'hello'
        assert settings.model == 'explicit:model'
        if interrupt:
            raise KeyboardInterrupt
        return 0

    monkeypatch.setattr(headless, 'run_headless', run_headless)
    with pytest.raises(SystemExit) as error:
        _cli.run()
    assert error.value.code == (130 if interrupt else 0)
    assert store.load().model == 'test'


async def test_cancel_saves_history_and_closes_plugins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    started = anyio.Event()

    async def request(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        started.set()
        await anyio.sleep_forever()
        yield 'unreachable'  # pragma: no cover

    model = FunctionModel(stream_function=request)
    agent = Agent(model)
    monkeypatch.setattr(headless, 'create_agent', lambda: agent)
    log = tmp_path / 'lifecycle'
    plugin = tmp_path / 'lifecycle.py'
    plugin.write_text(
        'from pathlib import Path\n'
        'def activate(host):\n'
        '    @host.on("turn_end")\n'
        '    async def ended(event):\n'
        f'        Path({str(log)!r}).write_text(event.outcome)\n'
        '    @host.on("session_end")\n'
        '    async def closed(event):\n'
        f'        p = Path({str(log)!r})\n'
        '        p.write_text(p.read_text() + ":" + event.reason)\n'
    )
    monkeypatch.setattr(
        headless, 'DEFAULT_PLUGINS', (PluginSettings(id='lifecycle', factory='lifecycle', path=str(plugin)),)
    )

    async def run() -> None:
        await headless.run_headless(
            text='retain me',
            settings=Settings(model='test'),
            store=SettingsStore(tmp_path / 'config.db'),
            project=ProjectSettings(),
        )

    with agent.override(model=model):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(run)
            with anyio.fail_after(10):
                await started.wait()
            tasks.cancel_scope.cancel()
    assert log.read_text() == 'cancelled:error'
    saved = SqliteConversationStore(database=tmp_path / 'sessions.db')
    entries = await saved.listing()
    assert entries[0].outcome == 'cancelled'
    assert capsys.readouterr().out == ''
