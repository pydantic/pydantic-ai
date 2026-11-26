"""Headless output, lifecycle, persistence, and CLI validation."""

import io
import json
from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, TextOutput
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from pydantic_ai_pax.app import create_stock_agent
from pydantic_ai_pax.cli import agent_import, cli, headless
from pydantic_ai_pax.config import PluginSettings, Settings
from pydantic_ai_pax.config.project_settings import ProjectSettings
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.runtime.workspace_trust import WorkspaceTrust, undecided_notice


def untrusted_notice(store: SettingsStore) -> str:
    """What a headless run in the current directory prints on stderr while you never decided whether to trust it."""
    notice = undecided_notice(WorkspaceTrust.saved(store, Path.cwd()))
    assert notice is not None
    return notice + '\n'


@pytest.mark.parametrize('supplied', [False, True])
async def test_delegation_defaults_are_scoped_to_stock_agents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, supplied: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    model = TestModel(call_tools=[], custom_output_text='done')
    monkeypatch.setattr(headless, 'create_agent', lambda: create_stock_agent(model))
    assert (
        await headless.run_headless(
            text='go',
            settings=Settings(model=None),
            store=SettingsStore(tmp_path / 'config.db'),
            project=ProjectSettings(),
            agent=Agent(model, deps_type=type(None)) if supplied else None,
        )
        == 0
    )
    assert model.last_model_request_parameters is not None
    names = {tool.name for tool in model.last_model_request_parameters.function_tools}
    if supplied:
        assert not names
    else:
        assert {'read_file', 'delegate_task'} <= names


async def test_answer_resume_and_no_ask_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    model = TestModel(call_tools=[], custom_output_text='[literal] ' + 'long ' * 100)
    agent = create_stock_agent(model)
    monkeypatch.setattr(headless, 'create_agent', lambda: agent)
    store = SettingsStore(tmp_path / 'config.db')
    # Even a user override must not activate in headless mode.
    store.save_plugin(PluginSettings(id='ask_user', factory='missing_module:fail'))
    assert (
        await headless.run_headless(
            text='/literal prompt', settings=Settings(model=None), store=store, project=ProjectSettings()
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
            settings=Settings(model=None),
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
    assert captured.err == untrusted_notice(store) * 2


@pytest.mark.parametrize(
    ('mode', 'error'),
    [
        ('cancel', 'declined'),
        ('raise', "Plugin 'guard': RuntimeError: guard failed"),
        ('screen', "Plugin 'guard': RuntimeError: User interaction is unavailable in headless mode"),
        ('load', "Plugin 'guard': RuntimeError: load failed"),
        ('model', 'Unknown model: unknown:missing'),
    ],
)
async def test_failure_is_nonzero_and_silent_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], mode: str, error: str
) -> None:
    plugin = tmp_path / 'guard.py'
    plugin.write_text(
        'from pydantic_ai_pax.plugins import Plugin\n'
        'class Guard(Plugin):\n'
        + (
            '    def __init__(self, host, settings):\n        raise RuntimeError("load failed")\n'
            if mode == 'load'
            else '    async def on_turn_start(self, event):\n'
            + {
                'cancel': '        event.cancel("declined")\n',
                'raise': '        raise RuntimeError("guard failed")\n',
                'screen': '        async with self.host.full_screen():\n            pass\n',
                'model': '        pass\n',
            }[mode]
        )
    )
    store = SettingsStore(tmp_path / 'config.db')
    monkeypatch.setattr(headless, 'STOCK_PLUGINS', (PluginSettings(id='guard', factory='guard', path=str(plugin)),))
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
    assert captured.err == f'{untrusted_notice(store)}{error}\n'


async def test_missing_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='Choose a model'):
        await headless.run_headless(
            text='hello',
            settings=Settings(model=None),
            store=SettingsStore(tmp_path / 'config.db'),
            project=ProjectSettings(),
        )


@pytest.mark.parametrize(
    'args',
    [
        ['-p'],
        ['-'],
        ['-p', ''],
        ['-p', 'x', '--resume'],
        ['-p', 'x', 'config'],
        ['-p', 'x', 'more'],
        ['x', '-p', 'more'],
        ['--output-format', 'json'],
        ['--output-format', 'yaml', '-p', 'x'],
    ],
)
def test_cli_invalid_prompt(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    monkeypatch.setattr('sys.stdin', io.StringIO(' \n'))
    monkeypatch.setattr('sys.argv', ['pax', *args])
    with pytest.raises(SystemExit) as error:
        cli.run()
    assert error.value.code == 2


@pytest.mark.parametrize('interrupt', [False, True])
def test_cli_alias_and_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: bool) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    store.set('model', 'test')
    monkeypatch.setenv('CLAI_MODEL', 'env:model')
    monkeypatch.setattr('sys.argv', ['pax', '--database', str(store.path), '-m', 'explicit:model', '-p', 'hello'])

    async def run_headless(
        *,
        text: str,
        settings: Settings,
        store: SettingsStore,
        project: ProjectSettings,
        resume: str | None,
        resume_from: str | None,
        agent: object,
        **_: object,
    ) -> int:
        assert agent is None
        assert text == 'hello'
        assert settings.model == 'explicit:model'
        if interrupt:
            raise KeyboardInterrupt
        return 0

    monkeypatch.setattr(headless, 'run_headless', run_headless)
    with pytest.raises(SystemExit) as error:
        cli.run()
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
    plugin = tmp_path / 'lifecycle_probe.py'
    plugin.write_text(
        'from pathlib import Path\n'
        'from pydantic_ai_pax.plugins import Plugin\n'
        'def record(text, *, after=False):\n'
        f'    log = Path({str(log)!r})\n'
        "    log.write_text((log.read_text() + ':' if after else '') + text)\n"
        'class Lifecycle(Plugin):\n'
        '    async def on_turn_end(self, event):\n'
        '        record(event.outcome)\n'
        '    async def on_session_end(self, event):\n'
        '        record(event.reason, after=True)\n'
    )
    monkeypatch.setattr(
        headless, 'STOCK_PLUGINS', (PluginSettings(id='lifecycle', factory='lifecycle_probe', path=str(plugin)),)
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


class City(BaseModel):
    name: str
    population: int


async def _echo(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    """Answer with the prompt the CLI sent, so a test sees how it was put together."""
    [part] = [part for message in messages if isinstance(message, ModelRequest) for part in message.parts]
    assert isinstance(part, UserPromptPart) and isinstance(part.content, str)
    yield f'echo: {part.content}'


async def _fail(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    raise RuntimeError('model is down')
    # Makes this an async generator.
    yield  # pragma: no cover


class Opaque:
    """An output that JSON can't serialize, so it prints as its `str()`."""

    def __str__(self) -> str:
        return 'opaque'


def _opaque(text: str) -> object:
    """Annotated `object`, since an output function's return type needs a schema and `Opaque` has none."""
    return Opaque()


_AGENTS: dict[str, Agent[None, object]] = {
    'echo': Agent(FunctionModel(stream_function=_echo)),
    'city': Agent(TestModel(), output_type=City),
    'opaque': Agent(FunctionModel(stream_function=_echo), output_type=TextOutput(_opaque)),
    'fail': Agent(FunctionModel(stream_function=_fail)),
}


def run_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *args: str
) -> tuple[int | str | None, str, str]:
    """Run `pax --database ... ARGS` in-process with `-a NAME` naming one of `_AGENTS`; the exit code, stdout, and stderr."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent_import, 'import_agent', _AGENTS.__getitem__)
    monkeypatch.setattr('sys.argv', ['pax', '--database', str(tmp_path / 'config.db'), *args])
    with pytest.raises(SystemExit) as exited:
        cli.run()
    captured = capsys.readouterr()
    return exited.value.code, captured.out, captured.err


@pytest.mark.parametrize(
    'args',
    [
        ['-a', 'echo', 'what is up'],
        ['-a', 'echo', 'what', 'is', 'up'],
        ['what', 'is', '-a', 'echo', 'up'],
        ['-a', 'echo', '-p', 'what is up'],
        ['-a', 'echo', '--', 'what', 'is', 'up'],
        ['-a', 'echo', 'what', '--', 'is', 'up'],
    ],
)
def test_cli_positional_prompt_runs_like_p(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], args: list[str]
) -> None:
    assert run_cli(tmp_path, monkeypatch, capsys, *args)[:2] == (0, 'echo: what is up\n')


@pytest.mark.parametrize(
    ('args', 'answer'),
    [
        (['-a', 'echo', '--', 'explain', '--version'], 'echo: explain --version\n'),
        (['-a', 'echo', '--', 'review', '--agent=other:agent', '-p', 'x'], 'echo: review --agent=other:agent -p x\n'),
    ],
)
def test_cli_words_after_double_dash_are_prompt_text_not_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], args: list[str], answer: str
) -> None:
    """After `--`, words that look like options stay in the prompt, so prompt text can't pick another agent."""
    assert run_cli(tmp_path, monkeypatch, capsys, *args)[:2] == (0, answer)


def test_cli_subcommand_is_not_a_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The assertion is that it never runs.
    async def run_headless(**_: object) -> int:  # pragma: no cover
        raise AssertionError('`pax config` ran a prompt')

    monkeypatch.setattr(headless, 'run_headless', run_headless)
    monkeypatch.setattr('sys.argv', ['pax', '--database', str(tmp_path / 'config.db'), 'config', 'get', 'model'])
    cli.run()
    assert capsys.readouterr().out == f'"{Settings().model}"\n'


@pytest.mark.parametrize('args', [['-'], ['-p'], ['-p', '-']])
def test_cli_prompt_from_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], args: list[str]
) -> None:
    monkeypatch.setattr('sys.stdin', io.StringIO('from stdin'))
    assert run_cli(tmp_path, monkeypatch, capsys, '-a', 'echo', *args)[:2] == (0, 'echo: from stdin\n')


class _Terminal(io.StringIO):
    """Stdin attached to a terminal, which a test fails on if it is read."""

    def isatty(self) -> bool:
        return True

    def read(self, size: int | None = -1) -> str:  # pragma: no cover
        raise AssertionError('read the prompt from a terminal')


@pytest.mark.parametrize('args', [['-'], ['-p']])
def test_cli_prompt_from_terminal_stdin_fails_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], args: list[str]
) -> None:
    """With nothing piped in, waiting on stdin would look like a hang."""
    monkeypatch.setattr('sys.stdin', _Terminal())
    code, out, err = run_cli(tmp_path, monkeypatch, capsys, '-a', 'echo', *args)
    assert (code, out) == (2, '')
    assert err.endswith('reads the prompt from stdin; pipe it in or give TEXT\n')


@pytest.mark.parametrize(
    ('args', 'expected'),
    [
        (['-a', 'city', '-p', 'go'], {'name': 'a', 'population': 0}),
        (['-a', 'city', '--output-format', 'text', 'go'], {'name': 'a', 'population': 0}),
        (['-a', 'city', '--output-format', 'json', 'go'], {'output': {'name': 'a', 'population': 0}}),
        (['-a', 'echo', '--output-format', 'json', 'go'], {'output': 'echo: go'}),
        (['-a', 'opaque', 'go'], 'opaque'),
        (['-a', 'opaque', '--output-format', 'json', 'go'], {'output': 'opaque'}),
    ],
)
def test_cli_prints_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    args: list[str],
    expected: object,
) -> None:
    code, out, err = run_cli(tmp_path, monkeypatch, capsys, *args)
    assert code == 0, err
    assert out.endswith('\n') and json.loads(out) == expected


def test_cli_failure_goes_to_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run_cli(tmp_path, monkeypatch, capsys, '-a', 'fail', '--output-format', 'json', 'go')
    assert (code, out) == (1, '')
    assert err.endswith('model is down\n')
