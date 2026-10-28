"""`/system_prompt`: your instructions persist, follow the built-in and plugin ones, and reach the next request."""

import shlex
import sys
from collections.abc import AsyncIterator
from io import StringIO
from pathlib import Path

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from termflow.tui import MenuItem
from termflow.tui.menu import MenuResult
from termflow.tui.pager import Pager

from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_clai2.app import create_stock_agent
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import TurnStart
from pydantic_clai2.runtime.session import Session
from pydantic_clai2.runtime.session_runtime import create_shell
from pydantic_clai2.ui.menus.field_menu import FieldMenu, Runners, save_and_close_item
from pydantic_clai2.ui.menus.system_prompt_menu import (
    APPEND,
    INSTRUCTIONS,
    PROMPT,
    SystemPromptSource,
    summary,
    system_prompt_command,
    system_prompt_menu,
)
from pydantic_clai2.ui.menus.text_editor import EditorApp, TextEditor, build_text_area, edit_text
from tests.clai2.dialog_script import Back, ScriptedDialogs, Type
from tests.clai2.menu_script import Script, make_context, pick
from tests.function_model_prefix import INSTRUCTIONS_CHANGE_MID_RUN

CLOSE = MenuResult(item=save_and_close_item())


def recording(seen: list[str | None]) -> FunctionModel:
    """A model that records the instructions of every request it gets; CLAI streams them all."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        seen.append(info.instructions)
        yield 'done'

    return FunctionModel(stream_function=stream)


def replying(*replies: str | None) -> tuple[list[tuple[str, str]], TextEditor]:
    """An editor returning `replies` in order, and the (title, text) it was opened with each time."""
    opened: list[tuple[str, str]] = []
    queue = list(replies)

    def editor(text: str, /, *, title: str) -> str | None:
        opened.append((title, text))
        return queue.pop(0)

    return opened, editor


def reset(menu: FieldMenu, key: str) -> MenuResult:
    """What the terminal list's R key hands back for the row `key`."""
    return menu.reset_marker(None, MenuItem(key, value=key))


@pytest.mark.parametrize('stock', [True, False], ids=['stock-agent', 'supplied-agent'])
@pytest.mark.moves_cache_prefix(reason=INSTRUCTIONS_CHANGE_MID_RUN)
async def test_saved_instructions_follow_plugin_instructions_and_reach_the_next_request(
    tmp_path: Path, stock: bool
) -> None:
    seen: list[str | None] = []
    store = SettingsStore(tmp_path / 'config.db')
    store.set('model', None)
    store.set('run.instructions', 'Saved before launch.')
    shell = create_shell(
        create_stock_agent(recording(seen)) if stock else Agent(recording(seen)),
        deps=None,
        plugins=(Capability[None](instructions='Plugin guidance.'),),
        usage_limits=None,
        console=Console(file=StringIO()),
        settings=store.load(),
        store=store,
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )
    assert shell.commands.runs_during_turn('/system_prompt')
    assert '/system_prompt: View the system prompt' in shell.commands.help([])
    declared = shell.commands.get('system_prompt')
    assert declared is not None and declared.menu is not None
    assert declared.menu().source.title == 'System prompt', 'the web UI shows the same menu'

    async def turn(text: str) -> str:
        ended = await shell.run_turn(TurnStart(text=text), headless=True)
        assert ended.outcome == 'completed', ended.error
        instructions = seen[-1]
        assert instructions is not None
        return instructions

    first = await turn('first')
    assert first.startswith('Plugin guidance.\n\n')
    assert first.endswith('\n\nSaved before launch.')
    assert first.count('Saved before launch.') == 1, 'bound into a stock agent or passed to the run, never both'
    agent = shell.session.agent
    assert shell.fork_session(None, []).instructions == 'Saved before launch.'

    async def menu(*lists: MenuResult, replies: tuple[str | None, ...] = ()) -> str:
        _, editor = replying(*replies)
        script = Script(lists=list(lists), choices=[], texts=[])
        runners = Runners(run_list=script.run_list, edit=editor)
        return await system_prompt_command(shell.context, [], history=lambda: shell.session.messages, runners=runners)

    assert await menu(pick(INSTRUCTIONS), CLOSE, replies=('Edited in the menu.\n',)) == (
        'Saved your instructions. They apply from your next prompt.'
    )
    second = await turn('second')
    assert second == first.replace('Saved before launch.', 'Edited in the menu.')
    assert shell.session.agent is agent, 'new text does not rebuild the agent'
    assert SettingsStore(tmp_path / 'config.db').load().instructions == 'Edited in the menu.'

    assert (
        await menu(reset(FieldMenu(declared.menu().source), INSTRUCTIONS), CLOSE)
        == 'Removed your instructions. They apply from your next prompt.'
    )
    assert await turn('third') == first.removesuffix('\n\nSaved before launch.')
    assert 'run.instructions' not in SettingsStore(tmp_path / 'config.db').overrides()
    assert await menu(MenuResult(cancelled=True)) == 'No changes.'


async def test_a_stock_agent_without_plugins_sends_them_from_its_first_request() -> None:
    seen: list[str | None] = []
    session = Session(create_stock_agent(recording(seen)), deps=None)
    session.instructions = 'Be brief.'
    await session.prompt('hello')
    assert seen[-1] is not None and seen[-1].endswith('\n\nBe brief.')


async def test_arguments_are_refused(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    with pytest.raises(ValueError, match='Usage: /system_prompt'):
        await system_prompt_command(context, ['show'], history=lambda: ())


def test_rows_summarize_yours_and_take_values_only_there(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    source = SystemPromptSource(context, history=lambda: ())
    rows = {row.key: row for row in source.rows()}
    assert list(rows) == [INSTRUCTIONS, APPEND, PROMPT]
    assert [row.button for row in rows.values()] == ['Edit…', 'Append…', 'View…']
    assert source.current(rows[INSTRUCTIONS]) == rows[INSTRUCTIONS].default == 'none', 'nothing to reset'
    assert source.current(rows[APPEND]) == source.current(rows[PROMPT]) == ''
    assert rows[INSTRUCTIONS].note == ''
    assert source.reset(rows[INSTRUCTIONS]) == 'You have no instructions to remove.'

    assert source.problem(rows[INSTRUCTIONS], 'One line') is None
    assert (
        source.apply(rows[INSTRUCTIONS], '  One line ') == 'Saved your instructions. They apply from your next prompt.'
    )
    assert source.current(rows[INSTRUCTIONS]) == 'One line (1 line)'
    assert source.problem(rows[PROMPT], 'x') == 'Full system prompt takes no value; open it instead.'
    with pytest.raises(ValueError, match='takes no value'):
        source.apply(rows[APPEND], 'x')
    assert source.reset(rows[APPEND]) == 'Append to yours has nothing to reset.'
    assert context.settings.instructions == 'One line', 'only its own row removes them'

    context.project = ProjectSettings(overrides={'run.instructions': 'From the project.'})
    assert 'project file' in {row.key: row for row in source.rows()}[INSTRUCTIONS].note


def test_summary_fits_a_row_and_disarms_control_characters() -> None:
    assert summary('') == 'none'
    assert summary(f'First line {"word " * 20}\nSecond') == 'First line word word word word word wor… (2 lines)'
    shown = summary('From a repository: \x1b]52;c;ZXZpbA==\x07')
    assert '\x1b' not in shown and '\x07' not in shown, 'control characters cannot reach the terminal'


def test_the_full_prompt_is_what_the_latest_answered_request_sent(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    history: list[ModelMessage] = []
    source = SystemPromptSource(context, history=lambda: history)
    assert source.full_prompt().startswith('Nothing was sent in this conversation yet.')
    done = ModelResponse(parts=[TextPart('done')])
    history += [
        ModelRequest(parts=[UserPromptPart('hello')], instructions='Built-in guidance.\n\nYours.'),
        done,
        # A final tool return after the latest response was never sent.
        ModelRequest(parts=[UserPromptPart('never sent')], instructions='Unsent.'),
    ]
    assert source.full_prompt() == 'Built-in guidance.\n\nYours.'
    history.insert(0, ModelRequest(parts=[SystemPromptPart('Agent system prompt.'), UserPromptPart('first')]))
    assert source.full_prompt() == 'Agent system prompt.\n\nBuilt-in guidance.\n\nYours.'
    assert SystemPromptSource(context, history=lambda: [history[0], done]).full_prompt() == 'Agent system prompt.'
    # Instructions removed since an earlier request are not shown as sent.
    removed = [*history[1:3], ModelRequest(parts=[UserPromptPart('again')]), done]
    assert SystemPromptSource(context, history=lambda: removed).full_prompt() == (
        'The latest request had no system prompt.'
    )


async def test_actions_edit_append_and_view_through_dialogs(tmp_path: Path) -> None:
    context, applied = make_context(tmp_path)
    history = [ModelRequest(parts=[UserPromptPart('hi')], instructions='Built-in.'), ModelResponse(parts=[])]
    menu = system_prompt_menu(context, history=lambda: history)
    edit, append, view = (menu.actions[key] for key in (INSTRUCTIONS, APPEND, PROMPT))
    saved = ['Saved your instructions. They apply from your next prompt.']

    assert await edit(ScriptedDialogs([Back()])) == []
    dialogs = ScriptedDialogs([Type(text='  First.\n')])
    assert await edit(dialogs) == saved
    [asked] = dialogs.asked
    assert (asked.kind, asked.title, asked.multiline, asked.initial) == ('text', 'Edit your instructions', True, '')
    unchanged = ScriptedDialogs([Type(text='First.')])
    assert await edit(unchanged) == [], 'nothing changed'
    assert unchanged.asked[0].initial == 'First.'

    assert await append(ScriptedDialogs([Type(text='Second.')])) == saved
    assert context.settings.instructions == 'First.\n\nSecond.', 'a blank line between paragraphs'
    assert await append(ScriptedDialogs([Type(text='  ')])) == []
    assert await append(ScriptedDialogs([Back()])) == []

    viewed = ScriptedDialogs([])
    assert await view(viewed) == []
    assert [(asked.kind, asked.text) for asked in viewed.asked] == [('view', 'Built-in.')]

    assert await edit(ScriptedDialogs([Type(text='')])) == [
        'Removed your instructions. They apply from your next prompt.'
    ], 'emptying them removes them'
    assert await append(ScriptedDialogs([Type(text='Only.')])) == saved
    assert context.settings.instructions == 'Only.'
    assert applied == ['run.instructions'] * 4


async def test_the_terminal_runs_the_same_menu_with_its_editor_and_pager(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    opened, editor = replying('Mine.\n', None, 'More.')
    viewed: list[Pager] = []
    script = Script(lists=[pick(INSTRUCTIONS), pick(APPEND), pick(APPEND), pick(PROMPT), CLOSE], choices=[], texts=[])
    runners = Runners(run_list=script.run_list, edit=editor, run_pager=viewed.append)
    message = await system_prompt_command(context, [], history=lambda: (), runners=runners)
    assert message == '\n'.join(['Saved your instructions. They apply from your next prompt.'] * 2)
    assert opened == [
        ('Edit your instructions', ''),
        ('Append to your instructions', ''),
        ('Append to your instructions', ''),
    ]
    [pager] = viewed
    assert pager.line_count >= 1
    assert context.settings.instructions == 'Mine.\n\nMore.'


def _editor(tmp_path: Path, name: str) -> str:
    """A command line for an editor that prefixes the file with `name:`, as `$VISUAL` or `$EDITOR`."""
    script = tmp_path / f'{name}.py'
    script.write_text(
        'import pathlib, sys\n'
        'path = pathlib.Path(sys.argv[1])\n'
        f'path.write_text({name!r} + ":" + path.read_text(encoding="utf-8"), encoding="utf-8")\n'
    )
    return shlex.join([sys.executable, str(script)])


@pytest.mark.subprocess(reason='runs Python scripts as the `$VISUAL`/`$EDITOR` commands')
def test_edit_text_opens_visual_then_editor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, 'platform', 'linux')
    built_in: list[EditorApp] = []
    area = built_in.append
    monkeypatch.setenv('VISUAL', _editor(tmp_path, 'visual'))
    monkeypatch.setenv('EDITOR', _editor(tmp_path, 'editor'))
    assert edit_text('Hé there', title='Edit', run_area=area) == 'visual:Hé there'
    monkeypatch.setenv('VISUAL', '')
    assert edit_text('text', title='Edit', run_area=area) == 'editor:text'
    monkeypatch.setenv('EDITOR', shlex.join([sys.executable, '-c', 'raise SystemExit(1)']))
    assert edit_text('text', title='Edit', run_area=area) is None, 'an editor exiting with an error cancels'
    assert built_in == []


@pytest.mark.parametrize(
    ('platform', 'editor'),
    [
        ('linux', None),
        ('linux', ''),
        ('linux', 'no-such-editor-for-clai-tests'),
        ('linux', 'vim "unclosed'),
        ('win32', 'vim'),
    ],
    ids=['unset', 'empty', 'cannot-start', 'unparsable', 'windows'],
)
def test_edit_text_falls_back_to_the_built_in_editor(
    monkeypatch: pytest.MonkeyPatch, platform: str, editor: str | None
) -> None:
    monkeypatch.delenv('VISUAL', raising=False)
    if editor is None:
        monkeypatch.delenv('EDITOR', raising=False)
    else:
        monkeypatch.setenv('EDITOR', editor)
    built_in: list[EditorApp] = []

    def area(app: EditorApp) -> str:
        built_in.append(app)
        return 'built-in'

    # Its own session, made before patching `sys.platform`: otherwise the editor's input and output are created for
    # the patched platform, unless an earlier test on this worker already created them.
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        monkeypatch.setattr(sys, 'platform', platform)
        assert edit_text('text', title='Edit', run_area=area) == 'built-in'
    assert len(built_in) == 1


@pytest.mark.parametrize(
    ('keys', 'expected'),
    [
        ('\x1b[F and more\rSecond line\x13', 'Your text and more\nSecond line'),
        ('typed\x1b', None),
        ('typed\x03', None),
    ],
    ids=['ctrl-s-saves', 'esc-cancels', 'ctrl-c-cancels'],
)
def test_built_in_editor_saves_multiple_lines_or_cancels(keys: str, expected: str | None) -> None:
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        app = build_text_area(title='Edit your instructions', text='Your text')
        pipe.send_text(keys)
        assert app.run() == expected
