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
from termflow.tui.menu import MenuResult
from termflow.tui.pager import Pager

from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_clai2._app import create_shell, create_stock_agent
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import TurnStart
from pydantic_clai2.runtime._session import Session
from pydantic_clai2.ui.menus.field_menu import SAVE_AND_CLOSE_DETAILS, save_and_close_item
from pydantic_clai2.ui.menus.system_prompt_menu import (
    Action,
    SystemPromptMenu,
    TextEditor,
    build_pager,
    run_system_prompt_menu,
    system_prompt_command,
)
from pydantic_clai2.ui.menus.text_editor import EditorApp, build_text_area, edit_text
from tests.clai2.menu_script import Script, make_context, pick

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


@pytest.mark.parametrize('stock', [True, False], ids=['stock-agent', 'supplied-agent'])
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
        return await system_prompt_command(
            shell.context, [], history=lambda: shell.session.messages, runners=script.runners, editor=editor
        )

    assert await menu(pick(Action.EDIT), CLOSE, replies=('Edited in the menu.\n',)) == (
        'Saved your instructions. They apply from your next prompt.'
    )
    second = await turn('second')
    assert second == first.replace('Saved before launch.', 'Edited in the menu.')
    assert shell.session.agent is agent, 'new text does not rebuild the agent'
    assert SettingsStore(tmp_path / 'config.db').load().instructions == 'Edited in the menu.'

    assert await menu(pick(Action.RESET), CLOSE) == 'Removed your instructions. They apply from your next prompt.'
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


def test_menu_shows_yours_as_editable_and_the_full_prompt_as_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr('pydantic_clai2.ui.menus.system_prompt_menu.terminal_size', lambda: (120, 30))
    monkeypatch.delenv('VISUAL', raising=False)
    monkeypatch.setenv('EDITOR', 'vim -u NONE')
    context, _ = make_context(tmp_path)
    history: list[ModelMessage] = []
    menu = SystemPromptMenu(context, history=lambda: history)
    items = menu.items()
    assert [item.label for item in items] == [action.value for action in Action] + ['Save & close']
    assert items[0].description.endswith('none')
    assert items[2].disabled, 'nothing to reset'
    edit = menu.details(items[0])
    assert edit.startswith('Your instructions (editable)\n\n(none)\n')
    assert "Sent after CLAI's built-in instructions, AGENTS.md, and plugin instructions" in ' '.join(edit.split())
    assert 'Opens in vim -u NONE.' in edit
    assert 'project file' not in edit
    assert 'Nothing was sent in this conversation yet.' in menu.details(items[3])
    assert menu.details(items[4]) == SAVE_AND_CLOSE_DETAILS
    assert menu.build(1) is not None

    done = ModelResponse(parts=[TextPart('done')])
    history += [
        ModelRequest(parts=[UserPromptPart('hello')], instructions='Built-in guidance.\n\nYours.'),
        done,
        # A final tool return after the latest response was never sent.
        ModelRequest(parts=[UserPromptPart('never sent')], instructions='Unsent.'),
    ]
    assert menu.full_prompt() == 'Built-in guidance.\n\nYours.'
    history.insert(0, ModelRequest(parts=[SystemPromptPart('Agent system prompt.'), UserPromptPart('first')]))
    assert menu.full_prompt() == 'Agent system prompt.\n\nBuilt-in guidance.\n\nYours.'
    assert SystemPromptMenu(context, history=lambda: [history[0], done]).full_prompt() == 'Agent system prompt.'
    view = menu.details(items[3])
    assert view.startswith('Full system prompt (read-only)\n')
    assert view.endswith('Built-in guidance.\n\nYours.')
    # Instructions removed since an earlier request are not shown as sent.
    reset = [*history[1:3], ModelRequest(parts=[UserPromptPart('again')]), done]
    assert SystemPromptMenu(context, history=lambda: reset).full_prompt() == 'The latest request had no system prompt.'
    history += [ModelRequest(parts=[], instructions='\n'.join(f'Rule {n}.' for n in range(100))), done]
    view = menu.details(items[3]).splitlines()
    assert len(view) < 30, 'the panel fits the screen'
    assert view[-2:] == ['Rule 14.', '…'], 'Enter opens the rest'

    context.set_setting(['run.instructions', 'Single line'])
    assert menu.items()[0].description.endswith('1 line')
    context.set_setting(['run.instructions', f'First line\n{"word " * 60}'])
    items = menu.items()
    assert items[0].description.endswith('2 lines')
    assert not items[2].disabled
    wrapped = menu.details(items[0]).splitlines()
    assert wrapped[2] == 'First line'
    assert len([line for line in wrapped if line.startswith('word')]) > 1, 'long lines wrap to the panel'

    context.set_setting(['run.instructions', 'From a repository: \x1b]52;c;ZXZpbA==\x07'])
    shown = menu.details(items[0])
    assert '\x1b' not in shown and '\x07' not in shown, 'control characters cannot reach the terminal'
    assert 'From a repository: \\x1b]52;c;ZXZpbA==\\x07' in shown

    monkeypatch.delenv('EDITOR')
    context.project = ProjectSettings(overrides={'run.instructions': 'From the project.'})
    edit = ' '.join(menu.details(items[1]).split())
    assert 'Opens in the built-in editor: ctrl-s save · esc cancel.' in edit
    assert 'The project file sets them again at next start.' in edit


def test_flow_edits_appends_resets_and_views(tmp_path: Path) -> None:
    context, applied = make_context(tmp_path)
    menu = SystemPromptMenu(context, history=lambda: ())
    opened, editor = replying(
        '  First.\n',  # Saved without the surrounding whitespace.
        None,  # Cancelled.
        'First.\n',  # Unchanged.
        'Second.',
        '  \n',  # Nothing to add.
        None,
        'Only.',
        '',  # Emptied: removes them.
    )
    script = Script(
        lists=[
            pick(Action.EDIT),
            pick(Action.EDIT),
            pick(Action.EDIT),
            pick(Action.APPEND),
            pick(Action.APPEND),
            pick(Action.APPEND),
            pick(Action.VIEW),
            pick(Action.RESET),
            pick(Action.APPEND),
            pick(Action.EDIT),
            MenuResult(cancelled=True),
        ],
        choices=[],
        texts=[],
    )
    viewed: list[Pager] = []
    messages = run_system_prompt_menu(menu, runners=script.runners, editor=editor, view=viewed.append)
    saved = 'Saved your instructions. They apply from your next prompt.'
    removed = 'Removed your instructions. They apply from your next prompt.'
    assert messages == [saved, saved, removed, saved, removed]
    assert opened == [
        ('Edit your instructions', ''),
        ('Edit your instructions', 'First.'),
        ('Edit your instructions', 'First.'),
        ('Append to your instructions', ''),
        ('Append to your instructions', ''),
        ('Append to your instructions', ''),
        ('Append to your instructions', ''),
        ('Edit your instructions', 'Only.'),
    ]
    [pager] = viewed
    assert pager.line_count == 2  # The not-yet-sent note, wrapped.
    assert applied == ['run.instructions'] * 5
    assert context.settings.instructions == ''
    assert 'run.instructions' not in context.store.overrides()


def test_appending_keeps_a_blank_line_between_paragraphs(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    context.set_setting(['run.instructions', 'First.'])
    _, editor = replying('Second.\n')
    assert SystemPromptMenu(context, history=lambda: ()).append(editor) is not None
    assert context.store.load().instructions == 'First.\n\nSecond.'


def test_pager_wraps_the_full_prompt() -> None:
    pager = build_pager('short\n\n' + 'word ' * 100)
    assert pager.line_count > 3


def _editor(tmp_path: Path, name: str) -> str:
    """A command line for an editor that prefixes the file with `name:`, as `$VISUAL` or `$EDITOR`."""
    script = tmp_path / f'{name}.py'
    script.write_text(
        'import pathlib, sys\n'
        'path = pathlib.Path(sys.argv[1])\n'
        f'path.write_text({name!r} + ":" + path.read_text(encoding="utf-8"), encoding="utf-8")\n'
    )
    return shlex.join([sys.executable, str(script)])


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
    monkeypatch.setattr(sys, 'platform', platform)
    monkeypatch.delenv('VISUAL', raising=False)
    if editor is None:
        monkeypatch.delenv('EDITOR', raising=False)
    else:
        monkeypatch.setenv('EDITOR', editor)
    built_in: list[EditorApp] = []

    def area(app: EditorApp) -> str:
        built_in.append(app)
        return 'built-in'

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
