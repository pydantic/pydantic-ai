"""Every pax surface that reads keys, keyed by where `pydantic_ai_pax` constructs it, with what each key does there.

`test_key_surfaces.py` fails on a construction of a key-reading widget that this table does not list, and presses
every key in `KEYS` on each case. A key a case does not list stays open. Each cell declares what the surface does on
`main`; a change that gives a key a new meaning on purpose updates its cell.

A case drives the construction site it is registered under, so it calls that site even when the site is private.
Every case paints on `sys.stdout`, as termflow's widgets do, so a test can read the frames it drew.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from contextlib import nullcontext
from io import StringIO
from pathlib import Path

import anyio
import pytest
from rich.console import Console
from termflow.tui.menu import MenuItem, MenuResult
from termflow.tui.pager import Pager
from termflow.tui.textinput import TextInput

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai_harness.ask_user import AskUserRequest, Question, QuestionOption
from pydantic_ai_pax.auth import _pick_login  # pyright: ignore[reportPrivateUsage]
from pydantic_ai_pax.builtin_plugins.ask_user_menu import TerminalAnswerer
from pydantic_ai_pax.config.api_keys import build_key_menu
from pydantic_ai_pax.config.credential_store import save_codex_credentials
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.models import openrouter, vllm
from pydantic_ai_pax.plugins import PluginLogin
from pydantic_ai_pax.plugins.menus import Option
from pydantic_ai_pax.runtime.sessions import ConversationSummary
from pydantic_ai_pax.ui.menus import chain_menu, worktree_cleanup
from pydantic_ai_pax.ui.menus.accounts_menu import AccountsMenu, choose_account
from pydantic_ai_pax.ui.menus.dialogs import TerminalDialogs, build_pager
from pydantic_ai_pax.ui.menus.field_menu import FieldMenu, Runners
from pydantic_ai_pax.ui.menus.key_menu import build_keys_menu
from pydantic_ai_pax.ui.menus.model_menu import ModelMenu, build_model_settings_picker
from pydantic_ai_pax.ui.menus.model_picker import (
    _confirm_delete,  # pyright: ignore[reportPrivateUsage]
    build_model_picker,
)
from pydantic_ai_pax.ui.menus.plugin_menu import PluginMenu
from pydantic_ai_pax.ui.menus.question_menu import QuestionMenu
from pydantic_ai_pax.ui.menus.rewind import build_rewind_menu
from pydantic_ai_pax.ui.menus.session_browser import SessionBrowser
from pydantic_ai_pax.ui.menus.set_menu import SettingsSource
from pydantic_ai_pax.ui.menus.spinner_picker import SpinnerPicker
from pydantic_ai_pax.ui.menus.theme_picker import build_theme_picker
from pydantic_ai_pax.ui.prompt import line_input
from pydantic_ai_pax.ui.prompt.line_input import LineInput, read_line
from pydantic_ai_pax.ui.prompt.prompt_surface import PromptSurface
from tests.manifests import Exempt
from tests.pax.key_contract import CLOSE, PICK, Keys, Surface, reader_case, widget_case
from tests.pax.menu_script import make_context
from tests.pax.test_accounts import signed
from tests.pax.test_ask_user_menu import ScreenLog
from tests.pax.test_plugin_menu import PLUGIN, make_loader, run_now
from tests.pax.test_spinners import catalogue

Site = tuple[str, str, str]
"""Module under `pydantic_ai_pax`, the enclosing qualified name, and the widget or builder it constructs."""

QUESTION = Question(
    header='Approach',
    question='How should we do it?',
    options=(QuestionOption(label='Refactor'), QuestionOption(label='Patch')),
)
CODEX = '{"access_token": "a", "refresh_token": "r", "account_id": "x"}'


def _history() -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart('first')], run_id='first'),
        ModelResponse(parts=[TextPart('one')], run_id='first'),
        ModelRequest(parts=[UserPromptPart('second')], run_id='second'),
        ModelResponse(parts=[TextPart('two')], run_id='second'),
    ]


def _two_accounts(path: Path) -> object:
    save_codex_credentials(account='openai-codex', value=CODEX)
    save_codex_credentials(account='openai-codex@work', value=CODEX)
    return choose_account(SettingsStore(path / 'config.db'), 'openai-codex:gpt-6', Runners())


def _deletable_model(path: Path) -> object:
    context, _ = make_context(path)
    context.store.add_model(name='openai:gpt-6')
    return _confirm_delete(context, 'openai:gpt-6', Runners(), message='', messages=[])


def _typed_member(path: Path) -> object:
    typing = MenuResult(item=MenuItem('Type a model name...', value=chain_menu._Row.TYPE))  # pyright: ignore[reportPrivateUsage]
    return chain_menu._pick_member(make_context(path)[0], [], Runners(run_list=lambda menu: typing))  # pyright: ignore[reportPrivateUsage]


def _plugins(path: Path) -> object:
    loader = make_loader(path)
    (loader.plugins_dir / 'alpha.py').write_text(f'"""Say hello on start."""\n{PLUGIN}')
    return PluginMenu(loader, apply=run_now).build().run()


def _settings_fields(path: Path, *, searchable: bool) -> FieldMenu:
    return FieldMenu(SettingsSource(make_context(path)[0]), searchable=searchable)


def _choice_row_menu(path: Path) -> object:
    fields = _settings_fields(path, searchable=True)
    row = next(row for row in fields.rows if len(row.choices) > 1)
    return fields.build_choices(row).run()


def _text_row_editor(path: Path) -> object:
    fields = _settings_fields(path, searchable=True)
    row = next(row for row in fields.rows if not row.choices)
    return fields.build_editor(row).run()


async def _removal_question(path: Path) -> object:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(worktree_cleanup, 'question_input', lambda: nullcontext(lambda: 'x'))
        return await worktree_cleanup.ask_removal('uncommitted changes')


async def _asked_question(path: Path) -> object:
    answerer = TerminalAnswerer(full_screen=ScreenLog(), console=Console(file=StringIO()))
    return await answerer.answer_questions(request=AskUserRequest(questions=(QUESTION,)), key_source=lambda: 'x')


async def _read_code(path: Path) -> object:
    surface = PromptSurface(output=StringIO(), size=lambda: (80, 24))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(line_input, 'question_input', lambda: nullcontext(lambda: 'x'))
        return await read_line('Code: ', console=Console(file=surface, width=80))


def _browser(path: Path) -> object:
    entries = [
        ConversationSummary(id='one', workspace='/a/project', title='Fix renderer'),
        ConversationSummary(id='two', workspace='/a/project', title='Add resume'),
    ]
    return SessionBrowser(
        entries=entries,
        workspace='/a/project',
        active_id='one',
        refresh=lambda query, limit, archived: entries,
        preview=lambda session_id: f'Transcript {session_id}',
        delete=entries.remove,
        rename=lambda entry, title: None,
        archive=lambda entry, archived: None,
        output=sys.stdout,
        key_source=lambda: 'x',
        size=lambda: (110, 24),
    ).run()


def _question(menu: QuestionMenu, keys: Keys) -> object:
    return menu.run(console=Console(file=sys.stdout, width=80, height=24), key_source=keys)


def _line(line: LineInput, keys: Keys) -> object:
    return line.run(surface=PromptSurface(output=sys.stdout, size=lambda: (80, 24)), keys=keys)


def _browse(browser: SessionBrowser, keys: Keys) -> object:
    browser.key_source = keys
    return browser.loop()


async def _never(show: Callable[[str], None]) -> None:
    await anyio.sleep_forever()


_PROMPT_TOOLKIT_LINE = Exempt(
    "a prompt_toolkit line prompt for a secret or an address: prompt_toolkit's own line editing, Enter submits, "
    'Ctrl-C and Esc cancel; pax binds no key here'
)

KEY_SURFACES: dict[Site, Surface | Exempt] = {
    ('auth', '_pick_login', 'MenuBuilder'): Surface(
        reason='`/login`, which sign-in to run',
        cases=(
            widget_case(
                'pick',
                lambda path: _pick_login({'demo': PluginLogin(name='demo', handler=signed)}, Runners()),
                PICK,
            ),
        ),
    ),
    ('builtin_plugins.ask_user_menu', 'TerminalAnswerer.answer_questions', 'QuestionMenu'): Surface(
        reason="the model's `ask_user` question, answered inline",
        cases=(reader_case('single choice', QuestionMenu, _asked_question, _question, PICK),),
    ),
    ('config.api_keys', 'build_key_menu', 'MenuBuilder'): Surface(
        reason='which saved API key a model uses',
        cases=(
            widget_case(
                'pick', lambda path: build_key_menu(names=['openai'], label='OpenAI', optional=True).run(), PICK
            ),
        ),
    ),
    ('config.api_keys', 'set_api_key', 'PromptSession'): _PROMPT_TOOLKIT_LINE,
    ('models.key_profiles', '_save_key', 'PromptSession'): _PROMPT_TOOLKIT_LINE,
    ('models.openrouter', 'authentication_menu', 'MenuBuilder'): Surface(
        reason='how to sign in to OpenRouter',
        cases=(widget_case('pick', lambda path: openrouter.authentication_menu().run(), PICK),),
    ),
    ('models.openrouter', 'choose', 'MenuBuilder'): Surface(
        reason='which discovered OpenRouter model to add',
        cases=(widget_case('pick', lambda path: openrouter.choose(['a/one', 'b/two']), PICK),),
    ),
    ('models.openrouter', 'connection_action', 'MenuBuilder'): Surface(
        reason='reuse or replace the saved OpenRouter connection',
        cases=(widget_case('pick', lambda path: openrouter.connection_action(), PICK),),
    ),
    ('models.openrouter', 'prompt_connection', 'PromptSession'): _PROMPT_TOOLKIT_LINE,
    ('models.vllm', 'choose', 'MenuBuilder'): Surface(
        reason='which discovered vLLM model to add',
        cases=(widget_case('pick', lambda path: vllm.choose(['one', 'two']), PICK),),
    ),
    ('models.vllm', 'connection_action', 'MenuBuilder'): Surface(
        reason='reuse or replace the saved vLLM connection',
        cases=(widget_case('pick', lambda path: vllm.connection_action(), PICK),),
    ),
    ('models.vllm', 'prompt_connection', 'PromptSession'): _PROMPT_TOOLKIT_LINE,
    ('runtime.sessions', 'Sessions.browse.browse', 'SessionBrowser'): Surface(
        reason='`/resume`, the saved sessions by project',
        cases=(reader_case('browse', SessionBrowser, _browser, _browse, CLOSE, cancelled=lambda result: result == ''),),
    ),
    ('ui.menus.accounts_menu', 'AccountsMenu.build', 'MenuBuilder'): Surface(
        reason='`/accounts`, the signed-in accounts',
        cases=(widget_case('list', lambda path: AccountsMenu(SettingsStore(path / 'config.db')).build().run(), PICK),),
    ),
    ('ui.menus.accounts_menu', 'choose_account', 'MenuBuilder'): Surface(
        reason='which account runs a model, when its provider has more than one',
        cases=(widget_case('pick', _two_accounts, PICK),),
    ),
    ('ui.menus.chain_menu', '_ask_name', 'TextInputBuilder'): Surface(
        reason="a fallback chain's name",
        cases=(
            widget_case(
                'type',
                lambda path: chain_menu._ask_name(make_context(path)[0], Runners(), title='Name', initial='fast'),  # pyright: ignore[reportPrivateUsage]
                PICK,
                widget=TextInput,
            ),
        ),
    ),
    ('ui.menus.chain_menu', '_build_editor', 'MenuBuilder'): Surface(
        reason="a fallback chain's members, in order",
        cases=(
            widget_case(
                'list',
                lambda path: chain_menu._build_editor('fast', ['openai:gpt-6'], cursor=0, notice='').run(),  # pyright: ignore[reportPrivateUsage]
                CLOSE,
            ),
        ),
    ),
    ('ui.menus.chain_menu', '_pick_member', 'MenuBuilder'): Surface(
        reason='a model to add to a fallback chain',
        cases=(
            widget_case(
                'pick',
                lambda path: chain_menu._pick_member(make_context(path)[0], [], Runners()),  # pyright: ignore[reportPrivateUsage]
                PICK,
            ),
        ),
    ),
    ('ui.menus.chain_menu', '_pick_member', 'TextInputBuilder'): Surface(
        reason='a model name typed for a fallback chain',
        # Enter on an empty name fails the validator and stays.
        cases=(widget_case('type', _typed_member, CLOSE, widget=TextInput),),
    ),
    ('ui.menus.dialogs', 'TerminalDialogs.choose', 'MenuBuilder'): Surface(
        reason="a plugin settings menu's choice",
        cases=(
            widget_case(
                'pick',
                lambda path: TerminalDialogs().choose(
                    title='Pick', options=[Option(value='a', label='A'), Option(value='b', label='B')]
                ),
                PICK,
            ),
        ),
    ),
    ('ui.menus.dialogs', 'TerminalDialogs.confirm', 'MenuBuilder'): Surface(
        reason="a plugin settings menu's yes or no",
        cases=(widget_case('pick', lambda path: TerminalDialogs().confirm(title='Sure?', yes='Yes', no='No'), PICK),),
    ),
    ('ui.menus.dialogs', 'TerminalDialogs.text', 'TextInputBuilder'): Surface(
        reason="a plugin settings menu's typed value",
        cases=(widget_case('type', lambda path: TerminalDialogs().text(title='Name'), PICK, widget=TextInput),),
    ),
    ('ui.menus.dialogs', 'TerminalDialogs.wait', 'MenuBuilder'): Surface(
        reason='a waiting screen with one Cancel row, such as a browser sign-in',
        cases=(
            widget_case(
                'wait', lambda path: TerminalDialogs().wait(title='Signing in', description='...', task=_never), PICK
            ),
        ),
    ),
    ('ui.menus.dialogs', 'build_pager', 'PagerBuilder'): Surface(
        reason='a read-only page of text',
        cases=(widget_case('page', lambda path: build_pager('Notes', 'one\ntwo').run(), PICK, widget=Pager),),
    ),
    ('ui.menus.field_menu', 'FieldMenu.build', 'MenuBuilder'): Surface(
        reason='`/set` and every plugin settings list: Space cycles a row with choices',
        cases=(
            widget_case('searchable', lambda path: _settings_fields(path, searchable=True).build().run(), PICK),
            widget_case('not searchable', lambda path: _settings_fields(path, searchable=False).build().run(), PICK),
        ),
    ),
    ('ui.menus.field_menu', 'FieldMenu.build_choices', 'MenuBuilder'): Surface(
        reason="one setting's choices",
        cases=(widget_case('pick', _choice_row_menu, PICK),),
    ),
    ('ui.menus.field_menu', 'FieldMenu.build_editor', 'TextInputBuilder'): Surface(
        reason="one setting's typed value",
        cases=(widget_case('type', _text_row_editor, PICK, widget=TextInput),),
    ),
    ('ui.menus.key_menu', 'build_keys_menu', 'MenuBuilder'): Surface(
        reason='`/keys`, the saved API keys',
        cases=(widget_case('list', lambda path: build_keys_menu(names=['openai']).run(), PICK),),
    ),
    ('ui.menus.model_menu', 'ModelMenu.build', 'MenuBuilder'): Surface(
        reason='`/model add`, the catalog of models',
        cases=(widget_case('list', lambda path: ModelMenu(make_context(path)[0]).build().run(), PICK),),
    ),
    ('ui.menus.model_menu', 'ModelMenu.build_providers', 'MenuBuilder'): Surface(
        reason='`/model add`, the providers',
        cases=(widget_case('list', lambda path: ModelMenu(make_context(path)[0]).build_providers().run(), PICK),),
    ),
    ('ui.menus.model_menu', 'build_model_settings_picker', 'MenuBuilder'): Surface(
        reason='which model `/model settings` configures',
        cases=(
            widget_case(
                'pick',
                lambda path: build_model_settings_picker(context=make_context(path)[0], current=None).run(),
                PICK,
            ),
        ),
    ),
    ('ui.menus.model_picker', '_confirm_delete', 'MenuBuilder'): Surface(
        reason='whether to delete a saved model',
        cases=(widget_case('pick', _deletable_model, PICK),),
    ),
    ('ui.menus.model_picker', 'build_model_picker', 'MenuBuilder'): Surface(
        reason='`/model`, the saved models and chains',
        cases=(widget_case('list', lambda path: build_model_picker(make_context(path)[0]).run(), PICK),),
    ),
    ('ui.menus.plugin_menu', 'PluginMenu.build', 'MenuBuilder'): Surface(
        reason='`/plugins`, where Enter and Space toggle the highlighted plugin',
        cases=(widget_case('list', _plugins, CLOSE),),
    ),
    ('ui.menus.rewind', 'build_rewind_menu', 'MenuBuilder'): Surface(
        reason='`/rewind`, the turns to go back to',
        cases=(widget_case('pick', lambda path: build_rewind_menu(_history()).run(), PICK),),
    ),
    ('ui.menus.spinner_picker', 'SpinnerPicker.build', 'MenuBuilder'): Surface(
        reason='`/spinner`',
        cases=(
            widget_case(
                'pick',
                lambda path: SpinnerPicker(catalogue(path, selected='dots'), clock=lambda: 0.0).build().run(),
                PICK,
            ),
        ),
    ),
    ('ui.menus.task_view', 'TaskView.__init__', 'LiveApp'): Exempt(
        "the `/tasks` view's frame loop, registered under `open_tasks.view`"
    ),
    ('ui.menus.task_view', 'open_tasks.view', 'TaskView'): Exempt(
        '`/tasks`: letters type a message and actions are Ctrl chords; `tests/pax/test_task_view.py` drives its keys'
    ),
    ('ui.menus.text_editor', 'build_text_area', 'Application'): Exempt(
        'the built-in multi-line editor: keys type text; `tests/pax/test_system_prompt_menu.py` drives its bindings'
    ),
    ('ui.menus.theme_picker', 'build_theme_picker', 'MenuBuilder'): Surface(
        reason='`/theme`',
        cases=(widget_case('pick', lambda path: build_theme_picker(make_context(path)[0]).run(), PICK),),
    ),
    ('ui.menus.worktree_cleanup', 'ask_removal', 'QuestionMenu'): Surface(
        reason='whether to remove a `--worktree` checkout on exit',
        cases=(reader_case('pick', QuestionMenu, _removal_question, _question, PICK),),
    ),
    ('ui.prompt.line_input', 'read_line', 'LineInput'): Surface(
        reason='a sign-in code or other line a command asks for in the live editor',
        cases=(reader_case('type', LineInput, _read_code, _line, PICK),),
    ),
    ('ui.prompt.line_input', 'read_line', 'PromptSession'): _PROMPT_TOOLKIT_LINE,
    ('ui.prompt.terminal', 'Terminal._line_prompt', 'PromptSession'): _PROMPT_TOOLKIT_LINE,
    ('ui.prompt.terminal', 'Terminal._opened', 'LivePrompt'): Exempt(
        'the editor: its chords are decoded per terminal in `tests/pax/chords.py`'
    ),
}
