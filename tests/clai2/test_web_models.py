"""`/model`'s declared menu: the Settings window's Models section, and the terminal menus it shares."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import JsonValue, TypeAdapter
from termflow.tui.menu import MenuResult

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_clai2.builtin_plugins.web.protocol import PROTOCOL, DialogOpened, SettingsChanged, StatusChanged
from pydantic_clai2.config.credential_store import save_codex_credentials
from pydantic_clai2.ui.menus.dialogs import run_settings_menu
from pydantic_clai2.ui.menus.model_menu import add_model
from pydantic_clai2.ui.menus.model_picker import (
    ModelsSource,
    _remove,  # pyright: ignore[reportPrivateUsage]
    models_menu,
    remove_model,
)
from tests.clai2.dialog_script import Back, Confirm, Pick, ScriptedDialogs
from tests.clai2.menu_script import Script, make_context, pick
from tests.clai2.test_accounts import CODEX
from tests.clai2.test_web import TOKEN, MakeWeb, Web, make_web as make_web, receive_until

SOCKET = f'/ws?token={TOKEN}&protocol={PROTOCOL}'
ORIGIN = {'origin': 'http://testserver'}
GPT = 'openai:gpt-5.6'
ADDED = 'openai:gpt-5.1'


def rows(web_: Web, path: str) -> dict[str, dict[str, JsonValue]]:
    response = web_.get(path)
    assert response.status_code == 200, response.text
    menu = TypeAdapter(dict[str, JsonValue]).validate_python(response.json())
    fields = TypeAdapter(list[dict[str, JsonValue]]).validate_python(menu['fields'])
    return {str(field['key']): field for field in fields}


def test_bare_model_and_effort_open_the_models_section(make_web: MakeWeb) -> None:
    """Bare `/model` and `/effort` open their menus in the page; `/fast`, which toggles, still runs as typed."""
    web_ = make_web(Agent(TestModel()))
    assert web_.post('/api/input', {'text': '/model'}).json() == {'menu': 'model'}
    assert web_.post('/api/input', {'text': '/effort'}).json() == {'menu': 'effort'}
    assert web_.post('/api/input', {'text': '/fast'}).json() == {'menu': None}
    # `/effort`'s menu is the active model's settings, the rows `/effort` reads and writes.
    effort = rows(web_, '/api/commands/effort/fields')
    assert 'max_tokens' in effort and effort['custom_params']['action'] is True


def test_the_models_menu_selects_and_edits_models_from_the_page(make_web: MakeWeb) -> None:
    web_ = make_web(Agent(TestModel()))
    store = web_.shell.context.store
    for name in (GPT, 'chain:pair', 'add'):
        store.add_model(name=name)
    store.save_chain(name='pair', models=[GPT, 'anthropic:claude-sonnet-4-5'])
    store.save_model_settings(GPT, {'openai_reasoning_effort': 'high'})
    listed = rows(web_, '/api/commands/model/fields')
    assert list(listed) == ['model', GPT, 'add', 'remove', 'chain:pair'], 'a model named like a row has none'
    assert listed['model']['value'] == '(not set)' and GPT in str(listed['model']['choices'])
    assert (listed[GPT]['menu'], listed[GPT]['value']) == (True, 'Reasoning Effort')
    assert listed['chain:pair']['description'] == f'Tries in order: 1. {GPT} 2. anthropic:claude-sonnet-4-5'
    assert (listed['add']['action'], listed['add']['menu']) == (True, False)
    with web_.client.websocket_connect(SOCKET, headers=ORIGIN) as socket:
        socket.receive_text()
        # Choosing a model is `/model NAME`: saved, the session switched, and the status row shows it at once.
        chosen = web_.post('/api/commands/model/fields/model', {'value': GPT})
        assert chosen.json()['messages'] == ['Saved model. Applied.']
        assert web_.shell.session.model == GPT and store.load().model == GPT
        assert receive_until(socket, StatusChanged)[1].status.model == GPT
        assert {field['key']: field['note'] for field in chosen.json()['menu']['fields']}[GPT] == 'current'
        # A saved model's row opens its settings: the same rows and store `/model settings NAME` and `/effort` use.
        settings = rows(web_, f'/api/commands/model/fields?menu={GPT}')
        assert settings['openai_reasoning_effort']['value'] == 'high'
        edited = web_.post(f'/api/commands/model/fields/openai_reasoning_effort?menu={GPT}', {'value': 'low'})
        assert edited.json()['menu']['fields'][0]['key'] in settings
        assert store.model_settings(GPT) == {'openai_reasoning_effort': 'low'}
        assert rows(web_, '/api/commands/effort/fields')['openai_reasoning_effort']['value'] == 'low'
        receive_until(socket, SettingsChanged)
        # Its custom parameters are an action, asked through the page's dialogs.
        with ThreadPoolExecutor(1) as pool:
            action = pool.submit(web_.post, f'/api/commands/model/fields/custom_params/action?menu={GPT}')
            dialog = receive_until(socket, DialogOpened)[1].dialog
            assert dialog.title == f'Custom Params - {GPT}'
            assert web_.post(f'/api/dialogs/{dialog.id}', {'value': None}).status_code == 204
            assert action.result(timeout=10).json()['messages'] == []
    # A row that opens a menu, or an action, takes no value; a menu that is not there is not found.
    assert web_.post('/api/commands/model/fields/' + 'chain:pair', {'value': 'x'}).status_code == 404
    refused = web_.post('/api/commands/model/fields/add', {'value': 'x'})
    assert refused.json()['detail'] == 'Add a model takes no value; choose it instead.'
    assert web_.post('/api/commands/model/fields/add', {'value': ''}).json()['messages'] == [
        'Add a model has nothing to reset.'
    ]
    assert web_.get('/api/commands/model/fields?menu=nope').status_code == 404
    assert web_.get('/api/commands/set/fields?menu=model').status_code == 404


def test_the_models_source_reads_and_resets_like_the_set_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context, applied = make_context(tmp_path)
    source = ModelsSource(context)
    [model, *_] = source.rows()

    def missing(name: str) -> None:
        raise ValueError(f'Cannot use {name}: its extra is not installed.')

    with monkeypatch.context() as patched:
        patched.setattr('pydantic_clai2.models.model_catalog.check_installed', missing)
        assert source.problem(model, 'groq:llama') == 'Cannot use groq:llama: its extra is not installed.'
    assert source.problem(model, 'test') is None
    assert source.apply(model, 'test') == 'Saved model. Applied.' and context.settings.model == 'test'
    assert source.current(next(row for row in source.rows() if row.key == 'test')) == 'Defaults'
    assert source.reset(model).startswith('Reset model.') and applied == ['model', 'model']


async def test_adding_a_model_asks_provider_model_and_account(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    assert await add_model(context, ScriptedDialogs([Back()])) == []
    dialogs = ScriptedDialogs(
        [Pick(value='openai'), Back(), Pick(value='openai'), Pick(value=ADDED)],
    )
    assert await add_model(context, dialogs) == ['Saved model. Applied.']
    providers = dialogs.asked[0]
    assert 'openai' in [option.value for option in providers.options]
    assert not {'github-copilot', 'openrouter', 'vllm'} & {option.value for option in providers.options}
    assert providers.current == 'openai-codex', 'the current model is a Codex one'
    assert dialogs.asked[1].title == 'Add a model: openai'
    assert context.settings.model == ADDED and ADDED in context.store.models()
    # Two signed-in accounts: which one runs it, or every one in turn.
    save_codex_credentials(account='openai-codex', value=CODEX)
    save_codex_credentials(account='openai-codex@work', value=CODEX)
    model = 'openai-codex:gpt-6-astra'
    every = 'openai-codex@*:gpt-6-astra'
    dialogs = ScriptedDialogs(
        [Pick(value='openai-codex'), Pick(value=model), Back(), Pick(value='openai-codex'), Pick(value=model)]
        + [Pick(value=every)]
    )
    assert await add_model(context, dialogs) == ['Saved model. Applied.']
    assert dialogs.asked[2].title == 'Which openai-codex account?'
    assert context.settings.model == every


async def test_removing_a_model_confirms_and_keeps_the_protected_ones(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    current = context.settings.model
    assert current is not None
    context.store.add_model(name=GPT)
    assert await remove_model(context, ScriptedDialogs([Back()])) == []
    protected = ScriptedDialogs([Pick(value=current)])
    assert await remove_model(context, protected) == ['Select another model before deleting the current model.']
    assert protected.asked[0].options[0].description
    assert await remove_model(context, ScriptedDialogs([Pick(value=GPT), Confirm(yes=False)])) == []
    assert await remove_model(context, ScriptedDialogs([Pick(value=GPT), Confirm(yes=True)])) == [f'Deleted {GPT}.']
    assert GPT not in context.store.models()
    # Another session made it the saved default since the list was read.
    context.store.set('model', GPT)
    assert _remove(context, GPT) == (False, f'Kept {GPT}: it became the saved default.')


async def test_a_row_that_opens_a_menu_opens_it_in_the_terminal(tmp_path: Path) -> None:
    """`run_settings_menu` shows the menu a row opens, here a saved model's settings, then its own list again."""
    context, _ = make_context(tmp_path)
    menu = models_menu(context)
    assert set(menu.menus) == set(context.store.models())
    closed = MenuResult(cancelled=True)
    script = Script(lists=[pick(context.settings.model), closed, closed], choices=[], texts=[])
    assert await run_settings_menu(menu, script.runners) == []
    assert script.opened == ['list', 'list', 'list']
