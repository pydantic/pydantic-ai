"""`/accounts`: listing, adding, ordering, and signing out of accounts, and `PROVIDER@*` pooling."""

import io
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import keyring
import pytest
from rich.console import Console
from termflow.tui import MenuItem
from termflow.tui.keys import Key
from termflow.tui.menu import MenuResult
from termflow.tui.textinput import TextInputResult

from pydantic_ai import Agent
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelHTTPError, UserError
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.test import TestModel
from pydantic_clai2._app import _ModelResolver, create_shell  # pyright: ignore[reportPrivateUsage]
from pydantic_clai2.auth import CodexAuth, login_command
from pydantic_clai2.config.credential_store import has_credentials, save_codex_credentials
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore, StoredAccount
from pydantic_clai2.errors import error_message
from pydantic_clai2.models import accounts as accounts_module, key_profiles
from pydantic_clai2.models.accounts import Account, LoginChoice, accounts, login_choices, pool, remember, sign_out
from pydantic_clai2.plugins import ModelProvider, PluginLogin
from pydantic_clai2.ui.menus import accounts_menu
from pydantic_clai2.ui.menus.accounts_menu import AccountsMenu, choose_account, new_account, open_accounts_menu
from pydantic_clai2.ui.menus.field_menu import SAVE_AND_CLOSE_DETAILS, Runners, is_save_and_close
from pydantic_clai2.ui.menus.model_menu import ModelMenu, run_model_flow
from tests.clai2.menu_script import Script, make_context, pick, typed
from tests.clai2.test_plugin_menu import FakeMenu, unstyled

CODEX = json.dumps({'access_token': 'a', 'refresh_token': 'r', 'account_id': 'x'})


def store_at(tmp_path: Path) -> SettingsStore:
    return SettingsStore(tmp_path / 'config.db')


async def signed(message: str = 'Signed in.') -> str:
    return message


def plugin_login(
    *, name: str = 'claude', profiles: bool = True, models: tuple[str, ...] = ('claude-code:opus',)
) -> PluginLogin:
    async def for_profile(profile: str) -> str:
        return f'Signed in as {profile}.'

    return PluginLogin(name=name, handler=signed, profile_handler=for_profile if profiles else None, models=models)


def test_the_store_keeps_order_labels_and_owners(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    for profile in (None, 'work', 'side'):
        store.add_account(StoredAccount(provider='openai-codex', profile=profile))
    store.add_account(StoredAccount(provider='claude-code', profile='work', plugin_login='claude'))
    # Adding a known account keeps its place and label.
    store.rename_account(provider='openai-codex', profile='work', label='Work laptop')
    store.add_account(StoredAccount(provider='openai-codex', profile='work'))
    assert [(row.provider, row.profile, row.label) for row in store.accounts()] == [
        ('claude-code', 'work', None),
        ('openai-codex', None, None),
        ('openai-codex', 'work', 'Work laptop'),
        ('openai-codex', 'side', None),
    ]
    store.move_account(provider='openai-codex', profile='side', offset=-5)  # clamped to the front
    store.move_account(provider='openai-codex', profile=None, offset=1)
    store.move_account(provider='openai-codex', profile='missing', offset=1)
    assert [row.profile for row in store.accounts() if row.provider == 'openai-codex'] == ['side', 'work', None]
    store.remove_account(provider='openai-codex', profile='work')
    store.rename_account(provider='openai-codex', profile=None, label=None)
    assert [row.profile for row in store.accounts()] == ['work', 'side', None]


def test_accounts_find_existing_logins_and_their_state(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    save_codex_credentials(account='openai-codex', value=CODEX)
    save_codex_credentials(account='openai-codex@work', value=CODEX)
    save_codex_credentials(account='gateway/openai@team', value='{}')
    store.add_account(StoredAccount(provider='claude-code', profile='work', plugin_login='claude'))
    store.add_account(StoredAccount(provider='github-copilot', profile='gone'))
    found = {(item.provider, item.profile): item for item in accounts(store)}
    assert set(found) == {
        ('claude-code', 'work'),
        ('gateway/openai', 'team'),
        ('github-copilot', 'gone'),
        ('openai-codex', None),
        ('openai-codex', 'work'),
    }
    assert found['openai-codex', None].signed_in and found['openai-codex', None].name == 'default'
    assert found['openai-codex', 'work'].login == 'openai-codex@work'
    assert found['openai-codex', 'work'].model('gpt-6') == 'openai-codex@work:gpt-6'
    assert not found['github-copilot', 'gone'].signed_in
    # A plugin keeps its own tokens, so its accounts count as signed in.
    plugin = found['claude-code', 'work']
    assert plugin.signed_in and plugin.login == 'claude@work'
    assert [item.profile for item in pool(store, 'openai-codex')] == [None, 'work']
    assert pool(store, 'github-copilot') == []


def test_a_login_an_older_clai_kept_in_the_keyring_is_listed_and_pooled(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    save_codex_credentials(account='openai-codex@work', value=CODEX)
    keyring.set_password('pydantic-clai2', 'openai-codex', CODEX)  # how CLAI stored logins before files
    assert [item.login for item in accounts(store)] == ['openai-codex', 'openai-codex@work']
    assert [item.profile for item in pool(store, 'openai-codex')] == [None, 'work']
    assert keyring.get_password('pydantic-clai2', 'openai-codex') is None, 'it moved into its file'


def test_remember_and_sign_out(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    remember(store, login='claude', profile='work', plugin=plugin_login())
    remember(store, login='solo', profile=None, plugin=plugin_login(name='solo', models=()))
    save_codex_credentials(account='openai-codex@work', value=CODEX)
    remember(store, login='openai-codex', profile='work')
    rows = {(row.provider, row.profile): row for row in store.accounts()}
    assert rows['claude-code', 'work'].plugin_login == 'claude'
    assert rows['solo', None].plugin_login == 'solo'  # no models: listed under its sign-in name
    by_login = {item.login: item for item in accounts(store)}
    assert sign_out(store, by_login['openai-codex@work']) == 'Signed out of openai-codex@work.'
    assert not has_credentials(account='openai-codex@work')
    assert 'its own logout' in sign_out(store, by_login['claude@work'])
    assert [item.login for item in accounts(store)] == ['solo']


def test_login_choices_group_and_describe_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    accounts_module._keyed_providers.cache_clear()  # pyright: ignore[reportPrivateUsage]

    def keyed(name: str) -> object:
        if name == 'mistral':
            raise ImportError('no SDK')
        return object() if name in ('openai', 'openrouter', 'anthropic') else None

    monkeypatch.setattr(key_profiles, 'keyed_provider', keyed)
    plugins = {'claude': plugin_login(), 'solo': plugin_login(name='solo', profiles=False, models=())}
    choices = login_choices(plugins)
    accounts_module._keyed_providers.cache_clear()  # pyright: ignore[reportPrivateUsage]
    assert [(choice.kind, choice.login, choice.provider) for choice in choices] == [
        ('subscription', 'openai-codex', 'openai-codex'),
        ('subscription', 'github-copilot', 'github-copilot'),
        ('plugin', 'claude', 'claude-code'),
        ('plugin', 'solo', 'solo'),
        ('connection', 'openrouter', 'openrouter'),
        ('connection', 'vllm', 'vllm'),
        ('api key', 'anthropic', 'anthropic'),
        ('api key', 'openai', 'openai'),
    ]
    assert [choice.profiles for choice in choices[2:4]] == [True, False]
    assert not choices[-1].has_default and choices[0].has_default


async def test_login_records_accounts_it_signs_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = store_at(tmp_path)
    codex = CodexAuth(Console(file=io.StringIO()))
    replies = iter(['sk-work', None])

    async def prompt_api_key(*, prompt: object, label: str) -> str | None:
        return next(replies)

    monkeypatch.setattr(key_profiles, 'prompt_api_key', prompt_api_key)
    assert (await login_command(['openai@work'], codex=codex, store=store)).startswith('openai@work connected')
    # A cancelled prompt saves nothing, so nothing is listed.
    assert await login_command(['openai@side'], codex=codex, store=store) == 'Sign-in cancelled.'
    plugins = {'claude': plugin_login()}
    assert await login_command(['claude@work'], codex=codex, plugins=plugins, store=store) == 'Signed in as work.'
    assert await login_command(['claude'], codex=codex, plugins=plugins, store=store) == 'Signed in.'
    assert [item.login for item in accounts(store)] == ['claude@work', 'claude', 'openai@work']
    with pytest.raises(ValueError, match=r'openai-codex@\* means every openai-codex account'):
        await login_command(['openai-codex@*'], codex=codex, store=store)


def keyed_resolver(tmp_path: Path, *, store: SettingsStore | None = None) -> _ModelResolver:
    def resolve_profile(name: str, profile: str) -> Model:
        return TestModel(custom_output_text=f'{profile} answered {name}')

    def resolve(name: str) -> Model:  # pragma: no cover -- only accounts with a profile run here
        return TestModel()

    providers = {'claude-code': ModelProvider(prefix='claude-code', resolve=resolve, resolve_profile=resolve_profile)}
    return _ModelResolver(console=Console(file=io.StringIO()), plugins=lambda: providers, store=store)


async def test_all_accounts_fall_back_in_list_order(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    resolver = keyed_resolver(tmp_path, store=store)
    with pytest.raises(UserError, match=r'No claude-code account is signed in\. Add one with /accounts\.'):
        await resolver.resolve('claude-code@*:opus')
    with pytest.raises(UserError, match='No claude-code account'):
        await keyed_resolver(tmp_path).resolve('claude-code@*:opus')
    remember(store, login='claude', profile='work', plugin=plugin_login())
    single = await resolver.resolve('claude-code@*:opus')
    assert isinstance(single, TestModel) and single.custom_output_text == 'work answered opus'
    remember(store, login='claude', profile='side', plugin=plugin_login())
    store.move_account(provider='claude-code', profile='side', offset=-1)
    model = await resolver.resolve('claude-code@*:opus')
    assert isinstance(model, FallbackModel)
    assert [member.custom_output_text for member in model.models if isinstance(member, TestModel)] == [
        'side answered opus',
        'work answered opus',
    ]
    assert (await Agent(model).run('hi')).output == 'side answered opus'


def pooling_resolver(store: SettingsStore, pooled: Callable[[], bool]) -> _ModelResolver:
    def resolve(name: str) -> Model:
        return TestModel(custom_output_text=f'default answered {name}')

    def resolve_profile(name: str, profile: str) -> Model:
        return TestModel(custom_output_text=f'{profile} answered {name}')

    providers = {'claude-code': ModelProvider(prefix='claude-code', resolve=resolve, resolve_profile=resolve_profile)}
    return _ModelResolver(
        console=Console(file=io.StringIO()), plugins=lambda: providers, store=store, pool_accounts=pooled
    )


def answers(model: Model | str | None) -> list[str | None]:
    members = model.models if isinstance(model, FallbackModel) else [model]
    return [member.custom_output_text for member in members if isinstance(member, TestModel)]


async def test_plain_names_pool_accounts_while_the_setting_is_on(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    pooled = True
    resolver = pooling_resolver(store, lambda: pooled)
    remember(store, login='claude', profile=None, plugin=plugin_login())
    # One signed-in account: the plain name runs on it alone.
    assert answers(await resolver.resolve('claude-code:opus')) == ['default answered opus']
    remember(store, login='claude', profile='work', plugin=plugin_login())
    store.move_account(provider='claude-code', profile='work', offset=-1)
    model = await resolver.resolve('claude-code:opus')
    assert isinstance(model, FallbackModel)
    assert answers(model) == ['work answered opus', 'default answered opus']
    assert (await Agent(model).run('hi')).output == 'work answered opus'
    # `@default` and a named profile each pin one account; `@*` still pools.
    assert answers(await resolver.resolve('claude-code@default:opus')) == ['default answered opus']
    assert answers(await resolver.resolve('claude-code@work:opus')) == ['work answered opus']
    assert len(answers(await resolver.resolve('claude-code@*:opus'))) == 2
    # A provider without accounts, or a name without a provider, is left as it was.
    assert await resolver.resolve('openai:gpt-5') == 'openai:gpt-5'
    assert await resolver.resolve('openai@default:gpt-5') == 'openai:gpt-5'
    assert await resolver.resolve('test') == 'test'
    pooled = False
    assert answers(await resolver.resolve('claude-code:opus')) == ['default answered opus']
    with pytest.raises(UserError, match=r'claude-code@Work:opus: Profile .Work. must be'):
        await resolver.resolve('claude-code@Work:opus')


async def test_set_accounts_pool_applies_to_the_next_run(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    store.plugins_dir.mkdir(parents=True)
    (store.plugins_dir / 'claude_pool.py').write_text("""
from pydantic_ai.models.test import TestModel
from pydantic_clai2.plugins import ModelProvider, Plugin


def resolve(name):
    return TestModel(custom_output_text='default')


def resolve_profile(name, profile):
    return TestModel(custom_output_text=profile)


class ClaudePool(Plugin):
    def get_model_providers(self):
        return (ModelProvider(prefix='claude-code', resolve=resolve, resolve_profile=resolve_profile),)
""")
    remember(store, login='claude', profile=None, plugin=plugin_login())
    remember(store, login='claude', profile='work', plugin=plugin_login())
    shell = create_shell(
        Agent(TestModel()),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=io.StringIO()),
        settings=store.load(),
        store=store,
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )
    await shell.loader.load_all()
    shell.session.model = 'claude-code:opus'
    # The running turn keeps the model it bound, so `/accounts` opens at once instead of queueing.
    assert shell.commands.runs_during_turn('/accounts')
    try:
        assert answers(await shell.session.resolved_model()) == ['default', 'work']
        assert shell.context.set_setting(['accounts.pool', 'false']) == 'Saved accounts.pool. Applied.'
        assert answers(await shell.session.resolved_model()) == ['default']
        assert store.load().pool_accounts is False
    finally:
        await shell.loader.disable('claude_pool')


def test_menu_rows_details_and_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = store_at(tmp_path)
    menu = AccountsMenu(store)
    rows = menu.items()
    assert [row.label for row in rows[:2]] == ['No accounts yet. Add one below.', '+ Add an account...']
    assert is_save_and_close(rows[-1])
    assert 'Sign in to another account.' in menu.details(rows[1])
    assert menu.details(rows[-1]) == SAVE_AND_CLOSE_DETAILS
    save_codex_credentials(account='openai-codex', value=CODEX)
    save_codex_credentials(account='openai-codex@work', value=CODEX)
    remember(store, login='claude', profile='work', plugin=plugin_login())
    store.add_account(StoredAccount(provider='claude-code', profile='old'))
    rows = menu.items()
    assert [row.label for row in rows] == [
        'claude-code',
        '  ● work',
        '  ○ old',
        'openai-codex',
        '  ● default',
        '  ● work',
        '+ Add an account...',
        'Save & close',
    ]
    assert rows[0].disabled and rows[1].description == 'claude@work'
    assert menu.details(rows[0]) == ''  # a heading has nothing to explain
    assert 'Its plugin keeps the sign-in.' in menu.details(rows[1])
    assert 'signed out: Enter signs in (claude-code@old)' in menu.details(rows[2])
    default = menu.details(rows[4])
    assert 'use      openai-codex@default:MODEL' in default and 'this one is number 1.' in default
    monkeypatch.setattr(accounts_menu, 'terminal_size', lambda: (60, 20))
    menu.notice = 'A long notice that wraps across the narrow details pane so it can be read.'
    wrapped = menu.details(rows[4]).split('\n\n')[-1].splitlines()
    assert len(wrapped) > 1 and max(map(len, wrapped)) <= 20 and ' '.join(wrapped) == menu.notice
    menu.notice = None

    fake = FakeMenu()
    work = rows[5]
    assert isinstance(menu.add(fake, work).item, MenuItem)
    for action in (menu.rename, menu.sign_out):
        result = action(fake, work)
        assert result is not None and result.item is not None
    assert menu.rename(fake, rows[0]) is None and menu.sign_out(fake, rows[0]) is None
    menu.move_up(fake, rows[0])  # a heading does not move
    menu.move_down(fake, work)  # already last: nothing moves
    assert fake.redraws == [] and menu._pending == []  # pyright: ignore[reportPrivateUsage]
    menu.move_up(fake, work)
    assert [row.profile for row in store.accounts() if row.provider == 'openai-codex'] == ['work', None]
    assert len(fake.redraws) == 1 and menu.read_key() == Key.UP


def run_keys(
    menu: AccountsMenu, keys: Sequence[str], *, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> tuple[MenuResult, list[str]]:
    frames: list[str] = []
    inputs = iter(keys)

    def read_key() -> str:
        frames.append(unstyled(capsys.readouterr().out) or (frames[-1] if frames else ''))
        return next(inputs)

    monkeypatch.setattr(accounts_menu, 'menu_key', read_key)
    return menu.build().run(), frames


def test_the_cursor_follows_a_moved_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = store_at(tmp_path)
    for profile in ('a', 'b', 'c'):
        save_codex_credentials(account=f'openai-codex@{profile}', value=CODEX)
    menu = AccountsMenu(store)
    menu.focus = 'openai-codex@a'
    result, _ = run_keys(menu, [']', ']', Key.ENTER], monkeypatch=monkeypatch, capsys=capsys)
    assert result.item is not None and isinstance(result.item.value, Account)
    assert result.item.value.login == 'openai-codex@a'
    assert [row.profile for row in store.accounts()] == ['b', 'c', 'a']


def menu_script(lists: list[MenuResult], *, choices: Sequence[MenuResult] = (), texts: Sequence[str] = ()) -> Script:
    return Script(lists=lists, choices=list(choices), texts=[typed(text) for text in texts])


async def test_open_accounts_menu_adds_signs_in_renames_and_signs_out(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    save_codex_credentials(account='openai-codex', value=CODEX)
    logins: list[list[str]] = []
    forgotten: list[str] = []

    async def login(args: list[str]) -> str:
        logins.append(args)
        if args == ['openai-codex@broken']:
            raise UserError('Codex login timed out.')
        save_codex_credentials(account=args[0], value=CODEX)
        remember(store, login='openai-codex', profile=args[0].partition('@')[2] or None)
        return f'Signed in as {args[0]}.'

    default = next(item for item in accounts(store) if item.profile is None)
    codex = LoginChoice(login='openai-codex', provider='openai-codex', kind='subscription', profiles=True)
    script = menu_script(
        [
            pick('add'),
            pick(codex),  # the provider list: a name is suggested, and kept
            pick(default),  # Enter on an account signs in again
            pick('add'),
            pick(codex),
            MenuResult(item=MenuItem('rename', value=accounts_menu._Rename(default))),  # pyright: ignore[reportPrivateUsage]
            MenuResult(item=MenuItem('sign out', value=accounts_menu._SignOut(default))),  # pyright: ignore[reportPrivateUsage]
            MenuResult(item=MenuItem('sign out', value=accounts_menu._SignOut(default))),  # pyright: ignore[reportPrivateUsage]
            pick('add'),
            MenuResult(cancelled=True),  # the provider list closes without a choice
            MenuResult(cancelled=True),
        ],
        choices=[pick(False), pick(True)],
        texts=['account-2', 'broken', 'Personal'],
    )
    rename = accounts_menu._rename  # pyright: ignore[reportPrivateUsage]
    kept = Script(lists=[], choices=[], texts=[TextInputResult(cancelled=True)])
    assert rename(store, default, kept.runners) is None
    result = await open_accounts_menu(store, login=login, plugins=dict, forget=forgotten.append, runners=script.runners)
    assert logins == [['openai-codex@account-2'], ['openai-codex'], ['openai-codex@broken']]
    assert result.splitlines() == [
        'Signed in as openai-codex@account-2.',
        'Signed in as openai-codex.',
        'Codex login timed out.',
        'Signed out of openai-codex.',
    ]
    assert forgotten == ['openai-codex']
    assert [row.label for row in store.accounts()] == [None]  # renamed, then signed out
    assert [item.login for item in accounts(store)] == ['openai-codex@account-2']


async def test_open_accounts_menu_reports_an_account_it_cannot_add(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    remember(store, login='solo', profile=None, plugin=plugin_login(name='solo', profiles=False, models=()))
    solo = LoginChoice(login='solo', provider='solo', kind='plugin', profiles=False)

    async def login(args: list[str]) -> str:
        raise AssertionError('nothing to sign in')  # pragma: no cover

    script = menu_script([pick('add'), pick(solo), MenuResult(cancelled=True)])
    assert await open_accounts_menu(store, login=login, plugins=dict, forget=print, runners=script.runners) == (
        'No changes.'
    )


def test_new_account_names_and_defaults(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    codex = LoginChoice(login='openai-codex', provider='openai-codex', kind='subscription', profiles=True)
    openai = LoginChoice(login='openai', provider='openai', kind='api key', profiles=True)
    # No default account yet: sign in to it, with no name to type.
    assert new_account(store, {}, menu_script([pick(codex)]).runners) == 'openai-codex'
    save_codex_credentials(account='openai-codex', value=CODEX)
    save_codex_credentials(account='openai-codex@account-2', value=CODEX)
    script = menu_script([pick(codex)], texts=['  account-3 '])
    assert new_account(store, {}, script.runners) == 'openai-codex@account-3'
    # An API-key provider's default is the environment, so its first account gets a name.
    assert new_account(store, {}, menu_script([pick(openai)], texts=['team']).runners) == 'openai@team'
    cancelled = Script(lists=[pick(openai)], choices=[], texts=[TextInputResult(cancelled=True)])
    assert new_account(store, {}, cancelled.runners) is None
    assert new_account(store, {}, menu_script([pick('heading')]).runners) is None
    problem = accounts_menu._name_problem  # pyright: ignore[reportPrivateUsage]
    assert problem('account-2', {'account-2'}) == 'account-2 is already an account here.'
    assert problem('Bad', set()) is not None and 'lowercase' in str(problem('Bad', set()))
    assert problem('default', set()) == 'default is the account without a name.'
    assert problem('work', set()) is None
    details = accounts_menu._choice_details  # pyright: ignore[reportPrivateUsage]
    assert 'Models run as openai@NAME:MODEL.' in details(MenuItem('openai', value=openai))
    assert details(MenuItem('API keys', disabled=True)) == ''


def test_choose_account(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    assert choose_account(store, 'openai-codex:gpt-6', menu_script([]).runners) == 'openai-codex:gpt-6'
    save_codex_credentials(account='openai-codex', value=CODEX)
    assert choose_account(store, 'openai-codex:gpt-6', menu_script([]).runners) == 'openai-codex:gpt-6'
    save_codex_credentials(account='openai-codex@work', value=CODEX)
    assert choose_account(store, 'openai-codex@work:gpt-6', menu_script([]).runners) == 'openai-codex@work:gpt-6'
    every = pick('openai-codex@*:gpt-6')
    assert choose_account(store, 'openai-codex:gpt-6', menu_script([], choices=[every]).runners) == (
        'openai-codex@*:gpt-6'
    )
    cancelled = menu_script([], choices=[MenuResult(cancelled=True)])
    assert choose_account(store, 'openai-codex:gpt-6', cancelled.runners) is None
    # One named API-key account: the environment's default is offered too, but not all accounts.
    save_codex_credentials(account='openai@team', value='{}')
    shown: list[list[tuple[str, object]]] = []

    def run_choice(menu: object) -> MenuResult:
        assert isinstance(menu, accounts_menu.Menu)
        shown.append([(row.label, row.value) for row in menu._items])  # pyright: ignore[reportPrivateUsage]
        return pick('openai@default:gpt-5')

    runners = Runners(run_list=menu_script([]).run_list, run_choice=run_choice)
    assert choose_account(store, 'openai:gpt-5', runners) == 'openai@default:gpt-5'
    # Picked accounts are pinned, so pooling plain names does not spread them over the others.
    assert shown == [[('default', 'openai@default:gpt-5'), ('team  (openai@team)', 'openai@team:gpt-5')]]
    assert choose_account(store, 'openai-codex:gpt-6', runners) == 'openai@default:gpt-5'
    assert shown[1][0] == ('default  (openai-codex)', 'openai-codex@default:gpt-6')


def test_model_add_asks_which_account(tmp_path: Path) -> None:
    context, applied = make_context(tmp_path)
    save_codex_credentials(account='openai-codex', value=CODEX)
    save_codex_credentials(account='openai-codex@work', value=CODEX)
    script = Script(
        lists=[pick('openai-codex'), pick('openai-codex:gpt-6-astra'), pick('openai-codex:gpt-6-astra')],
        choices=[MenuResult(cancelled=True), pick('openai-codex@*:gpt-6-astra')],
        texts=[],
    )
    run_model_flow(ModelMenu(context), script.runners)
    # Esc on the account question returns to the models; the second pick runs on every account.
    assert context.settings.model == 'openai-codex@*:gpt-6-astra' and applied == ['model']


def test_a_failed_pool_lists_why_each_account_failed() -> None:
    long_body = 'x' * 500
    group = FallbackExceptionGroup(
        'All models from FallbackModel failed',
        [ModelHTTPError(429, 'gpt-6', body='usage limit'), ModelHTTPError(401, 'gpt-6', body=long_body)],
    )
    lines = error_message(group).splitlines()
    assert lines[0] == 'All models from FallbackModel failed (2 sub-exceptions)'
    assert lines[1].startswith('  1. ModelHTTPError: status_code: 429') and 'usage limit' in lines[1]
    assert lines[2].endswith('...') and len(lines[2]) < 230


async def test_accounts_command_opens_the_menu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = store_at(tmp_path)
    resolver = keyed_resolver(tmp_path, store=store)
    seen: list[SettingsStore] = []

    async def opened(store: SettingsStore, **kwargs: object) -> str:
        seen.append(store)
        return 'No changes.'

    monkeypatch.setattr(accounts_menu, 'open_accounts_menu', opened)
    assert await resolver.accounts([]) == 'No changes.'
    assert seen == [store]
    with pytest.raises(ValueError, match='Usage: /accounts'):
        await resolver.accounts(['list'])


def test_signing_out_drops_the_cached_codex_provider() -> None:
    auth = CodexAuth(Console(file=io.StringIO()))
    work = auth.model('openai-codex@work:gpt-6').provider
    auth.forget('openai-codex@work')
    auth.forget('openai-codex@never')
    assert auth.model('openai-codex@work:gpt-6').provider is not work


async def test_a_chain_can_pool_accounts_before_another_provider(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    remember(store, login='claude', profile='work', plugin=plugin_login())
    remember(store, login='claude', profile='side', plugin=plugin_login())
    store.save_chain(name='best', models=['claude-code@*:opus', 'test'])
    model = await keyed_resolver(tmp_path, store=store).resolve('chain:best')
    assert isinstance(model, FallbackModel)
    pooled, last = model.models
    assert isinstance(pooled, FallbackModel) and len(pooled.models) == 2
    assert isinstance(last, TestModel)
    assert (await Agent(model).run('hi')).output == 'work answered opus'
