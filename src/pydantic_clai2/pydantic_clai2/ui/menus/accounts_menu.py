"""`/accounts`: every signed-in account per provider, added and ordered without typing `@PROFILE`.

The list order is the order `PROVIDER@*:MODEL` tries accounts in. Adding an account picks a
provider, fills in a free name, and signs in; the menu closes for the sign-in, which may open a
browser or ask for a key, and reopens after it.

Adding, renaming, and signing out ask through `Dialogs`, so `accounts_settings_menu` runs the same
steps from the web UI's Settings window.
"""

import textwrap
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import groupby

from anyio import create_task_group, to_thread
from termflow.tui import MenuBuilder, MenuItem
from termflow.tui.keys import Key
from termflow.tui.menu import Menu, MenuResult
from termflow.tui.terminal import terminal_size

from pydantic_ai.exceptions import UserError
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.models.accounts import Account, LoginChoice, accounts, login_choices, sign_out
from pydantic_clai2.models.profiles import ALL, DEFAULT, check_name, provider_of, with_profile
from pydantic_clai2.models.usage import UsageFetch
from pydantic_clai2.plugins import PluginLogin
from pydantic_clai2.plugins.menus import Action, Dialogs, FieldRow, Option, SettingsMenu
from pydantic_clai2.ui.menus.account_usage import UsageBoard
from pydantic_clai2.ui.menus.dialogs import TerminalDialogs
from pydantic_clai2.ui.menus.field_menu import (
    SAVE_AND_CLOSE_DETAILS,
    TERMINAL,
    Runners,
    is_save_and_close,
    save_and_close_item,
)
from pydantic_clai2.ui.menus.menu_worker import menu_key, run_worker
from pydantic_clai2.ui.menus.plugin_menu import Redrawable
from pydantic_clai2.ui.menus.slash_search import slash_search
from pydantic_clai2.ui.rendering._rendering import markdown_style

_HINT = 'Enter sign in again - a add - r rename - d sign out - [ ] reorder - Esc close'
_ADD = 'add'
_REDRAW = 'accounts:usage'
"""A key no keyboard sends and no handler takes: termflow repaints after it, showing new usage."""
_LIST_WIDTH = 46


@dataclass(frozen=True)
class _Rename:
    account: Account


@dataclass(frozen=True)
class _SignOut:
    account: Account


class AccountsMenu:
    """Rows, details, and key actions; `build()` wires them into a termflow menu."""

    def __init__(self, store: SettingsStore, usage: UsageBoard | None = None) -> None:
        """Read accounts from `store` on every redraw, so changes show at once; `usage` fills in as it loads."""
        self._store = store
        self.usage = usage if usage is not None else UsageBoard()
        self._menu: Redrawable | None = None
        """The open menu, once built, so `read_key` can swap in rows with new usage."""
        self.notice: str | None = None
        self._pending: list[str] = []
        """Arrow keys queued for the menu's key reader, so the cursor follows a moved account."""
        self.focus: str | None = None
        """The `/login` name of the account to highlight when the menu opens again."""
        self._accounts = accounts(store)

    def items(self) -> list[MenuItem]:
        """A heading per provider and a row per account, then add, then Save & close."""
        self._accounts = accounts(self._store)
        rows: list[MenuItem] = []
        for provider, group in groupby(self._accounts, key=lambda item: item.provider):
            rows.append(MenuItem(provider, disabled=True))
            for item in group:
                mark = '●' if item.signed_in else '○'
                usage = self.usage.summary(item.login)
                label = f'  {mark} {item.name}' + (f'  {usage}' if usage else '')
                rows.append(MenuItem(label, value=item, description=item.login))
        if not rows:
            rows.append(MenuItem('No accounts yet. Add one below.', disabled=True))
        return [*rows, MenuItem('+ Add an account...', value=_ADD), save_and_close_item()]

    def details(self, item: MenuItem) -> str:
        """The right-hand panel: how to use the account, and its place in `@*`."""
        lines: list[str] = []
        if isinstance(item.value, Account):
            lines = self._account_details(item.value)
        elif item.value == _ADD:
            lines = [
                'Sign in to another account.',
                '',
                'Pick a provider; CLAI fills in a name',
                'you can keep or change.',
            ]
        elif is_save_and_close(item):
            lines = [SAVE_AND_CLOSE_DETAILS]
        if self.notice:
            width = max(20, terminal_size()[0] - _LIST_WIDTH - 4)
            lines += ['', *textwrap.wrap(self.notice, width)]
        return '\n'.join(lines)

    def _account_details(self, item: Account) -> list[str]:
        siblings = [other for other in self._accounts if other.provider == item.provider and other.signed_in]
        lines = [
            f'{item.provider} · {item.name}',
            'signed in' if item.signed_in else f'signed out: Enter signs in ({item.login})',
            '',
            f'use      {item.pinned("MODEL")}',
            f'login    /login {item.login}',
        ]
        if item in siblings and len(siblings) > 1:
            place = siblings.index(item) + 1
            lines += ['', f'{item.provider}@*:MODEL tries {len(siblings)} accounts;', f'this one is number {place}.']
        if usage := self.usage.details(item.login):
            lines += ['', *usage]
        if item.plugin_login is not None:
            lines += ['', 'Its plugin keeps the sign-in.']
        return lines

    def add(self, menu: Redrawable, item: MenuItem) -> MenuResult:
        """A: add an account."""
        return MenuResult(item=MenuItem('add', value=_ADD))

    def rename(self, menu: Redrawable, item: MenuItem) -> MenuResult | None:
        """R: give the highlighted account a name to show."""
        return (
            MenuResult(item=MenuItem('rename', value=_Rename(item.value))) if isinstance(item.value, Account) else None
        )

    def sign_out(self, menu: Redrawable, item: MenuItem) -> MenuResult | None:
        """D: sign out of the highlighted account, after confirmation."""
        if isinstance(item.value, Account):
            return MenuResult(item=MenuItem('sign out', value=_SignOut(item.value)))
        return None

    def move_up(self, menu: Redrawable, item: MenuItem) -> None:
        """[: try this account earlier."""
        self._move(menu, item, -1)

    def move_down(self, menu: Redrawable, item: MenuItem) -> None:
        """]: try this account later."""
        self._move(menu, item, 1)

    def _move(self, menu: Redrawable, item: MenuItem, offset: int) -> None:
        if not isinstance(item.value, Account):
            return
        siblings = [other for other in self._accounts if other.provider == item.value.provider]
        index = siblings.index(item.value)
        if not 0 <= index + offset < len(siblings):
            return
        self._store.move_account(provider=item.value.provider, profile=item.value.profile, offset=offset)
        menu.replace_items(self.items())
        self._pending.append(Key.UP if offset < 0 else Key.DOWN)

    def _focused(self, row: MenuItem) -> bool:
        return isinstance(row.value, Account) and row.value.login == self.focus

    def read_key(self) -> str:
        """A queued arrow first, then a redraw for newly arrived usage, then the terminal.

        Termflow calls this on the menu's own thread, every 50 ms while no key is pressed, so rows
        swapped here never race a paint, and a search that hides every row still gets them.
        """
        if self._pending:
            return self._pending.pop(0)
        if self.usage.changed.is_set() and self._menu is not None:
            self.usage.changed.clear()
            self._menu.replace_items(self.items())
            return _REDRAW
        return menu_key()

    def build(self) -> Menu:
        """Wire rows, details, and keys into a termflow menu."""
        rows = self.items()
        focused = next((index for index, row in enumerate(rows) if self._focused(row)), 1)
        builder = MenuBuilder('Accounts').style(markdown_style()).items(rows).list_width(_LIST_WIDTH)
        builder = builder.preview(self.details).initial_index(focused)
        hotkeys = {
            'a': self.add,
            'r': self.rename,
            'd': self.sign_out,
            '[': self.move_up,
            ']': self.move_down,
        }
        menu = slash_search(builder, footer=_HINT, key_source=self.read_key, hotkeys=hotkeys)
        self._menu = menu
        return menu


Login = Callable[[list[str]], Awaitable[str]]


async def open_accounts_menu(
    store: SettingsStore,
    *,
    login: Login,
    plugins: Callable[[], Mapping[str, PluginLogin]],
    forget: Callable[[str], None],
    usage: Callable[[Account], UsageFetch | None] = lambda _: None,
    runners: Runners = TERMINAL,
) -> str:
    """Show the menu until it closes; sign-ins run between its openings. Returns what changed.

    `login` is `/login`'s handler, `plugins` the loaded plugins' sign-ins, and `forget` drops a
    cached provider once its account signs out. `usage` says how to fetch an account's usage;
    fetches run in the background while the menu is open and stop when it closes.
    """
    menu = await to_thread.run_sync(AccountsMenu, store)
    result = ''
    async with create_task_group() as fetches:
        try:
            result = await _run_menu(
                menu,
                store,
                login=login,
                plugins=plugins,
                forget=forget,
                runners=runners,
                load_usage=lambda items: menu.usage.load(fetches, items, usage),
            )
        finally:
            fetches.cancel_scope.cancel()
    return result


async def _run_menu(
    menu: AccountsMenu,
    store: SettingsStore,
    *,
    login: Login,
    plugins: Callable[[], Mapping[str, PluginLogin]],
    forget: Callable[[str], None],
    runners: Runners,
    load_usage: Callable[[list[Account]], None],
) -> str:
    messages: list[str] = []
    dialogs = TerminalDialogs(runners)
    while True:
        load_usage(await to_thread.run_sync(accounts, store))
        result = await run_worker(lambda: runners.run_list(menu.build()))
        value = result.item.value if not result.cancelled and result.item is not None else None
        if value == _ADD:
            try:
                target = await new_account(store, plugins(), dialogs)
            except ValueError as exc:
                menu.notice = str(exc)
                continue
        elif isinstance(value, Account):
            target = value.login
        elif isinstance(value, _Rename):
            menu.focus = value.account.login
            menu.notice = await rename_account(store, value.account, dialogs) or None
            continue
        elif isinstance(value, _SignOut):
            leaving = value.account
            # A usage fetch may refresh this account's tokens; let it end before they are deleted.
            await menu.usage.stop(leaving.login)
            menu.notice = await sign_out_account(store, leaving, dialogs) or None
            if menu.notice is not None:
                forget(leaving.login)
                messages.append(menu.notice)
            continue
        else:
            return '\n'.join(messages) or 'No changes.'
        if target is None:
            menu.notice = None
            continue
        menu.focus = target
        try:
            menu.notice = await login([target])
        except (UserError, ValueError) as exc:
            menu.notice = str(exc)
        menu.usage.forget(target)
        messages.append(menu.notice)


async def new_account(store: SettingsStore, plugins: Mapping[str, PluginLogin], dialogs: Dialogs) -> str | None:
    """Pick a provider and a free name; returns the `/login` argument, or `None` when cancelled."""
    choices = await to_thread.run_sync(login_choices, plugins)
    login = await dialogs.choose(
        title='Add an account',
        options=[
            Option(value=choice.login, label=choice.login, description=_choice_details(choice)) for choice in choices
        ],
    )
    choice = next((choice for choice in choices if choice.login == login), None)
    if choice is None:
        return None
    taken = [item for item in await to_thread.run_sync(accounts, store) if item.provider == choice.provider]
    if choice.has_default and not any(item.profile is None and item.signed_in for item in taken):
        return choice.login
    if not choice.profiles:
        raise ValueError(f'The {choice.login} sign-in has one account, and it is signed in.')
    used = {item.profile for item in taken}
    suggestion = next(
        f'account-{number}' for number in range(len(taken) + 1, len(taken) + 100) if f'account-{number}' not in used
    )
    name = await dialogs.text(
        title=f'Name the new {choice.provider} account',
        label='Name',
        initial=suggestion,
        check=partial(_name_problem, used=used),
    )
    if not name:
        return None
    # Checked again: the terminal checks as you type, but the page sends what it sends.
    if (problem := _name_problem(name, used=used)) is not None:
        raise ValueError(problem)
    return f'{choice.login}@{name}'


_KINDS = {
    'subscription': 'Subscription',
    'plugin': 'Plugin sign-in',
    'connection': 'Connection',
    'api key': 'API key',
}
_HOW = {
    'subscription': 'Signs in through the browser.',
    'plugin': 'Signs in through the plugin.',
    'connection': 'Asks for a server or key, as /model add does.',
    'api key': 'Asks for a key or a /keys entry. Other settings come from the environment.',
}


def _choice_details(choice: LoginChoice) -> str:
    return f'{_KINDS[choice.kind]}. {_HOW[choice.kind]} Models run as {choice.provider}@NAME:MODEL.'


def _name_problem(text: str, used: set[str | None]) -> str | None:
    if text in used:
        return f'{text} is already an account here.'
    try:
        check_name(text, kind='Name')
    except ValueError as exc:
        return str(exc)
    return 'default is the account without a name.' if text == 'default' else None


async def rename_account(store: SettingsStore, item: Account, dialogs: Dialogs) -> str:
    """Ask for the name to show for `item` and save it; empty when backed out of."""
    label = await dialogs.text(
        title=f'Rename {item.login}', label='Shown as', initial=item.label or '', description='Empty shows the profile.'
    )
    if label is None:
        return ''
    rename = partial(store.rename_account, provider=item.provider, profile=item.profile, label=label or None)
    await to_thread.run_sync(rename)
    return f'{item.login} is shown as {label or item.profile or "default"}.'


async def sign_out_account(store: SettingsStore, item: Account, dialogs: Dialogs) -> str:
    """Sign out of `item` once the user confirms; empty when they keep it."""
    if not await dialogs.confirm(
        title=f'Sign out of {item.login}?',
        yes='Sign out',
        no='Keep account',
        description='Deletes the saved login. Models naming this account stop working.',
    ):
        return ''
    return await to_thread.run_sync(sign_out, store, item)


ADD = 'add'
"""The settings menu's row that adds an account; no account's row key is this."""

SignIn = Callable[[str, Dialogs], Awaitable[str]]
"""`/login`'s handler for one target, such as `openai-codex@work`, asking through the given dialogs."""


def account_key(item: Account) -> str:
    """The account's row in `accounts_settings_menu`: its `/login` name, with `/` (a route separator) as `:`."""
    return item.login.replace('/', ':')


def _summary(item: Account, found: Sequence[Account]) -> str:
    """What the page says about an account: how to use it, and its place among the provider's."""
    lines = [f'Use {item.pinned("MODEL")}.']
    siblings = [other for other in found if other.provider == item.provider and other.signed_in]
    if item in siblings and len(siblings) > 1:
        place = siblings.index(item) + 1
        lines.append(f'{item.provider}@*:MODEL tries {len(siblings)} accounts; this one is number {place}.')
    if item.plugin_login is not None:
        lines.append('Its plugin keeps the sign-in.')
    return ' '.join(lines)


def _state(item: Account) -> str:
    return 'signed in' if item.signed_in else 'signed out'


class AccountsSource:
    """Every account as a row, in the order `@*` tries them, and a row that adds one."""

    title = 'Accounts'

    def __init__(self, store: SettingsStore) -> None:
        """Read accounts from `store` each time the rows are read."""
        self._store = store
        self._found: dict[str, Account] = {}

    def rows(self) -> list[FieldRow]:
        """A row per account, then the row that adds one."""
        found = accounts(self._store)
        self._found = {account_key(item): item for item in found}
        # Its state is its default too: there is nothing to reset an account to.
        rows = [
            FieldRow(
                key=account_key(item),
                label=f'{item.provider} · {item.name}',
                description=_summary(item, found),
                default=_state(item),
            )
            for item in found
        ]
        add = FieldRow(key=ADD, label='Add an account', description='Sign in to another account.', default='')
        return [*rows, add]

    def current(self, row: FieldRow) -> str:
        """Whether the account was signed in when the rows were last read."""
        item = self._found.get(row.key)
        return '' if item is None else _state(item)

    def problem(self, row: FieldRow, text: str) -> str | None:
        """Accounts change through their action, not typed text."""
        return 'Choose the account to change it.'

    def apply(self, row: FieldRow, raw: str) -> str:
        """Accounts change through their action, not typed text."""
        raise ValueError('Choose the account to change it.')

    def reset(self, row: FieldRow) -> str:
        """Signing out asks first, through the account's action."""
        raise ValueError('Choose the account to sign out of it.')


async def _signed_in(sign_in: SignIn, target: str, dialogs: Dialogs) -> list[str]:
    """Sign in to `target`; a failed sign-in is an error the page shows."""
    try:
        message = await sign_in(target, dialogs)
    except (UserError, OSError) as exc:
        raise ValueError(str(exc)) from exc
    return [message] if message else []


async def _add(
    dialogs: Dialogs, *, store: SettingsStore, sign_in: SignIn, plugins: Callable[[], Mapping[str, PluginLogin]]
) -> list[str]:
    target = await new_account(store, plugins(), dialogs)
    return [] if target is None else await _signed_in(sign_in, target, dialogs)


async def _change(
    dialogs: Dialogs, *, key: str, store: SettingsStore, sign_in: SignIn, forget: Callable[[str], None]
) -> list[str]:
    found = await to_thread.run_sync(accounts, store)
    item = next((item for item in found if account_key(item) == key), None)
    if item is None:
        raise LookupError(f'{key} is no longer an account.')
    siblings = [other for other in found if other.provider == item.provider]
    place = siblings.index(item)
    order = f'{item.provider}@*:MODEL tries accounts in this order.'
    options = [
        Option(value='sign-in', label='Sign in again' if item.signed_in else 'Sign in'),
        Option(value='rename', label='Rename...', description='Change the name shown for it.'),
    ]
    if place > 0:
        options.append(Option(value='earlier', label='Try it earlier', description=order))
    if place < len(siblings) - 1:
        options.append(Option(value='later', label='Try it later', description=order))
    leave = 'Remove from the list' if item.plugin_login is not None else 'Sign out...'
    options.append(Option(value='sign-out', label=leave))
    title = f'{item.provider} · {item.name}'
    match await dialogs.choose(title=title, options=options, description=_summary(item, found)):
        case 'sign-in':
            return await _signed_in(sign_in, item.login, dialogs)
        case 'rename':
            message = await rename_account(store, item, dialogs)
        case 'earlier' | 'later' as direction:
            offset = -1 if direction == 'earlier' else 1
            await to_thread.run_sync(
                partial(store.move_account, provider=item.provider, profile=item.profile, offset=offset)
            )
            message = f'{item.provider}@*:MODEL now tries {item.login} {direction}.'
        case 'sign-out':
            message = await sign_out_account(store, item, dialogs)
            if message:
                forget(item.login)
        case _:
            message = ''
    return [message] if message else []


def accounts_settings_menu(
    store: SettingsStore,
    *,
    sign_in: SignIn,
    plugins: Callable[[], Mapping[str, PluginLogin]],
    forget: Callable[[str], None],
) -> SettingsMenu:
    """`/accounts` as a declared menu: a row per account to sign in, rename, reorder, or sign out, and one to add one.

    `sign_in` is `/login`'s handler for one target, `plugins` the loaded plugins' sign-ins, and `forget`
    drops a cached provider once its account signs out. Reads the accounts, so call it off the loop.
    """
    change = partial(_change, store=store, sign_in=sign_in, forget=forget)
    actions: dict[str, Action] = {account_key(item): partial(change, key=account_key(item)) for item in accounts(store)}
    actions[ADD] = partial(_add, store=store, sign_in=sign_in, plugins=plugins)
    return SettingsMenu(source=AccountsSource(store), actions=actions, searchable=False)


def choose_account(store: SettingsStore, model: str, runners: Runners) -> str | None:
    """Ask which account runs `model` when its provider has more than one; `None` when cancelled.

    Offers each account, the default when it is not listed (an API key from the environment), and
    every account in turn (`PROVIDER@*`) when two or more are signed in. A model already naming an
    account is returned as it is. A picked account is pinned, the default one as `PROVIDER@default`,
    so `accounts.pool` does not spread it over the others.
    """
    options = account_options(store, model)
    if not options:
        return model
    result = runners.run_choice(
        MenuBuilder(f'Which {model.partition(":")[0]} account?')
        .style(markdown_style())
        .items([MenuItem(option.label, value=option.value) for option in options])
        .preview(lambda item: f'Runs as {item.value}')
        .footer_hint('Enter choose - Esc back')
        .key_source(menu_key)
        .build()
    )
    if result.cancelled or result.item is None or not isinstance(result.item.value, str):
        return None
    return result.item.value


def account_options(store: SettingsStore, model: str) -> list[Option]:
    """The accounts `choose_account` offers to run `model`, each the model name it picks; none when there is no choice."""
    provider, _, name = model.partition(':')
    if '@' in provider:
        return []
    found = [item for item in accounts(store) if item.provider == provider_of(model)]
    if not any(item.profile is not None for item in found):
        return []
    choices = [(item.pinned(name), f'{item.name}  ({item.login})') for item in found if item.signed_in]
    if not any(item.profile is None for item in found):
        choices.insert(0, (with_profile(model, DEFAULT), 'default'))
    if sum(item.signed_in for item in found) > 1:
        choices.append((with_profile(model, ALL), 'all accounts, in /accounts order'))
    return [Option(value=value, label=label, description=f'Runs as {value}') for value, label in choices]
