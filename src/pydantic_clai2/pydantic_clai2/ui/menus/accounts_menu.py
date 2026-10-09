"""`/accounts`: every signed-in account per provider, added and ordered without typing `@PROFILE`.

The list order is the order `PROVIDER@*:MODEL` tries accounts in. Adding an account picks a
provider, fills in a free name, and signs in; the menu closes for the sign-in, which may open a
browser or ask for a key, and reopens after it.
"""

import textwrap
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from itertools import groupby

from anyio import create_task_group, to_thread
from termflow.tui import MenuBuilder, MenuItem, TextInputBuilder
from termflow.tui.keys import Key
from termflow.tui.menu import Menu, MenuResult
from termflow.tui.terminal import terminal_size

from pydantic_ai.exceptions import UserError
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.models.accounts import Account, LoginChoice, accounts, login_choices, sign_out
from pydantic_clai2.models.profiles import ALL, DEFAULT, check_name, provider_of, with_profile
from pydantic_clai2.models.usage import UsageFetch
from pydantic_clai2.plugins import PluginLogin
from pydantic_clai2.ui.menus.account_usage import UsageBoard
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

_HINT = 'a add · r rename · d sign out · [ ] reorder · enter sign in again · esc close'
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
    while True:
        load_usage(await to_thread.run_sync(accounts, store))
        result = await run_worker(lambda: runners.run_list(menu.build()))
        value = result.item.value if not result.cancelled and result.item is not None else None
        if value == _ADD:
            available = plugins()
            try:
                target = await run_worker(lambda: new_account(store, available, runners))
            except ValueError as exc:
                menu.notice = str(exc)
                continue
        elif isinstance(value, Account):
            target = value.login
        elif isinstance(value, _Rename):
            renamed = value.account
            menu.focus = renamed.login
            menu.notice = await run_worker(lambda: _rename(store, renamed, runners))
            continue
        elif isinstance(value, _SignOut):
            leaving = value.account
            # A usage fetch may refresh this account's tokens; let it end before they are deleted.
            await menu.usage.stop(leaving.login)
            menu.notice = await run_worker(lambda: _sign_out(store, leaving, runners))
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


def new_account(store: SettingsStore, plugins: Mapping[str, PluginLogin], runners: Runners) -> str | None:
    """Pick a provider and a free name; returns the `/login` argument, or `None` when cancelled."""
    choices = login_choices(plugins)
    rows: list[MenuItem] = []
    for kind, group in groupby(choices, key=lambda choice: choice.kind):
        rows.append(MenuItem(_KINDS[kind], disabled=True))
        rows += [MenuItem(f'  {choice.login}', value=choice) for choice in group]
    picked = runners.run_list(
        MenuBuilder('Add an account')
        .style(markdown_style())
        .items(rows)
        .searchable()
        .initial_index(1)
        .list_width(_LIST_WIDTH)
        .preview(_choice_details)
        .footer_hint('type to filter - Enter choose - Esc back')
        .key_source(menu_key)
        .build()
    )
    choice = picked.item.value if not picked.cancelled and picked.item is not None else None
    if not isinstance(choice, LoginChoice):
        return None
    taken = [item for item in accounts(store) if item.provider == choice.provider]
    if choice.has_default and not any(item.profile is None and item.signed_in for item in taken):
        return choice.login
    if not choice.profiles:
        raise ValueError(f'The {choice.login} sign-in has one account, and it is signed in.')
    used = {item.profile for item in taken}
    suggestion = next(
        f'account-{number}' for number in range(len(taken) + 1, len(taken) + 100) if f'account-{number}' not in used
    )
    typed = runners.run_text(
        TextInputBuilder(f'Name the new {choice.provider} account')
        .style(markdown_style())
        .prompt('Name: ')
        .initial(suggestion)
        .validator(lambda text: _name_problem(text.strip(), used))
        .footer_hint('Enter keeps this name - Esc back')
        .key_source(menu_key)
        .build()
    )
    if typed.cancelled or typed.value is None:
        return None
    return f'{choice.login}@{typed.value.strip()}'


_KINDS = {
    'subscription': 'Subscriptions',
    'plugin': 'Plugin sign-ins',
    'connection': 'Connections',
    'api key': 'API keys',
}


def _choice_details(item: MenuItem) -> str:
    choice = item.value
    if not isinstance(choice, LoginChoice):
        return ''
    how = {
        'subscription': 'Signs in through the browser.',
        'plugin': 'Signs in through the plugin.',
        'connection': 'Asks for a server or key, as /model add does.',
        'api key': 'Asks for a key or a /keys entry.\nOther settings come from the environment.',
    }[choice.kind]
    return f'{choice.login}\n\n{how}\nModels run as {choice.provider}@NAME:MODEL.'


def _name_problem(text: str, used: set[str | None]) -> str | None:
    if text in used:
        return f'{text} is already an account here.'
    try:
        check_name(text, kind='Name')
    except ValueError as exc:
        return str(exc)
    return 'default is the account without a name.' if text == 'default' else None


def _rename(store: SettingsStore, item: Account, runners: Runners) -> str | None:
    typed = runners.run_text(
        TextInputBuilder(f'Rename {item.login}')
        .style(markdown_style())
        .prompt('Shown as: ')
        .initial(item.label or '')
        .footer_hint('Enter save (empty shows the profile) - Esc back')
        .key_source(menu_key)
        .build()
    )
    if typed.cancelled or typed.value is None:
        return None
    label = typed.value.strip() or None
    store.rename_account(provider=item.provider, profile=item.profile, label=label)
    return f'{item.login} is shown as {label or item.profile or "default"}.'


def _sign_out(store: SettingsStore, item: Account, runners: Runners) -> str | None:
    confirmation = runners.run_choice(
        MenuBuilder(f'Sign out of {item.login}?')
        .style(markdown_style())
        .items([MenuItem('Keep account', value=False), MenuItem('Sign out', value=True)])
        .preview(lambda row: 'Deletes the saved login. Models\nnaming this account stop working.')
        .footer_hint('Enter select - Esc keep')
        .key_source(menu_key)
        .build()
    )
    if confirmation.cancelled or confirmation.item is None or confirmation.item.value is not True:
        return None
    return sign_out(store, item)


def choose_account(store: SettingsStore, model: str, runners: Runners) -> str | None:
    """Ask which account runs `model` when its provider has more than one; `None` when cancelled.

    Offers each account, the default when it is not listed (an API key from the environment), and
    every account in turn (`PROVIDER@*`) when two or more are signed in. A model already naming an
    account is returned as it is. A picked account is pinned, the default one as `PROVIDER@default`,
    so `accounts.pool` does not spread it over the others.
    """
    provider, _, name = model.partition(':')
    if '@' in provider:
        return model
    options = [item for item in accounts(store) if item.provider == provider_of(model)]
    if not any(item.profile is not None for item in options):
        return model
    rows = [MenuItem(f'{item.name}  ({item.login})', value=item.pinned(name)) for item in options if item.signed_in]
    if not any(item.profile is None for item in options):
        rows.insert(0, MenuItem('default', value=with_profile(model, DEFAULT)))
    if sum(item.signed_in for item in options) > 1:
        rows.append(MenuItem('all accounts, in /accounts order', value=with_profile(model, ALL)))
    result = runners.run_choice(
        MenuBuilder(f'Which {provider} account?')
        .style(markdown_style())
        .items(rows)
        .preview(lambda item: f'Runs as {item.value}' if isinstance(item.value, str) else '')
        .footer_hint('Enter choose - Esc back')
        .key_source(menu_key)
        .build()
    )
    if result.cancelled or result.item is None or not isinstance(result.item.value, str):
        return None
    return result.item.value
