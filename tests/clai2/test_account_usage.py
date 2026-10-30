"""Account usage in `/accounts`: Codex and Copilot fetchers, plugin usage, and the background redraw."""

import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import anyio
import httpx2
import pytest
from termflow.tui import MenuItem
from termflow.tui.keys import Key
from termflow.tui.menu import MenuResult

from pydantic_ai.exceptions import UserError
from pydantic_ai.providers.github_copilot import GitHubCopilotCredentials
from pydantic_ai.providers.openai_codex import OpenAICodexCredentials, OpenAICodexProvider
from pydantic_clai2.config.credential_store import has_credentials, save_codex_credentials
from pydantic_clai2.models import github_copilot
from pydantic_clai2.models.accounts import Account, accounts, remember
from pydantic_clai2.models.usage import (
    CODEX_USAGE_URL,
    COPILOT_USAGE_URL,
    UsageFetch,
    codex_usage,
    copilot_usage,
    usage_fetcher,
    window_label,
)
from pydantic_clai2.plugins import AccountUsage, PluginLogin, UsageWindow
from pydantic_clai2.ui.menus import account_usage, accounts_menu
from pydantic_clai2.ui.menus.account_usage import UsageBoard
from pydantic_clai2.ui.menus.accounts_menu import AccountsMenu, open_accounts_menu
from tests.clai2.menu_script import pick
from tests.clai2.test_accounts import CODEX, menu_script, plugin_login, signed, store_at

NOW = datetime(2026, 10, 5, 12, tzinfo=UTC)
RESPONSE = Callable[[httpx2.Request], httpx2.Response]


def account(provider: str, profile: str | None = None, *, plugin_login: str | None = None) -> Account:
    return Account(provider=provider, profile=profile, label=None, plugin_login=plugin_login, signed_in=True)


def codex_provider(respond: RESPONSE) -> OpenAICodexProvider:
    credentials = OpenAICodexCredentials(access_token='access', refresh_token='refresh', account_id='acct')
    return OpenAICodexProvider(credentials, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)))


def test_window_labels_name_hours_or_whole_days() -> None:
    assert [window_label(seconds) for seconds in (18000, 604800, 129600, 30)] == ['5h', '7d', '36h', '1h']


async def test_codex_usage_reads_the_plan_windows_as_the_signed_in_account() -> None:
    seen: list[httpx2.Request] = []
    body: object = {
        'plan_type': 'pro',
        'rate_limit': {
            'primary_window': {'used_percent': 92, 'limit_window_seconds': 18000, 'reset_at': 1791240000},
            'secondary_window': {'used_percent': 43.5, 'limit_window_seconds': 604800},
        },
    }

    def respond(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=body)

    provider = codex_provider(respond)
    assert await codex_usage(provider) == AccountUsage(
        windows=(
            UsageWindow(label='5h', used_percent=92, resets_at=datetime.fromtimestamp(1791240000, UTC)),
            UsageWindow(label='7d', used_percent=43.5),
        ),
        plan='pro',
    )
    [request] = seen
    assert str(request.url) == CODEX_USAGE_URL
    assert request.headers['authorization'] == 'Bearer access' and request.headers['chatgpt-account-id'] == 'acct'
    body = {'plan_type': 'free'}  # no rate limit reported
    assert await codex_usage(provider) == AccountUsage(windows=(), plan='free')
    body = {'rate_limit': {'primary_window': {'used_percent': 'lots'}}}
    with pytest.raises(UserError, match='Codex returned usage CLAI cannot read'):
        await codex_usage(provider)
    with pytest.raises(UserError, match='Codex usage unavailable'):
        await codex_usage(codex_provider(lambda request: httpx2.Response(503, json={'error': 'busy'})))


def save_copilot(account: str = 'github-copilot') -> None:
    credentials = GitHubCopilotCredentials(access_token='gho-token', token_type='bearer', scope='read:user')
    connection = github_copilot.Connection(credentials=credentials, issued_at=time.time())
    save_codex_credentials(account=account, value=connection.model_dump_json())


async def test_copilot_usage_reports_metered_quotas_until_they_reset() -> None:
    save_copilot()
    seen: list[httpx2.Request] = []
    body: dict[str, object] = {
        'copilot_plan': 'individual',
        'quota_reset_date': '2026-11-01',
        'quota_snapshots': {
            'chat': {'percent_remaining': 100, 'unlimited': True},
            'premium_interactions': {'percent_remaining': 75.0, 'unlimited': False},
        },
    }

    def respond(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=body)

    usage = await copilot_usage('github-copilot', transport=httpx2.MockTransport(respond))
    first_of_month = datetime(2026, 11, 1, tzinfo=UTC)
    assert usage == AccountUsage(
        windows=(UsageWindow(label='premium', used_percent=25.0, resets_at=first_of_month),), plan='individual'
    )
    assert str(seen[0].url) == COPILOT_USAGE_URL and seen[0].headers['authorization'] == 'token gho-token'
    body['quota_reset_date_utc'] = '2026-11-01T00:00:00Z'
    body['quota_snapshots'] = {'completions': {'percent_remaining': 10}}
    usage = await copilot_usage('github-copilot', transport=httpx2.MockTransport(respond))
    assert usage.windows == (UsageWindow(label='completions', used_percent=90, resets_at=first_of_month),)
    del body['quota_reset_date_utc'], body['quota_reset_date']
    usage = await copilot_usage('github-copilot', transport=httpx2.MockTransport(respond))
    assert usage.windows[0].resets_at is None


@pytest.mark.parametrize(
    ('status', 'content', 'error'),
    [(401, b'', 'GitHub answered 401'), (200, b'{"quota_snapshots": []}', 'cannot read')],
)
async def test_copilot_usage_reports_why_it_is_unavailable(status: int, content: bytes, error: str) -> None:
    save_copilot()

    def respond(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, content=content)

    with pytest.raises(UserError, match=error):
        await copilot_usage('github-copilot', transport=httpx2.MockTransport(respond))


async def test_copilot_usage_reports_an_unreachable_github() -> None:
    save_copilot('github-copilot@work')

    def unreachable(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError('offline', request=request)

    with pytest.raises(UserError, match='could not reach GitHub'):
        await copilot_usage('github-copilot@work', transport=httpx2.MockTransport(unreachable))


async def test_each_account_maps_to_its_providers_usage() -> None:
    providers: list[str] = []
    profiles: list[str | None] = []

    def codex(login: str) -> OpenAICodexProvider:
        providers.append(login)
        return codex_provider(lambda request: httpx2.Response(200, json={}))

    async def usage(profile: str | None) -> AccountUsage:
        profiles.append(profile)
        return AccountUsage(windows=())

    plugins = {'claude': PluginLogin(name='claude', handler=signed, usage=usage), 'quiet': plugin_login(name='quiet')}

    def fetcher(item: Account) -> UsageFetch | None:
        return usage_fetcher(item, codex=codex, plugins=plugins)

    claude = fetcher(account('claude-code', 'work', plugin_login='claude'))
    assert claude is not None and await claude() == AccountUsage(windows=()) and profiles == ['work']
    codex_fetch = fetcher(account('openai-codex', 'work'))
    assert codex_fetch is not None and await codex_fetch() == AccountUsage(windows=())
    assert providers == ['openai-codex@work']
    assert fetcher(account('quiet', plugin_login='quiet')) is None  # the plugin reports no usage
    assert fetcher(account('gone', plugin_login='unloaded')) is None
    assert fetcher(account('openai', 'work')) is None
    assert fetcher(account('github-copilot')) is not None


async def test_the_board_loads_in_the_background_and_says_how_each_account_stands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = UsageBoard(now=lambda: NOW)
    usage = AccountUsage(
        windows=(
            UsageWindow(label='5h', used_percent=92.4, resets_at=NOW + timedelta(hours=2, minutes=13)),
            UsageWindow(label='7d', used_percent=43, resets_at=NOW + timedelta(days=3, hours=4)),
            UsageWindow(label='7d opus', used_percent=150),
        ),
        plan='pro',
    )
    release = anyio.Event()

    async def slow() -> AccountUsage:
        await release.wait()
        return usage

    async def failing() -> AccountUsage:
        raise UserError('Codex usage unavailable: expired')

    async def broken() -> AccountUsage:
        raise RuntimeError

    fetches: dict[str, UsageFetch] = {'slow': slow, 'failing': failing, 'broken': broken}
    signed_out = Account(provider='x', profile='out', label=None, plugin_login=None, signed_in=False)
    items = [account(name) for name in fetches] + [account('none'), signed_out]
    async with anyio.create_task_group() as tasks:
        board.load(tasks, items, lambda item: fetches.get(item.login))
        loading = [item for item in items if item.login in fetches]
        board.load(tasks, loading, lambda item: pytest.fail('loading accounts are not fetched again'))
        assert board.summary('slow') == '…' and board.details('slow') == ['usage    loading…']
        assert board.summary('none') == '' and board.details('none') == [] and board.summary('x@out') == ''
        release.set()
    assert board.changed.is_set()
    assert board.summary('slow') == '5h 92% · 7d 43%'
    assert board.details('slow') == [
        'usage    pro plan',
        '5h       █████████░  92%',
        '         resets in 2h 13m',
        '7d       ████░░░░░░  43%',
        '         resets in 3d 4h',
        '7d opus  ██████████ 150%',
    ]
    assert board.summary('failing') == '?'
    assert board.details('failing') == ['usage    unavailable', 'Codex usage unavailable: expired']
    assert board.details('broken')[1] == 'RuntimeError'

    async def soon() -> AccountUsage:
        return AccountUsage(windows=(UsageWindow(label='premium', used_percent=1, resets_at=NOW),))

    async def unmetered() -> AccountUsage:
        return AccountUsage(windows=())

    async def hangs() -> AccountUsage:
        await anyio.Event().wait()
        raise AssertionError  # pragma: no cover -- the timeout cancels the wait

    board.forget('slow')
    monkeypatch.setattr(account_usage, 'TIMEOUT', -5)  # time out at once
    async with anyio.create_task_group() as tasks:
        board.load(tasks, [account('hangs')], lambda item: hangs)
    monkeypatch.undo()
    async with anyio.create_task_group() as tasks:
        board.load(tasks, [account('slow')], lambda item: soon)
        board.load(tasks, [account('unmetered')], lambda item: unmetered)
    assert board.details('slow')[-1] == '         resets in 0m'
    assert board.details('unmetered') == ['usage', 'no metered limits']
    assert board.details('hangs') == ['usage    unavailable', 'timed out']


@pytest.mark.parametrize('searching', [False, True])
def test_arriving_usage_redraws_the_open_menu_from_its_own_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], searching: bool
) -> None:
    store = store_at(tmp_path)
    save_codex_credentials(account='openai-codex', value=CODEX)
    board = UsageBoard(now=lambda: NOW)
    board._states['openai-codex'] = None  # pyright: ignore[reportPrivateUsage]
    menu = AccountsMenu(store, board)
    board.changed.set()  # before the menu is built there is nothing to redraw
    monkeypatch.setattr(accounts_menu, 'menu_key', lambda: Key.ESCAPE)
    assert menu.read_key() == Key.ESCAPE and board.changed.is_set()
    board.changed.clear()

    def arrive() -> str:
        board._states['openai-codex'] = AccountUsage(  # pyright: ignore[reportPrivateUsage]
            windows=(UsageWindow(label='7d', used_percent=16),)
        )
        board.changed.set()
        return ''  # the poll times out with no key; the next read swaps in the rows

    # A search for `zz` hides every row while the usage arrives; clearing it shows the new rows.
    before = ['/', 'z', 'z'] if searching else []
    steps = [*[lambda key=key: key for key in before], arrive, lambda: Key.ESCAPE, lambda: Key.ESCAPE]
    if not searching:
        steps.pop()
    script = iter(steps)
    frames: list[str] = []

    def key() -> str:
        frames.append(capsys.readouterr().out)
        return next(script)()

    monkeypatch.setattr(accounts_menu, 'menu_key', key)
    assert menu.build().run().cancelled
    assert '● default  …' in frames[0]
    assert '● default  7d 16%' in ''.join(frames)
    assert not board.changed.is_set()


async def test_open_accounts_menu_loads_usage_and_stops_when_it_closes(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    save_codex_credentials(account='openai-codex', value=CODEX)
    remember(store, login='claude', profile='work', plugin=plugin_login())
    started: list[str] = []

    async def never() -> AccountUsage:
        await anyio.Event().wait()
        raise AssertionError  # pragma: no cover -- closing the menu cancels the fetch

    def usage(item: Account) -> UsageFetch:
        started.append(item.login)
        return never

    async def login(args: list[str]) -> str:
        return f'Signed in as {args[0]}.'

    default = next(item for item in accounts(store) if item.login == 'openai-codex')
    script = menu_script([MenuResult(item=MenuItem('default', value=default)), MenuResult(cancelled=True)])
    with anyio.fail_after(5):
        message = await open_accounts_menu(
            store, login=login, plugins=lambda: {}, forget=lambda _: None, usage=usage, runners=script.runners
        )
    assert message == 'Signed in as openai-codex.'
    # Each account loads once; signing in again fetches that account afresh.
    assert sorted(started) == ['claude@work', 'openai-codex', 'openai-codex']


async def test_stopping_a_fetch_returns_only_once_a_save_it_started_has_finished() -> None:
    board = UsageBoard(now=lambda: NOW)
    events: list[str] = []
    saving = threading.Event()
    release = threading.Event()

    def save() -> None:
        saving.set()
        release.wait()
        events.append('saved refreshed tokens')

    async def refreshing() -> AccountUsage:
        await anyio.to_thread.run_sync(save)
        events.append('fetch ended')
        return AccountUsage(windows=(UsageWindow(label='5h', used_percent=1),))

    async def stop() -> None:
        await board.stop('openai-codex')
        events.append('stopped')

    async with anyio.create_task_group() as tasks:
        board.load(tasks, [account('openai-codex')], lambda item: refreshing)
        await anyio.to_thread.run_sync(saving.wait)
        tasks.start_soon(stop)
        await anyio.sleep(0)  # let `stop` cancel and start waiting
        release.set()
    # A thread cannot be interrupted, so the fetch finishes; but all of it happens before `stop` returns.
    assert events == ['saved refreshed tokens', 'fetch ended', 'stopped']
    assert board.summary('openai-codex') == '' and not board.changed.is_set()


async def test_a_fetch_stopped_before_it_starts_never_runs() -> None:
    board = UsageBoard(now=lambda: NOW)
    ran: list[str] = []

    async def fetch() -> AccountUsage:  # pragma: no cover -- cancelled before it starts
        ran.append('fetched')
        return AccountUsage(windows=())

    async with anyio.create_task_group() as tasks:
        board.load(tasks, [account('github-copilot')], lambda item: fetch)
        await board.stop('github-copilot')
        await board.stop('never-loaded')
    assert ran == [] and board.summary('github-copilot') == ''

    async def done() -> AccountUsage:
        return AccountUsage(windows=())

    async with anyio.create_task_group() as tasks:
        board.load(tasks, [account('github-copilot')], lambda item: done)
    await board.stop('github-copilot')  # already finished: nothing to wait for
    assert board.summary('github-copilot') == ''


async def test_signing_out_stops_that_accounts_fetch_before_deleting_its_login(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    save_codex_credentials(account='openai-codex@work', value=CODEX)
    signed_in_when_stopped: list[bool] = []

    async def holds() -> AccountUsage:
        try:
            await anyio.Event().wait()
        finally:
            signed_in_when_stopped.append(has_credentials(account='openai-codex@work'))
        raise AssertionError  # pragma: no cover -- the wait only ends by cancellation

    async def login(args: list[str]) -> str:  # pragma: no cover -- no sign-in in this script
        return ''

    work = next(item for item in accounts(store) if item.login == 'openai-codex@work')
    sign_out = MenuResult(item=MenuItem('sign out', value=accounts_menu._SignOut(work)))  # pyright: ignore[reportPrivateUsage]
    script = menu_script([sign_out, MenuResult(cancelled=True)], choices=[pick(True)])
    with anyio.fail_after(5):
        message = await open_accounts_menu(
            store,
            login=login,
            plugins=lambda: {},
            forget=lambda _: None,
            usage=lambda item: holds,
            runners=script.runners,
        )
    assert message == 'Signed out of openai-codex@work.'
    assert signed_in_when_stopped == [True], 'the fetch ended while the login still existed'
    assert not has_credentials(account='openai-codex@work')
