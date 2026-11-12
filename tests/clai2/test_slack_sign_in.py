"""The `slack` plugin's browser sign-in: set up the user's CLAI Slack app, sign in with PKCE, renew each turn."""

import json
import time
import webbrowser
from urllib.parse import parse_qs, urlsplit

import anyio
import pytest
from pydantic import SecretStr

from pydantic_ai.exceptions import UserError
from pydantic_clai2 import slack_app
from pydantic_clai2.builtin_plugins import slack as slack_plugin
from pydantic_clai2.config.credential_store import load_codex_credentials, save_codex_credentials
from pydantic_clai2.pkce import PKCEFlow, PKCESignIn, Tokens
from tests.clai2.dialog_script import Asked, Back, Finish, Type
from tests.clai2.menu_script import UNTIL_CLOSED, pick, typed
from tests.clai2.slack_shell import Shell, script, shell

pytestmark = pytest.mark.anyio

APP = '4717091972327.12139745453623'


class Browser:
    """Stands in for the user's browser: records what was opened, and whether it opened.

    Reloading the plugin imports its module again, so set this up after any reload.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, opens: bool | None = True) -> None:
        self.opened: list[str] = []

        def open_url(url: str) -> bool:
            self.opened.append(url)
            if opens is None:
                raise webbrowser.Error('no browser')
            return opens

        monkeypatch.setattr(slack_plugin, 'open_browser', open_url)


class SlackAccount:
    """Stands in for Slack's sign-in page: saves tokens the way `PKCESignIn.sign_in` does, or fails, or waits."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, *, answer: str | UserError = 'xoxe.xoxp-signed', waits: bool = False
    ) -> None:
        """`waits` leaves the page open until the sign-in is cancelled."""
        self.scopes: list[tuple[str, ...]] = []
        self.cancelled = False

        async def sign_in(session: PKCESignIn, flow: PKCEFlow | None = None, *, show: object = None) -> Tokens:
            assert flow is not None
            assert parse_qs(urlsplit(flow.authorization_url()).query)['client_id'] == [session.client.client_id]
            self.scopes.append(session.client.scopes)
            if waits:
                try:
                    await anyio.sleep_forever()
                finally:
                    self.cancelled = True
            if isinstance(answer, UserError):
                raise answer
            signed = signed_in_tokens(answer, client_id=session.client.client_id, scopes=session.client.scopes)
            save_codex_credentials(account=slack_app.ACCOUNT, value=signed.stored())
            return signed

        monkeypatch.setattr(PKCESignIn, 'sign_in', sign_in)


def signed_in_tokens(
    access: str = 'xoxe.xoxp-signed', *, client_id: str = APP, scopes: tuple[str, ...] = slack_app.READ_SCOPES
) -> Tokens:
    return Tokens(
        client_id=client_id,
        access_token=SecretStr(access),
        refresh_token=SecretStr('xoxe-1-refresh'),
        expires_at=time.time() + 43200,
        scopes=scopes,
    )


CLIENT_ID = Asked(
    kind='text',
    title="Paste your CLAI Slack app's Client ID (Basic Information > App Credentials)",
    label='Client ID',
    placeholder='Looks like 1234567890.1234567890',
    description=f'To create the app, open this URL, pick a workspace, and click Create:\n\n{slack_app.create_app_url()}',
)


async def browser_shell(*, signed_in: bool = False, client_id: str | None = APP) -> Shell:
    app = await shell()
    app.host().save_settings(slack_plugin.SlackSettings(auth='browser', client_id=client_id))
    if signed_in:
        save_codex_credentials(account=slack_app.ACCOUNT, value=signed_in_tokens().stored())
    await app.plugins.reload('slack')
    return app


async def test_setting_up_the_app_opens_its_manifest_then_signs_in_behind_a_waiting_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In the terminal: choose browser sign-in, then the Slack app row, answering the Client ID and the waiting screen."""
    browser = Browser(monkeypatch)
    account = SlackAccount(monkeypatch)
    script(
        monkeypatch,
        lists=[pick('auth'), pick('client_id')],
        choices=[pick('browser'), UNTIL_CLOSED],
        texts=[typed(f'  {APP} ')],
    )
    app = await shell()
    message = await app.plugins.command(['configure', 'slack'])
    assert message.splitlines() == [
        'Saved Sign-in.',
        'Saved Slack app.',
        'Signed in to Slack. Tokens are kept in the OS credential store and renew themselves.',
    ]
    [create_app] = browser.opened
    assert create_app == slack_app.create_app_url()
    assert app.saved() == {'auth': 'browser', 'client_id': APP, 'read_only': True, 'include_instructions': True}
    assert account.scopes == [slack_app.READ_SCOPES]
    assert await app.turn_token() == 'xoxe.xoxp-signed'
    assert [row.key for row in app.source().rows()] == [
        'auth',
        'client_id',
        'account',
        'read_only',
        'include_instructions',
    ]
    await app.plugins.close('exit')


def test_the_manifest_makes_a_public_mcp_client_with_every_scope_slack_documents() -> None:
    [manifest] = parse_qs(urlsplit(slack_app.create_app_url()).query)['manifest_json']
    config = json.loads(manifest)
    assert config['settings']['is_mcp_enabled'] is True  # Without it Slack's MCP server answers 400.
    assert config['settings']['token_rotation_enabled'] is True
    assert config['oauth_config']['pkce_enabled'] is True  # A public client: no client secret anywhere.
    assert config['oauth_config']['redirect_urls'] == [slack_app.REDIRECT_URI]
    assert config['oauth_config']['scopes'] == {'user': [*slack_app.READ_SCOPES, *slack_app.WRITE_SCOPES]}
    assert 'bot' not in config['oauth_config']['scopes']


@pytest.mark.parametrize('opens', [True, False, None])
async def test_setting_up_the_app_asks_for_the_client_id_with_the_create_link_then_signs_in(
    monkeypatch: pytest.MonkeyPatch, opens: bool | None
) -> None:
    """As the web UI runs it: the link to create the app is in the dialog, whether or not a browser opened."""
    SlackAccount(monkeypatch)
    app = await browser_shell(client_id=None)
    browser = Browser(monkeypatch, opens=opens)
    messages, dialogs = await app.act('client_id', [Type(text=f'  {APP} '), Finish()])
    assert messages == [
        'Saved Slack app.',
        'Signed in to Slack. Tokens are kept in the OS credential store and renew themselves.',
    ]
    assert browser.opened == [slack_app.create_app_url()]
    client_id, waiting = dialogs.asked
    assert client_id == CLIENT_ID
    assert (waiting.kind, waiting.title) == ('wait', 'Signing in to Slack')
    [shown] = waiting.shown
    assert 'https://slack.com/oauth/v2_user/authorize?' in shown
    assert app.source().settings.client_id == APP
    assert await app.turn_token() == 'xoxe.xoxp-signed'
    await app.plugins.close('exit')


@pytest.mark.parametrize('answer', [Back(), Type(text='  ')])
async def test_a_cancelled_or_empty_client_id_changes_nothing(
    monkeypatch: pytest.MonkeyPatch, answer: Back | Type
) -> None:
    account = SlackAccount(monkeypatch)
    app = await browser_shell(client_id=None)
    Browser(monkeypatch)
    messages, _ = await app.act('client_id', [answer])
    assert messages == []
    assert app.source().settings.client_id is None
    assert account.scopes == []
    await app.plugins.close('exit')


async def test_the_client_id_dialog_refuses_what_is_not_a_client_id(monkeypatch: pytest.MonkeyPatch) -> None:
    app = await browser_shell(client_id=None)
    Browser(monkeypatch)
    with pytest.raises(AssertionError, match=r"would not accept 'xoxp-not-an-id': String should match pattern"):
        await app.act('client_id', [Type(text='xoxp-not-an-id')])
    assert app.source().settings.client_id is None
    await app.plugins.close('exit')


async def test_backing_out_of_the_waiting_screen_stops_the_sign_in(monkeypatch: pytest.MonkeyPatch) -> None:
    account = SlackAccount(monkeypatch, waits=True)
    app = await browser_shell()
    messages, _ = await app.act('account', [Back()])
    assert messages == ['Slack sign-in cancelled.']
    assert account.cancelled
    assert load_codex_credentials(account=slack_app.ACCOUNT) is None
    await app.plugins.close('exit')


async def test_a_refused_sign_in_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    SlackAccount(monkeypatch, answer=UserError('Authorization failed: access_denied'))
    app = await browser_shell()
    messages, _ = await app.act('account', [Finish()])
    assert messages == ['Authorization failed: access_denied']
    await app.plugins.close('exit')


async def test_allowing_writes_without_signing_in_again_fails_closed_with_the_fix() -> None:
    app = await browser_shell(signed_in=True)  # A read-only sign-in.
    assert await app.turn_token() == 'xoxe.xoxp-signed'
    app.host().save_settings(slack_plugin.SlackSettings(auth='browser', client_id=APP, read_only=False))
    await app.plugins.reload('slack')
    assert await app.turn_token() is None, 'a token that cannot post must not back the write tools'
    assert (
        'Slack tools are off. The Slack sign-in does not grant what this setup needs. '
        'Run /plugins configure slack to sign in again.'
    ) in app.output.getvalue()
    await app.plugins.close('exit')


async def test_signing_in_again_asks_for_write_scopes_once_writing_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    account = SlackAccount(monkeypatch, answer='xoxe.xoxp-writer')
    app = await browser_shell(signed_in=True)
    script(monkeypatch, lists=[pick('read_only'), pick('account')], choices=[pick('false'), UNTIL_CLOSED])
    assert (await app.plugins.command(['configure', 'slack'])).splitlines() == [
        'Saved Tools. Sign in again (Browser sign-in row) so Slack grants the write scopes.',
        'Signed in to Slack. Tokens are kept in the OS credential store and renew themselves.',
    ]
    assert account.scopes == [(*slack_app.READ_SCOPES, *slack_app.WRITE_SCOPES)]
    assert await app.turn_token() == 'xoxe.xoxp-writer'
    await app.plugins.close('exit')


async def test_changing_the_client_id_does_not_create_another_app(monkeypatch: pytest.MonkeyPatch) -> None:
    account = SlackAccount(monkeypatch)
    app = await browser_shell(signed_in=True)
    browser = Browser(monkeypatch)
    messages, _ = await app.act('client_id', [Type(text='111.222'), Finish()])
    assert messages[-1].startswith('Signed in to Slack.')
    assert browser.opened == []
    assert app.source().settings.client_id == '111.222'
    assert account.scopes == [slack_app.READ_SCOPES]
    await app.plugins.close('exit')


async def test_rows_show_what_the_browser_sign_in_needs_next() -> None:
    app = await browser_shell()
    source = app.source()
    _, client_id, account, *_ = source.rows()
    assert (source.current(client_id), client_id.note) == (APP, 'sign in: Enter on Browser sign-in')
    assert source.current(account) == 'signed out'
    save_codex_credentials(account=slack_app.ACCOUNT, value=signed_in_tokens().stored())
    _, client_id, account, *_ = source.rows()
    assert (source.current(client_id), client_id.note, source.current(account)) == (APP, '', 'signed in')
    app.host().save_settings(slack_plugin.SlackSettings(auth='browser'))
    _, client_id, account, *_ = source.rows()
    assert (source.current(client_id), client_id.note) == ('(not set up)', 'Enter to set up')
    assert source.current(account) == 'signed out'
    await app.plugins.close('exit')


async def test_signing_out_with_no_app_set_up_just_says_so() -> None:
    app = await shell()
    app.host().save_settings(slack_plugin.SlackSettings(auth='browser'))
    source = app.source()
    _, _, account, *_ = source.rows()
    assert source.reset(account) == 'Signed out of Slack; its tools are off until you sign in again.'
    assert source.settings == slack_plugin.SlackSettings(auth='browser')
    await app.plugins.close('exit')


async def test_signing_in_needs_the_app_first() -> None:
    app = await shell()
    app.host().save_settings(slack_plugin.SlackSettings(auth='browser'))
    messages, _ = await app.act('account', [])
    assert messages == ['Set up the Slack app first (Slack app row).']
    await app.plugins.close('exit')


async def test_r_signs_out_or_forgets_the_app_and_turns_fail_closed() -> None:
    app = await browser_shell(signed_in=True)
    assert await app.turn_token() == 'xoxe.xoxp-signed'
    source = app.source()
    _, client_id, account, *_ = source.rows()
    assert source.reset(account) == 'Signed out of Slack; its tools are off until you sign in again.'
    assert load_codex_credentials(account=slack_app.ACCOUNT) is None
    assert await app.turn_token() is None
    assert 'Slack tools are off. Not signed in to Slack. Run /plugins configure slack to sign in.' in (
        app.output.getvalue()
    )
    save_codex_credentials(account=slack_app.ACCOUNT, value=signed_in_tokens().stored())
    assert source.reset(client_id) == 'Reset Slack app.'
    assert load_codex_credentials(account=slack_app.ACCOUNT) is None
    assert source.settings.client_id is None
    assert source.reset(client_id) == 'Reset Slack app.'  # Nothing left to sign out of.
    await app.plugins.reload('slack')
    assert await app.turn_token() is None
    assert 'Set up your Slack app' not in app.output.getvalue(), 'no app yet is not a failure worth a warning'
    await app.plugins.close('exit')


async def test_a_sign_in_for_another_app_is_not_used() -> None:
    app = await browser_shell()
    save_codex_credentials(account=slack_app.ACCOUNT, value=signed_in_tokens(client_id='1.2').stored())
    assert await app.turn_token() is None
    assert not app.source().session() or not app.source().session().signed_in()  # pyright: ignore[reportOptionalMemberAccess]
    await app.plugins.close('exit')
