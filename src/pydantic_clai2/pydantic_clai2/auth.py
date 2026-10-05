"""Subscription login dispatch and Codex OAuth credential management."""

import asyncio
import webbrowser
from collections.abc import Awaitable, Callable, Mapping
from functools import partial
from urllib.parse import parse_qs, urlparse

import anyio
from anyio import fail_after
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from pydantic import TypeAdapter, ValidationError
from rich.console import Console
from termflow.tui import MenuBuilder, MenuItem

from pydantic_ai.exceptions import UserError
from pydantic_ai.models.openai_codex import OpenAICodexModel
from pydantic_ai.providers.openai_codex import (
    OpenAICodexCredentials,
    OpenAICodexCredentialSource,
    OpenAICodexOAuthFlow,
    OpenAICodexProvider,
)
from pydantic_clai2.config.credential_store import credentials_path, load_codex_credentials, save_codex_credentials
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.models import LOGIN_ALIASES, github_copilot, login_names
from pydantic_clai2.plugins import PluginLogin
from pydantic_clai2.ui.menus.field_menu import TERMINAL, Runners
from pydantic_clai2.ui.menus.menu_worker import menu_key, run_worker
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering._rendering import markdown_style

_CREDENTIALS = TypeAdapter(OpenAICodexCredentials)
_PASTE_PROMPT = 'Paste the URL the browser lands on (or finish there): '

ReadLine = Callable[[str], Awaitable[str]]


async def login_command(
    args: list[str],
    *,
    codex: 'CodexAuth',
    plugins: Mapping[str, PluginLogin] | None = None,
    store: SettingsStore | None = None,
    runners: Runners = TERMINAL,
) -> str:
    """`/login NAME` signs in to CLAI's subscriptions or one a plugin adds; bare `/login` asks which.

    A plugin sign-in that succeeds saves its `models` to `store`.
    """
    plugins = plugins or {}
    if len(args) > 1:
        raise ValueError(_login_usage(plugins))
    if args:
        name = LOGIN_ALIASES.get(args[0], args[0])
    elif (picked := await _pick_login(plugins, runners)) is None:
        return ''
    else:
        name = picked
    if name == 'openai-codex':
        return await codex.login([])
    if name == 'github-copilot':
        return await github_copilot.login(console=codex.console)
    if (login := plugins.get(name)) is not None:
        message = await login.handler()
        if store is not None:
            for model in login.models:
                store.add_model(name=model)
        return message
    raise ValueError(_login_usage(plugins))


def _login_usage(plugins: Mapping[str, PluginLogin]) -> str:
    return f'Usage: /login [{"|".join(login_names(plugins))}]'


async def _pick_login(plugins: Mapping[str, PluginLogin], runners: Runners) -> str | None:
    """Ask which sign-in to run; `None` when cancelled."""
    menu = (
        MenuBuilder('Sign in')
        .style(markdown_style())
        .items([MenuItem(name, value=name) for name in login_names(plugins)])
        .footer_hint('Enter sign in - Esc cancel')
        .key_source(menu_key)
        .build()
    )
    result = await run_worker(lambda: runners.run_list(menu))
    if result.cancelled or result.item is None or not isinstance(result.item.value, str):
        return None
    return result.item.value


async def read_line(message: str) -> str:
    """Read one line with a throwaway prompt; logins run between turns, so nothing else owns the terminal.

    Console output while the prompt is open (a failed callback listener) goes above it, not into it.
    """
    with patch_stdout():
        return await PromptSession[str]().prompt_async(message)


def code_from_paste(*, text: str, state: str) -> str:
    """Accept the redirect URL the browser landed on, or the bare authorization code."""
    params = {name: values[0] for name, values in parse_qs(urlparse(text).query).items()}
    if 'code' not in params and 'error' not in params:
        return text
    if params.get('state') != state:
        raise UserError('That URL belongs to a different login attempt. Run /login openai-codex again.')
    if error := params.get('error'):
        raise UserError(f'Authorization failed: {error}')
    return params['code']


class CodexCredentials(OpenAICodexCredentialSource):
    """Keep tokens out of SQLite and persist core-managed refreshes in keyring."""

    async def load(self) -> OpenAICodexCredentials:
        """Load credentials without falling back to another application's tokens."""
        value = await anyio.to_thread.run_sync(load_codex_credentials, abandon_on_cancel=True)
        if value is None:
            raise UserError('Codex is not connected. Run /login openai-codex.')
        try:
            return _CREDENTIALS.validate_json(value)
        except ValidationError:
            raise UserError('Stored Codex credentials are invalid. Run /login openai-codex.') from None

    async def save(self, credentials: OpenAICodexCredentials) -> None:
        """Persist login or refresh results using the configured OS credential backend."""
        value = _CREDENTIALS.dump_json(credentials).decode()
        await anyio.to_thread.run_sync(partial(save_codex_credentials, value=value), abandon_on_cancel=True)


class CodexAuth:
    """Conversation-owned login command and cached native Codex provider."""

    def __init__(self, console: Console, *, read_line: ReadLine = read_line, login_timeout: float = 300) -> None:
        """Defer all credential access until login or a Codex request."""
        self.console = console
        self.read_line = read_line
        self.login_timeout = login_timeout
        self.source = CodexCredentials()
        self.provider: OpenAICodexProvider | None = None

    async def login(self, args: list[str]) -> str:
        """Run core's authorization-code + PKCE flow with a five-minute timeout."""
        if args not in ([], ['openai-codex']):
            raise ValueError('Usage: /login openai-codex')
        flow = OpenAICodexOAuthFlow()
        self.console.print(
            'Sign in to ChatGPT/Codex in your browser. Waiting up to five minutes.', style=theme.color(theme.INFO)
        )
        self.console.print(flow.authorization_url(), markup=False, highlight=False)
        self.console.print(
            'If the browser cannot reach this machine (for example over SSH), paste the URL it ends up on.',
            style=theme.color(theme.MUTED),
        )

        # Launching in a thread keeps the loop available for core's callback listener.
        async def open_browser() -> None:
            await anyio.to_thread.run_sync(webbrowser.open, flow.authorization_url(), abandon_on_cancel=True)

        browser = asyncio.create_task(open_browser())
        try:
            with fail_after(self.login_timeout):
                credentials = await self._receive(flow)
            await self.source.save(credentials)
            self.provider = None
        except TimeoutError:
            raise UserError('Codex login timed out. Run /login openai-codex to try again.') from None
        finally:
            browser.cancel()
            await asyncio.gather(browser, return_exceptions=True)
        # A keyring save removes the file, so its presence means the fallback was used.
        if (path := credentials_path()).exists():
            return f'Codex connected. No OS keyring is available, so credentials are saved in plaintext at {path}.'
        return 'Codex connected. Credentials saved in the OS credential store.'

    async def _receive(self, flow: OpenAICodexOAuthFlow) -> OpenAICodexCredentials:
        """Race the localhost callback against a pasted redirect; the first to succeed wins.

        A listener that cannot bind its port (another login, or a second CLAI) loses the race
        instead of ending it: the paste path exists for exactly that case. Anything else the
        callback reports, such as a denial in the browser, is a real outcome and ends the login.
        """
        callback = asyncio.create_task(flow.exchange_code_from_callback())
        paste = asyncio.create_task(self._exchange_paste(flow))
        pending = {callback, paste}
        try:
            while True:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    if task.exception() is None:
                        return task.result()
                failed = paste if paste in done else callback
                if failed is callback and isinstance(callback.exception(), OSError):
                    self.console.print(
                        f'The local callback is unavailable ({callback.exception()}). Paste the URL instead.',
                        style=theme.color(theme.WARNING),
                        markup=False,
                        highlight=False,
                    )
                    continue
                return failed.result()
        finally:
            callback.cancel()
            paste.cancel()
            await asyncio.gather(callback, paste, return_exceptions=True)

    async def _exchange_paste(self, flow: OpenAICodexOAuthFlow) -> OpenAICodexCredentials:
        """Prompt until something is pasted; Ctrl-C or Ctrl-D abandons the login."""
        try:
            while not (text := (await self.read_line(_PASTE_PROMPT)).strip()):
                pass
        except (KeyboardInterrupt, EOFError):
            raise UserError('Codex login cancelled.') from None
        return await flow.exchange_code(code_from_paste(text=text, state=flow.state))

    def model(self, name: str) -> OpenAICodexModel:
        """Reuse core's provider so it owns refresh and credential persistence."""
        if self.provider is None:
            self.provider = OpenAICodexProvider(credential_source=self.source)
        return OpenAICodexModel(name.removeprefix('openai-codex:'), provider=self.provider)
