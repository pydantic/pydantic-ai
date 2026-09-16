"""Codex login through core OAuth and an application-owned credential source."""

import asyncio
import webbrowser

from pydantic import TypeAdapter, ValidationError
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.openai_codex import OpenAICodexModel
from pydantic_ai.providers.openai_codex import (
    OpenAICodexCredentials,
    OpenAICodexCredentialSource,
    OpenAICodexOAuthFlow,
    OpenAICodexProvider,
)
from rich.console import Console

from . import theme
from .credential_store import load_codex_credentials, save_codex_credentials

_CREDENTIALS = TypeAdapter(OpenAICodexCredentials)


class CodexCredentials(OpenAICodexCredentialSource):
    """Keep tokens out of SQLite and persist core-managed refreshes in keyring."""

    async def load(self) -> OpenAICodexCredentials:
        """Load credentials without falling back to another application's tokens."""
        value = await asyncio.to_thread(load_codex_credentials)
        if value is None:
            raise UserError('Codex is not connected. Run /login openai-codex.')
        try:
            return _CREDENTIALS.validate_json(value)
        except ValidationError:
            raise UserError('Stored Codex credentials are invalid. Run /login openai-codex.') from None

    async def save(self, credentials: OpenAICodexCredentials) -> None:
        """Persist login or refresh results using the configured OS credential backend."""
        value = _CREDENTIALS.dump_json(credentials).decode()
        await asyncio.to_thread(save_codex_credentials, value=value)


class CodexAuth:
    """Conversation-owned login command and cached native Codex provider."""

    def __init__(self, console: Console) -> None:
        """Defer all credential access until login or a Codex request."""
        self.console = console
        self.source = CodexCredentials()
        self.provider: OpenAICodexProvider | None = None

    async def login(self, args: list[str]) -> str:
        """Run core's authorization-code + PKCE flow with a five-minute timeout."""
        if args not in ([], ['openai-codex']):
            raise ValueError('Usage: /login openai-codex')
        flow = OpenAICodexOAuthFlow()
        self.console.print('Sign in to ChatGPT/Codex in your browser. Waiting up to five minutes.', style=theme.INFO)
        self.console.print(flow.authorization_url(), markup=False, highlight=False)

        # Launching in a thread keeps the loop available for core's callback listener.
        async def open_browser() -> None:
            await asyncio.to_thread(webbrowser.open, flow.authorization_url())

        browser = asyncio.create_task(open_browser())
        try:
            credentials = await asyncio.wait_for(flow.exchange_code_from_callback(), timeout=300)
            await self.source.save(credentials)
            self.provider = None
        except TimeoutError:
            raise UserError('Codex login timed out. Run /login openai-codex to try again.') from None
        finally:
            browser.cancel()
            await asyncio.gather(browser, return_exceptions=True)
        return 'Codex connected. Credentials saved in the OS credential store.'

    def model(self, name: str) -> OpenAICodexModel:
        """Reuse core's provider so it owns refresh and credential persistence."""
        if self.provider is None:
            self.provider = OpenAICodexProvider(credential_source=self.source)
        return OpenAICodexModel(name.removeprefix('openai-codex:'), provider=self.provider)
