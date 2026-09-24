"""Keep MCP OAuth tokens in the OS keyring, so restarting CLAI does not mean signing in again.

FastMCP stores tokens through an `AsyncKeyValue`. This one keeps every entry for a server in one
credential (`mcp-NAME`) using CLAI's credential store: the keyring, chunked for Windows' size limit,
or a private file when no keyring exists. FastMCP keys entries by server URL, so a new URL signs in again.
"""

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import SupportsFloat

from anyio import to_thread
from fastmcp.client.auth import OAuth
from keyring.errors import KeyringError
from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError
from pydantic_ai.exceptions import UserError

from ..credential_store import delete_credentials, load_codex_credentials, save_codex_credentials
from ._settings import RemoteServer

_TOKENS = 'mcp-oauth-token'
"""The collection FastMCP keeps access and refresh tokens in."""


class _Entry(BaseModel):
    value: dict[str, JsonValue]
    expires_at: float | None = None

    def live(self, now: float) -> bool:
        return self.expires_at is None or self.expires_at > now


_WRITES = threading.Lock()
_BUNDLE: TypeAdapter[dict[str, _Entry]] = TypeAdapter(dict[str, _Entry])
Bundle = dict[str, _Entry]


def account(name: str) -> str:
    """The credential account holding one server's tokens."""
    return f'mcp-{name}'


def _slot(collection: str | None, key: str) -> str:
    return f'{collection or "default"}/{key}'


def _load(name: str) -> Bundle:
    try:
        raw = load_codex_credentials(account=account(name))
        return _BUNDLE.validate_json(raw) if raw else {}
    except (UserError, ValidationError, UnicodeDecodeError):
        return {}  # An unreadable bundle means signing in again, not a failed connection.


def _save(name: str, bundle: Bundle) -> None:
    now = time.time()
    live = {slot: entry for slot, entry in bundle.items() if entry.live(now)}
    if live:
        save_codex_credentials(value=_BUNDLE.dump_json(live).decode(), account=account(name))
    else:
        delete_credentials(account=account(name))


class TokenStore:
    """FastMCP's `AsyncKeyValue` protocol over one server's credential. Keyring calls run in a thread."""

    def __init__(self, name: str) -> None:
        """Entries are stored under the `mcp-NAME` credential account."""
        self.name = name

    def signed_in(self) -> bool | None:
        """Whether tokens are stored (they may still need a refresh); `None` when the keyring cannot be read."""
        try:
            bundle = _load(self.name)
        except KeyringError:
            return None
        now = time.time()
        return any(slot.startswith(f'{_TOKENS}/') and entry.live(now) for slot, entry in bundle.items())

    def forget(self) -> None:
        """Sign out: drop the tokens and the registered client."""
        delete_credentials(account=account(self.name))

    async def _update(self, change: Callable[[Bundle], int]) -> int:
        return await to_thread.run_sync(self._update_now, change)

    def _update_now(self, change: Callable[[Bundle], int]) -> int:
        # One lock for every store: FastMCP writes the token and the client entry for a server through
        # separate store instances, and each write rewrites the whole credential.
        with _WRITES:
            bundle = _load(self.name)
            count = change(bundle)
            _save(self.name, bundle)
            return count

    async def ttl_many(
        self, keys: Sequence[str], *, collection: str | None = None
    ) -> list[tuple[dict[str, JsonValue] | None, float | None]]:
        """Values and seconds left, `(None, None)` for missing or expired keys."""
        bundle = await to_thread.run_sync(_load, self.name)
        now = time.time()
        results: list[tuple[dict[str, JsonValue] | None, float | None]] = []
        for key in keys:
            entry = bundle.get(_slot(collection, key))
            if entry is None or not entry.live(now):
                results.append((None, None))
            else:
                results.append((entry.value, None if entry.expires_at is None else entry.expires_at - now))
        return results

    async def ttl(self, key: str, *, collection: str | None = None) -> tuple[dict[str, JsonValue] | None, float | None]:
        """One value and its seconds left."""
        [result] = await self.ttl_many([key], collection=collection)
        return result

    async def get_many(
        self, keys: Sequence[str], *, collection: str | None = None
    ) -> list[dict[str, JsonValue] | None]:
        """Values, `None` for missing or expired keys."""
        return [value for value, _ in await self.ttl_many(keys, collection=collection)]

    async def get(self, key: str, *, collection: str | None = None) -> dict[str, JsonValue] | None:
        """One value."""
        return (await self.ttl(key, collection=collection))[0]

    async def put_many(
        self,
        keys: Sequence[str],
        values: Sequence[Mapping[str, object]],
        *,
        collection: str | None = None,
        ttl: SupportsFloat | None = None,
    ) -> None:
        """Store values, expiring after `ttl` seconds when given."""
        expires_at = None if ttl is None else time.time() + float(ttl)
        entries = {
            _slot(collection, key): _Entry.model_validate({'value': value, 'expires_at': expires_at})
            for key, value in zip(keys, values, strict=True)
        }
        await self._update(lambda bundle: bundle.update(entries) or len(entries))

    async def put(
        self, key: str, value: Mapping[str, object], *, collection: str | None = None, ttl: SupportsFloat | None = None
    ) -> None:
        """Store one value."""
        await self.put_many([key], [value], collection=collection, ttl=ttl)

    async def delete_many(self, keys: Sequence[str], *, collection: str | None = None) -> int:
        """Remove keys; returns how many existed."""
        slots = [_slot(collection, key) for key in keys]
        return await self._update(lambda bundle: sum(bundle.pop(slot, None) is not None for slot in slots))

    async def delete(self, key: str, *, collection: str | None = None) -> bool:
        """Remove one key."""
        return await self.delete_many([key], collection=collection) == 1


def oauth(name: str, server: RemoteServer) -> OAuth | None:
    """A sign-in handler with keyring-backed tokens; FastMCP refreshes them or opens the browser on connect."""
    if not server.auth:
        return None
    return OAuth(client_name='CLAI', callback_host='127.0.0.1', token_storage=TokenStore(name))
