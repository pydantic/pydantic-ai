"""A plugin settings menu's credential rows: a `/keys` entry, or a browser sign-in, never in plugin settings.

Each asks through the `Dialogs` it is given, so it works wherever the menu was opened.
Browser sign-ins go through `pydantic_clai2.plugins.sign_in`.
"""

import asyncio
import concurrent.futures
from collections.abc import Callable, Coroutine
from functools import partial
from typing import TypeVar

from anyio import to_thread

from pydantic_clai2.config.api_keys import KeyExistsError, KeyReference, load_keys, save_key
from pydantic_clai2.pkce import PKCESignIn, finish_write
from pydantic_clai2.plugins.menus import Dialogs, Option
from pydantic_clai2.plugins.sign_in import wait_for_sign_in
from pydantic_clai2.ui import telemetry
from pydantic_clai2.ui.menus.menu_worker import worker_stopping

ResultT = TypeVar('ResultT')

_ENTER = 'enter'
"""The picker's "type a new one" choice; saved key names are upper case, so none can be this."""


async def choose_key(
    *, name: str, label: str, dialogs: Dialogs, check: Callable[[str], None] = lambda value: None
) -> KeyReference | None:
    """Pick a saved key, or save a new masked value under `name`; `None` means cancelled.

    `check` raises `ValueError` for a value the service would reject, whether picked or typed. Replacing an
    existing `name` asks first, because every plugin and connection naming that key would change with it.
    """
    with telemetry.span('key prompt', label=label) as span:
        choice = await _ask(name=name, label=label, dialogs=dialogs)
        span.set('answer', _answer(choice))
    if not choice:
        return None
    keys = await to_thread.run_sync(load_keys, abandon_on_cancel=True)
    if isinstance(choice, KeyReference):
        if choice.name in keys:  # Deleted since the picker opened: saving the reference reports it.
            check(keys[choice.name].get_secret_value())
        return choice
    check(choice)
    replace = name in keys
    while True:
        if replace and not await dialogs.confirm(
            title=f'{name} is already in /keys',
            yes=f'Replace {name} for every plugin and connection that uses it',
            no='Keep the saved token',
        ):
            return None
        try:
            await finish_write(asyncio.to_thread(partial(save_key, name=name, value=choice, replace=replace)))
        except KeyExistsError:  # Another session saved `name` since `keys` was read: ask before replacing it.
            replace = True
        else:
            return KeyReference(name=name)


async def _ask(*, name: str, label: str, dialogs: Dialogs) -> KeyReference | str | None:
    """A saved key's name, or a typed value; straight to typing when `/keys` is empty. The picker shows only names."""
    keys = await to_thread.run_sync(load_keys, abandon_on_cancel=True)
    if keys:
        picked = await dialogs.choose(
            title=label,
            options=[
                *(Option(value=saved, label=saved) for saved in sorted(keys)),
                Option(value=_ENTER, label='Enter a different API key'),
            ],
            description='Choose a saved key for this connection. Only key names are displayed.',
        )
        if picked is None:
            return None
        if picked != _ENTER:
            return KeyReference(name=picked)
    return await dialogs.text(
        title=label, label='Token', placeholder=f'Paste the token; it is saved in /keys as {name}', secret=True
    )


def _answer(choice: KeyReference | str | None) -> str:
    """What kind of answer the key prompt got; never the key's value."""
    if choice is None:
        return 'cancelled'
    if isinstance(choice, KeyReference):
        return 'saved key'
    return 'typed' if choice else 'no key'


async def browser_sign_in(session: PKCESignIn, dialogs: Dialogs) -> bool:
    """Sign in through the browser behind a waiting screen showing the URL; whether it finished (backing out cancels).

    The screen closes by itself when the callback arrives.
    Sign-in errors, such as a denial in the browser or a timeout, propagate as `UserError`.
    """
    return await wait_for_sign_in(_ShowsLink(session), dialogs)


class _ShowsLink:
    """A `PKCESignIn` for the waiting screen, always showing its link, not only when no browser opens."""

    def __init__(self, session: PKCESignIn) -> None:
        self.session = session
        self.service = session.service

    async def sign_in(self, *, show: Callable[[str], object]) -> None:
        flow = self.session.start()
        minutes = round(self.session.timeout / 60)
        show(f'Waiting up to {minutes} minutes. If no browser opened, open this URL:\n\n{flow.authorization_url()}')
        await self.session.sign_in(flow, show=lambda url: None)


def on_loop(
    operation: Callable[[], Coroutine[object, object, ResultT]], loop: asyncio.AbstractEventLoop
) -> ResultT | None:
    """Run an async flow from a menu worker thread; `None` when the worker is told to stop first.

    The flow's own widgets watch their own stop signal, so a stopping worker must cancel it explicitly, then wait
    for it to wind down.
    """
    running = asyncio.run_coroutine_threadsafe(operation(), loop)
    # Poll with `wait`, not `result(timeout=)`: its `TimeoutError` is the builtin, which the flow may raise too.
    while not concurrent.futures.wait([running], timeout=0.05).done:
        if worker_stopping():
            running.cancel()
            concurrent.futures.wait([running])  # A started `/keys` write finishes before the menu is released.
            return None
    return running.result()
