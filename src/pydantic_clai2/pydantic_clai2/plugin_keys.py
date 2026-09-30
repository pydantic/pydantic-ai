"""A plugin settings menu's credential rows: a `/keys` entry, or a browser sign-in, never in plugin settings."""

import asyncio
import concurrent.futures
from collections.abc import Callable, Coroutine
from functools import partial
from typing import TypeVar

from termflow.tui import MenuBuilder, MenuItem, TextInputBuilder

from ._rendering import markdown_style
from .api_keys import KeyExistsError, KeyReference, load_keys, prompt_api_key, save_key
from .field_menu import Runners
from .menu_worker import menu_key, run_worker, worker_stopping
from .pkce import PKCESignIn, finish_write

ResultT = TypeVar('ResultT')


class MaskedPrompt:
    """`prompt_api_key`'s value prompt as a masked Termflow input, so it matches the settings menu around it."""

    def __init__(self, runners: Runners, *, placeholder: str) -> None:
        """`runners` shows the widget; tests pass scripted ones."""
        self._runners = runners
        self._placeholder = placeholder

    async def prompt_async(self, label: str, /, *, is_password: bool = False) -> str:
        """Read one masked value; Esc raises `EOFError`, which `prompt_api_key` reads as cancellation."""
        builder = (
            TextInputBuilder(label)
            .style(markdown_style())
            .prompt('Token: ')
            .placeholder(self._placeholder)
            .footer_hint('Enter save - Esc cancel')
            .key_source(menu_key)
        )
        builder.mask()
        widget = builder.build()
        result = await run_worker(lambda: self._runners.run_text(widget))
        if result.cancelled or not isinstance(result.value, str):
            raise EOFError
        return result.value


async def choose_key(
    *, name: str, label: str, runners: Runners, check: Callable[[str], None] = lambda value: None
) -> KeyReference | None:
    """Pick a saved key, or save a new masked value under `name`; `None` means cancelled.

    `check` raises `ValueError` for a value the service would reject, whether picked or typed. Replacing an
    existing `name` asks first, because every plugin and connection naming that key would change with it.
    """
    placeholder = f'Paste the token; it is saved in /keys as {name}'
    choice = await prompt_api_key(prompt=MaskedPrompt(runners, placeholder=placeholder), label=label)
    if choice is None:
        return None
    keys = await asyncio.to_thread(load_keys)
    if isinstance(choice, KeyReference):
        if choice.name in keys:  # Deleted since the picker opened: saving the reference reports it.
            check(keys[choice.name].get_secret_value())
        return choice
    value = choice.strip()
    if not value:
        return None
    check(value)
    replace = name in keys
    while True:
        if replace and not await run_worker(lambda: confirm_replace(name, runners)):
            return None
        try:
            await finish_write(asyncio.to_thread(partial(save_key, name=name, value=value, replace=replace)))
        except KeyExistsError:  # Another session saved `name` since `keys` was read: ask before replacing it.
            replace = True
        else:
            return KeyReference(name=name)


def confirm_replace(name: str, runners: Runners) -> bool:
    """Ask before overwriting a key other plugins may share; Esc keeps it."""
    menu = (
        MenuBuilder(f'{name} is already in /keys')
        .style(markdown_style())
        .items(
            [
                MenuItem('Keep the saved token', value=False),
                MenuItem(f'Replace {name} for every plugin and connection that uses it', value=True),
            ]
        )
        .footer_hint('Enter select - Esc keep')
        .key_source(menu_key)
        .build()
    )
    pick = runners.run_choice(menu)
    return not pick.cancelled and pick.item is not None and pick.item.value is True


async def browser_sign_in(session: PKCESignIn, runners: Runners) -> bool:
    """Sign in through the browser behind a waiting screen showing the URL; whether it finished (Esc cancels).

    The screen closes by itself when the callback arrives: cancelling `run_worker` stops its widget.
    Sign-in errors, such as a denial in the browser or a timeout, propagate as `UserError`.
    """
    flow = session.start()
    url = flow.authorization_url()
    minutes = round(session.timeout / 60)
    screen = (
        MenuBuilder(f'Finish signing in to {session.service} in your browser')
        .style(markdown_style())
        .items([MenuItem('Cancel sign-in', value=None)])
        .preview(lambda item: f'Waiting up to {minutes} minutes. If no browser opened, open this URL:\n\n{url}')
        .footer_hint('Enter or Esc cancel')
        .key_source(menu_key)
        .build()
    )
    signing = asyncio.ensure_future(session.sign_in(flow, show=lambda url: None))
    waiting = asyncio.ensure_future(run_worker(lambda: runners.run_choice(screen)))
    try:
        await asyncio.wait({signing, waiting}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        signing.cancel()
        waiting.cancel()
        await asyncio.gather(signing, waiting, return_exceptions=True)
    if signing.cancelled():
        return False
    signing.result()
    return True


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
