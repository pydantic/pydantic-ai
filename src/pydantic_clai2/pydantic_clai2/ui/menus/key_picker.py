"""Choose a plugin's secret from `/keys` inside its settings menu; only a `KeyReference` comes back.

A plugin's settings menu runs on a menu worker thread, but `prompt_api_key` is a coroutine that opens its
own widgets through `run_worker`. `pick_key` hands the questions back to the event loop and waits, watching
the menu's stop signal so cancelling the menu cancels them too; the one write happens afterwards, on the
menu's own thread.
"""

import asyncio
import concurrent.futures
import threading
from dataclasses import dataclass, field

from anyio import to_thread
from termflow.tui import MenuBuilder, MenuItem, TextInputBuilder

from pydantic_clai2.config.api_keys import KeyReference, load_keys, prompt_api_key, save_key
from pydantic_clai2.ui.menus.field_menu import Runners
from pydantic_clai2.ui.menus.menu_worker import menu_key, run_worker, worker_stopping
from pydantic_clai2.ui.rendering._rendering import markdown_style


class MaskedPrompt:
    """`prompt_api_key`'s value prompt as a masked termflow input, matching the settings menu around it."""

    def __init__(self, runners: Runners) -> None:
        """Show widgets through `runners`, which tests replace with scripted ones."""
        self._runners = runners

    async def prompt_async(self, label: str, /, *, is_password: bool = False) -> str:
        """Ask for the secret; Esc raises `EOFError`, which `prompt_api_key` reads as cancellation."""
        builder = (
            TextInputBuilder(label)
            .style(markdown_style())
            .prompt('Value: ')
            .placeholder('Paste the secret; it is saved in /keys, not in plugin settings')
            .footer_hint('Enter save - Esc cancel')
            .key_source(menu_key)
        )
        builder.mask()
        widget = builder.build()
        result = await run_worker(lambda: self._runners.run_text(widget))
        if result.cancelled or not isinstance(result.value, str):
            raise EOFError
        return result.value


@dataclass(frozen=True)
class NewKey:
    """A value to save under the key's name; `replace` is whether the user agreed to replace a saved one."""

    value: str = field(repr=False)
    replace: bool


async def ask_key(*, name: str, label: str, runners: Runners) -> KeyReference | NewKey | None:
    """Ask for a saved key or a new masked value, writing nothing; `None` means nothing changed."""
    choice = await prompt_api_key(prompt=MaskedPrompt(runners), label=label)
    if choice is None or isinstance(choice, KeyReference):
        return choice
    value = choice.strip()
    if not value:
        return None
    exists = name in await to_thread.run_sync(load_keys, abandon_on_cancel=True)
    if exists and not await run_worker(lambda: _confirm_replace(name, runners)):
        return None
    return NewKey(value, replace=exists)


def pick_key(loop: asyncio.AbstractEventLoop, *, name: str, label: str, runners: Runners) -> KeyReference | None:
    """Ask on `loop` from a menu worker thread, then save any new value here.

    Stopping the worker cancels only the questions. The save runs on this thread after the user has decided,
    so a cancelled menu never leaves a key written without the `KeyReference` the caller records for it.
    """
    finished = threading.Event()

    async def ask() -> KeyReference | NewKey | None:
        try:
            return await ask_key(name=name, label=label, runners=runners)
        finally:
            finished.set()

    asking = asyncio.run_coroutine_threadsafe(ask(), loop)
    while not worker_stopping():
        try:
            decision = asking.result(timeout=0.05)
        except concurrent.futures.TimeoutError:
            continue
        return _save(name, decision)
    asking.cancel()
    # A cancelled future reports done at once; wait until `ask_key` has closed its widgets and released the screen.
    finished.wait()
    return None


def _save(name: str, decision: KeyReference | NewKey | None) -> KeyReference | None:
    if not isinstance(decision, NewKey):
        return decision
    # Without the user's agreement to replace, the save refuses a key another session created since they looked.
    save_key(name=name, value=decision.value, replace=decision.replace)
    return KeyReference(name=name)


def _confirm_replace(name: str, runners: Runners) -> bool:
    menu = (
        MenuBuilder(f'{name} is already in /keys')
        .style(markdown_style())
        .items(
            [
                MenuItem('Keep the saved key', value=False),
                MenuItem(f'Replace {name} for every plugin and connection that uses it', value=True),
            ]
        )
        .footer_hint('Enter select - Esc keep')
        .key_source(menu_key)
        .build()
    )
    pick = runners.run_choice(menu)
    return not pick.cancelled and pick.item is not None and pick.item.value is True
