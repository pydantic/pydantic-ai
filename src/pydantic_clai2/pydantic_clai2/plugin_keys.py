"""Choosing a plugin's `/keys` entry from its settings menu, so plugin settings only ever hold the key's name."""

import asyncio
import concurrent.futures

from termflow.tui import MenuBuilder, MenuItem, TextInputBuilder

from ._rendering import markdown_style
from .api_keys import KeyReference, load_keys, prompt_api_key, save_key
from .field_menu import Runners
from .menu_worker import menu_key, run_worker, worker_stopping


class MaskedPrompt:
    """`prompt_api_key`'s value prompt as a masked termflow input, matching the settings menu around it."""

    def __init__(self, *, runners: Runners, placeholder: str) -> None:
        """`placeholder` says what to paste and that it is saved in `/keys`."""
        self._runners = runners
        self._placeholder = placeholder

    async def prompt_async(self, label: str, /, *, is_password: bool = False) -> str:
        """Read a masked value; Esc raises `EOFError`, which `prompt_api_key` reads as cancellation."""
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


async def choose_key(*, name: str, label: str, placeholder: str, runners: Runners) -> KeyReference | None:
    """Pick a saved key, or save a masked new value under `name`; `None` means cancelled or left empty.

    Replacing an existing `name` asks first, because every plugin and connection naming it shares the value.
    """
    choice = await prompt_api_key(prompt=MaskedPrompt(runners=runners, placeholder=placeholder), label=label)
    if choice is None or isinstance(choice, KeyReference):
        return choice
    value = choice.strip()
    if not value:
        return None
    exists = name in await asyncio.to_thread(load_keys)
    if exists and not await run_worker(lambda: _confirm_replace(name, runners)):
        return None
    # Confirmed replacements replace whatever is there; otherwise refuse a key that appeared since the check.
    await asyncio.to_thread(save_key, name=name, value=value, replace=exists)
    return KeyReference(name=name)


def pick_key_from_menu(
    loop: asyncio.AbstractEventLoop, *, name: str, label: str, placeholder: str, runners: Runners
) -> KeyReference | None:
    """`choose_key` from a settings menu's worker thread, handed back to `loop` because the picker is async.

    The picker's widgets watch their own stop signal, so cancelling the menu's worker cancels the picker explicitly
    rather than leaving this thread blocked on it.
    """
    picking = asyncio.run_coroutine_threadsafe(
        choose_key(name=name, label=label, placeholder=placeholder, runners=runners), loop
    )
    while True:
        try:
            return picking.result(timeout=0.05)
        except concurrent.futures.TimeoutError:
            if worker_stopping():
                picking.cancel()
                return None


def _confirm_replace(name: str, runners: Runners) -> bool:
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
