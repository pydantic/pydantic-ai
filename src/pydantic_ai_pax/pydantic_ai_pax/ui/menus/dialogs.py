"""`Dialogs` and `SettingsMenu` in the terminal: termflow widgets, one `run_worker` each, under one hold."""

import asyncio
import textwrap
import threading
from collections.abc import Awaitable, Callable, Sequence
from functools import partial

from anyio import to_thread
from termflow.ansi import make_link
from termflow.tui import MenuBuilder, MenuItem, PagerBuilder, TextInputBuilder, terminal_size
from termflow.tui.pager import Pager

from pydantic_ai_pax.plugins.menus import Action, Option, SettingsMenu
from pydantic_ai_pax.ui.menus.field_menu import TERMINAL, FieldMenu, Runners, run_flow_async
from pydantic_ai_pax.ui.menus.menu_worker import full_terminal, menu_key, run_worker
from pydantic_ai_pax.ui.prompt.prompt_selection import URL, trim_url
from pydantic_ai_pax.ui.rendering.rendering import MAX_LINK, markdown_style
from pydantic_ai_pax.ui.rendering.tool_output import terminal_text

_REDRAW = 'dialogs:changed'
"""A key no keyboard sends and no handler takes: termflow repaints after it, showing the new text."""


class TerminalDialogs:
    """`Dialogs` as full-screen termflow widgets."""

    def __init__(self, runners: Runners = TERMINAL) -> None:
        """`runners` shows the widgets; tests pass scripted ones."""
        self._runners = runners

    async def choose(
        self, *, title: str, options: Sequence[Option], current: str | None = None, description: str = ''
    ) -> str | None:
        """A list of `options`, searchable past eight; a description shows beside it."""
        values = [option.value for option in options]
        builder = (
            MenuBuilder(title)
            .style(markdown_style())
            .items(
                [
                    MenuItem(f'{option.label}{" (current)" if option.value == current else ""}', value=option.value)
                    for option in options
                ]
            )
            .searchable(len(options) > 8)
            .initial_index(values.index(current) if current in values else 0)
            .footer_hint('Enter select - Esc back')
            .key_source(menu_key)
        )
        details = {option.value: option.description for option in options}
        if description or any(details.values()):
            builder.preview(
                lambda item: '\n\n'.join(text for text in (details.get(item.value, ''), description) if text)
            )
        menu = builder.build()
        result = await run_worker(lambda: self._runners.run_choice(menu))
        if result.cancelled or result.item is None or not isinstance(result.item.value, str):
            return None
        return result.item.value

    async def text(
        self,
        *,
        title: str,
        label: str = 'Value',
        placeholder: str = '',
        initial: str = '',
        secret: bool = False,
        check: Callable[[str], str | None] | None = None,
        description: str = '',
        multiline: bool = False,
    ) -> str | None:
        """A text input that checks non-empty text as you type; `secret` masks it, and `description` sits under the title.

        `multiline` opens `Runners.edit` instead, `$EDITOR` or the built-in editor, on `initial`.
        """
        if multiline:
            return await self._edit(title=title, initial=initial, check=check)
        builder = (
            TextInputBuilder(f'{title}\n\n{description}' if description else title)
            .style(markdown_style())
            .prompt(f'{label}: ')
            .placeholder(placeholder)
            .initial(initial)
            .footer_hint('Enter save - Esc cancel')
            .key_source(menu_key)
        )
        if check is not None:
            builder.validator(partial(_checked, check))
        if secret:
            builder.mask()
        widget = builder.build()
        result = await run_worker(lambda: self._runners.run_text(widget))
        if result.cancelled or not isinstance(result.value, str):
            return None
        return result.value.strip()

    async def _edit(self, *, title: str, initial: str, check: Callable[[str], str | None] | None) -> str | None:
        """Edit until `check` accepts the text, reopening it with the reason in the title; `None` when cancelled."""
        text, heading = initial, title
        while True:
            edited = await run_worker(partial(self._runners.edit, text, title=heading))
            if edited is None:
                return None
            text = edited.strip()
            problem = check(text) if check is not None and text else None
            if problem is None:
                return text
            heading = f'{title}: {problem}'

    async def view(self, *, title: str, text: str) -> None:
        """A pager over `text`, wrapped to the terminal, until q or Esc."""
        pager = build_pager(title, text)
        await run_worker(lambda: self._runners.run_pager(pager))

    async def confirm(self, *, title: str, yes: str, no: str, description: str = '') -> bool:
        """Two rows, `no` first; Esc is `no`."""
        builder = (
            MenuBuilder(title)
            .style(markdown_style())
            .items([MenuItem(no, value=False), MenuItem(yes, value=True)])
            .footer_hint('Enter select - Esc back')
            .key_source(menu_key)
        )
        if description:
            builder.preview(lambda item: description)
        menu = builder.build()
        result = await run_worker(lambda: self._runners.run_choice(menu))
        return not result.cancelled and result.item is not None and result.item.value is True

    async def wait(
        self, *, title: str, description: str, task: Callable[[Callable[[str], None]], Awaitable[object]]
    ) -> bool:
        """A screen with one Cancel row and `description`, then the latest text `task` shows; it closes when `task` ends."""
        shown: list[str] = []
        changed = threading.Event()

        def show(text: str) -> None:
            shown[:] = [text]
            changed.set()

        def read_key() -> str:
            # Termflow calls this on the menu's own thread every 50 ms, so a repaint here never races a paint.
            if changed.is_set():
                changed.clear()
                return _REDRAW
            return menu_key()

        screen = (
            MenuBuilder(title)
            .style(markdown_style())
            .items([MenuItem('Cancel', value=None)])
            .preview(lambda item: _linked('\n\n'.join([description, *shown]), width=_preview_width()))
            .footer_hint('Enter or Esc cancel')
            .key_source(read_key)
            .build()
        )
        running = asyncio.ensure_future(task(show))
        waiting = asyncio.ensure_future(run_worker(lambda: self._runners.run_choice(screen)))
        try:
            await asyncio.wait({running, waiting}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            running.cancel()
            waiting.cancel()
            await asyncio.gather(running, waiting, return_exceptions=True)
        # A screen that failed, rather than one the user closed, says why instead of reading as a cancel.
        if not waiting.cancelled() and (failure := waiting.exception()) is not None:
            raise failure
        if running.cancelled():
            return False
        running.result()
        return True


def _preview_width() -> int:
    """The width of the preview pane termflow lays out beside a list of its default width.

    This mirrors `Menu._frame` and `layout.two_columns` in termflow-md 1.0.0, which pass the preview no
    width; `test_wait_folds_urls_to_the_pane_termflow_draws` paints a real frame to catch a change.
    """
    columns = max(10, terminal_size()[0] - 1)
    return max(1, columns - max(20, columns // 2) - 3)


def _linked(text: str, *, width: int) -> str:
    """`text` with each URL a terminal hyperlink to all of it; a line with a URL is folded at `width`.

    The preview pane cuts each line at its width, which would also cut off a link's closing escape, so a
    URL too long for one line shows on several, each a link to the whole URL, as Rich prints one.
    """
    lines: list[str] = []
    for line in text.split('\n'):
        urls = [
            (match.start(), match.start() + len(url), url)
            for match in URL.finditer(line)
            if (url := trim_url(match[0])).isprintable() and len(url) <= MAX_LINK
        ]
        lines += (
            [_linked_row(line, start, start + width, urls) for start in range(0, len(line), width)] if urls else [line]
        )
    return '\n'.join(lines)


def _linked_row(line: str, start: int, end: int, urls: list[tuple[int, int, str]]) -> str:
    """`line[start:end]`, with the parts of `urls` (start, end, URL) in it linked to their whole URL."""
    row, done = '', start
    for first, last, url in urls:
        label_start, label_end = max(first, start), min(last, end)
        if label_start < label_end:
            row += line[done:label_start] + make_link(url, line[label_start:label_end])
            done = label_end
    return row + line[done:end]


def wrap(text: str, width: int) -> list[str]:
    """`text` wrapped to `width`, keeping its line breaks and blank lines.

    Control characters are made inert: the text can come from a repository's `AGENTS.md` or project file.
    """
    return [
        wrapped
        for line in text.splitlines()
        for wrapped in textwrap.wrap(terminal_text(line, keep='\t'), width) or ['']
    ]


def build_pager(title: str, text: str) -> Pager:
    """A scrollable, read-only view of `text`."""
    return (
        PagerBuilder(title)
        .style(markdown_style())
        .reflow(lambda width: wrap(text, width))
        .footer_hint('↑↓ scroll - q close')
        .key_source(menu_key)
        .build()
    )


def _checked(check: Callable[[str], str | None], text: str) -> str | None:
    """Empty text is always accepted: it is how the user says "none"."""
    text = text.strip()
    return check(text) if text else None


RUNNERS: Runners = TERMINAL
"""How `run_settings_menu` shows its widgets when a caller passes no runners; tests swap in scripted ones."""


async def run_settings_menu(menu: SettingsMenu, runners: Runners | None = None) -> list[str]:
    """Show `menu` until Esc or Save & close, with its actions asking through `TerminalDialogs`; the messages to show.

    The live panel stays off the screen for the whole menu, not only while each widget runs, so
    nothing paints between one dialog and the next, even while an action awaits something.
    """
    runners = runners or RUNNERS
    with full_terminal():
        return await _run_menu(menu, runners, TerminalDialogs(runners))


async def _run_menu(menu: SettingsMenu, runners: Runners, dialogs: TerminalDialogs) -> list[str]:
    """One menu's list; a row that opens another menu shows it, then this list again."""

    def acts(action: Action) -> Callable[[], Awaitable[Sequence[str]]]:
        return lambda: action(dialogs)

    async def open_menu(build: Callable[[], SettingsMenu]) -> list[str]:
        # Building it may read files or the keyring, as its rows may.
        return await _run_menu(await to_thread.run_sync(build), runners, dialogs)

    def opens(build: Callable[[], SettingsMenu]) -> Callable[[], Awaitable[Sequence[str]]]:
        return partial(open_menu, build)

    # Building the list reads the rows, which may read the keyring or `/keys`: off the loop, as each step is.
    fields = await to_thread.run_sync(partial(FieldMenu, menu.source, searchable=menu.searchable))
    return await run_flow_async(
        fields,
        runners,
        submenus={
            **{key: acts(action) for key, action in menu.actions.items()},
            **{key: opens(build) for key, build in menu.menus.items()},
        },
    )
