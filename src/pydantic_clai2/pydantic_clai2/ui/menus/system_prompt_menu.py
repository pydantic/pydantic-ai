"""`/system_prompt`: see the instructions the model receives, and edit your own.

Your instructions are the `run.instructions` setting. CLAI sends them after its built-in instructions,
the repository's AGENTS.md, and plugin instructions, which this menu shows but cannot change: yours add
to them and replace none. Clearing yours restores CLAI's defaults.
"""

import shlex
import textwrap
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Protocol

from termflow.tui import MenuBuilder, MenuItem, PagerBuilder
from termflow.tui.menu import Menu
from termflow.tui.pager import Pager
from termflow.tui.terminal import terminal_size

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, SystemPromptPart
from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.ui.menus.field_menu import (
    SAVE_AND_CLOSE_DETAILS,
    TERMINAL,
    Runners,
    is_save_and_close,
    picked,
    save_and_close_item,
)
from pydantic_clai2.ui.menus.menu_worker import menu_key, run_worker
from pydantic_clai2.ui.menus.slash_search import slash_search
from pydantic_clai2.ui.menus.text_editor import BUILT_IN_KEYS, edit_text, external_editor
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering._rendering import markdown_style
from pydantic_clai2.ui.rendering.tool_output import terminal_text

_KEY = 'run.instructions'
_LIST_WIDTH = 34
_ADDED = (
    "Sent after CLAI's built-in instructions, AGENTS.md, and plugin instructions, "
    'which stay as they are. Changes apply from your next prompt.'
)


class Action(Enum):
    """A row of the menu; its value is the row's label."""

    EDIT = 'Edit your instructions'
    APPEND = 'Append to your instructions'
    RESET = 'Reset: remove your instructions'
    VIEW = 'View the full system prompt'


class TextEditor(Protocol):
    """Edit `text` under `title`; the result, or `None` when the user cancelled."""

    def __call__(self, text: str, /, *, title: str) -> str | None: ...


Viewer = Callable[[Pager], object]


def sent_instructions(messages: Sequence[ModelMessage]) -> str | None:
    """What the model received with the latest request in `messages` it answered; `None` before any.

    That is the conversation's system prompt parts, which an agent's `system_prompt` adds to its first
    request and history repeats, then the instructions of the request the latest response answers. A
    request after that response, such as a final tool return, was never sent.
    """
    parts = [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, SystemPromptPart)
    ]
    answered = [
        request.instructions
        for request, response in zip(messages, messages[1:])
        if isinstance(request, ModelRequest) and isinstance(response, ModelResponse)
    ]
    if not answered:
        return None
    latest = answered[-1]
    return '\n\n'.join([*parts, *([latest] if latest else [])])


def _wrap(text: str, width: int) -> list[str]:
    """`text` wrapped to `width`, keeping its line breaks and blank lines.

    Control characters are made inert: the text can come from a repository's `AGENTS.md` or project file.
    """
    return [
        wrapped
        for line in text.splitlines()
        for wrapped in textwrap.wrap(terminal_text(line, keep='\t'), width) or ['']
    ]


class SystemPromptMenu:
    """The rows, their details, and the edits they make, all through `context`."""

    def __init__(self, context: CommandContext, *, history: Callable[[], Sequence[ModelMessage]]) -> None:
        """`history` is the conversation, whose latest request shows the full system prompt."""
        self._context = context
        self._history = history

    @property
    def instructions(self) -> str:
        """Your saved instructions; empty when you have none."""
        return self._context.settings.instructions

    def full_prompt(self) -> str:
        """What the model last received, or what it will receive when nothing was sent yet."""
        sent = sent_instructions(self._history())
        if sent is None:
            return (
                "Nothing was sent in this conversation yet. Your next prompt sends CLAI's built-in "
                'instructions, AGENTS.md, and plugin instructions, followed by yours.'
            )
        return sent or 'The latest request had no system prompt.'

    def items(self) -> list[MenuItem]:
        """The actions, then Save & close. Reset is greyed out while you have no instructions."""
        lines = len(self.instructions.splitlines())
        summary = f'{lines} line{"s" if lines != 1 else ""}' if lines else 'none'
        return [
            MenuItem(Action.EDIT.value, value=Action.EDIT, description=f'{theme.sgr(theme.MUTED)}{summary}'),
            MenuItem(Action.APPEND.value, value=Action.APPEND),
            MenuItem(Action.RESET.value, value=Action.RESET, disabled=not self.instructions),
            MenuItem(Action.VIEW.value, value=Action.VIEW),
            save_and_close_item(),
        ]

    def details(self, item: MenuItem) -> str:
        """The right-hand panel: your instructions, or the full prompt, wrapped to fit."""
        if is_save_and_close(item):
            return SAVE_AND_CLOSE_DETAILS
        columns, rows = terminal_size()
        width = max(20, columns - _LIST_WIDTH - 4)
        if item.value is Action.VIEW:
            lines = [
                'Full system prompt (read-only)',
                '',
                *_wrap(
                    "As sent with this conversation's latest request, yours included. Enter opens it to scroll.",
                    width,
                ),
                '',
            ]
            # Leave room for the title and footer: the pager shows the rest.
            room = max(1, rows - len(lines) - 8)
            prompt = _wrap(self.full_prompt(), width)
            return '\n'.join([*lines, *prompt[:room], *(['…'] if len(prompt) > room else [])])
        command = external_editor()
        opens = f'Opens in {shlex.join(command)}.' if command else f'Opens in the built-in editor: {BUILT_IN_KEYS}.'
        lines = ['Your instructions (editable)', '', *_wrap(self.instructions or '(none)', width), '']
        lines += _wrap(_ADDED, width)
        if self._context.from_project(_KEY):
            lines += _wrap('The project file sets them again at next start.', width)
        lines += ['', *_wrap(opens, width)]
        return '\n'.join(lines)

    def build(self, initial: int = 0) -> Menu:
        """The action list with its details panel."""
        builder = (
            MenuBuilder('System prompt')
            .style(markdown_style())
            .items(self.items())
            .initial_index(initial)
            .list_width(_LIST_WIDTH)
            .preview(self.details)
        )
        return slash_search(builder, footer='enter select · esc close', key_source=menu_key)

    def edit(self, editor: TextEditor) -> str | None:
        """Edit your instructions; `None` when cancelled or unchanged. Emptying them removes them."""
        edited = editor(self.instructions, title=Action.EDIT.value)
        if edited is None or edited.strip() == self.instructions:
            return None
        return self._save(edited.strip())

    def append(self, editor: TextEditor) -> str | None:
        """Add a paragraph after your instructions; `None` when cancelled or empty."""
        added = editor('', title=Action.APPEND.value)
        if added is None or not added.strip():
            return None
        return self._save('\n\n'.join(part for part in (self.instructions, added.strip()) if part))

    def reset(self) -> str:
        """Remove your instructions, leaving CLAI's defaults."""
        return self._save('')

    def _save(self, text: str) -> str:
        if text:
            self._context.set_setting([_KEY, text])
        else:
            self._context.reset_setting(_KEY)
        saved = 'Saved your instructions.' if text else 'Removed your instructions.'
        return f'{saved} They apply from your next prompt.'


def build_pager(text: str) -> Pager:
    """A scrollable, read-only view of the full system prompt."""
    return (
        PagerBuilder('System prompt (read-only)')
        .style(markdown_style())
        .reflow(lambda width: _wrap(text, width))
        .footer_hint('↑↓ scroll · q close')
        .key_source(menu_key)
        .build()
    )


def run_pager(pager: Pager) -> object:  # pragma: no cover -- needs a real terminal.
    """Show a pager on the real terminal."""
    return pager.run()


def run_system_prompt_menu(
    menu: SystemPromptMenu, *, runners: Runners = TERMINAL, editor: TextEditor = edit_text, view: Viewer = run_pager
) -> list[str]:
    """List, act, and back to the list, until Esc or Save & close. Returns the messages to show afterwards."""
    messages: list[str] = []
    cursor = 0
    while True:
        item = picked(runners.run_list(menu.build(cursor)))
        if item is None or not isinstance(item.value, Action):
            return messages
        action = item.value
        cursor = list(Action).index(action)
        if action is Action.VIEW:
            view(build_pager(menu.full_prompt()))
            continue
        if action is Action.RESET:
            messages.append(menu.reset())
            cursor = 0  # The reset row is greyed out now.
            continue
        message = menu.edit(editor) if action is Action.EDIT else menu.append(editor)
        if message is not None:
            messages.append(message)


async def system_prompt_command(
    context: CommandContext,
    args: list[str],
    *,
    history: Callable[[], Sequence[ModelMessage]],
    runners: Runners = TERMINAL,
    editor: TextEditor = edit_text,
    view: Viewer = run_pager,
) -> str:
    """Open the menu in a worker thread; edits save as they happen."""
    if args:
        raise ValueError('Usage: /system_prompt. To set your instructions directly: /set run.instructions TEXT')
    menu = SystemPromptMenu(context, history=history)
    messages = await run_worker(lambda: run_system_prompt_menu(menu, runners=runners, editor=editor, view=view))
    return '\n'.join(messages) or 'No changes.'
