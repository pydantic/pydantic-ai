"""`/system_prompt`: see the instructions the model receives, and edit your own.

Your instructions are the `run.instructions` setting. CLAI sends them after its built-in instructions,
the repository's AGENTS.md, and plugin instructions, which this menu shows but cannot change: yours add
to them and replace none. Clearing yours restores CLAI's defaults. The menu is a `SettingsMenu`, so the
terminal and the web UI show the same rows, each asking through its own `Dialogs`.
"""

from collections.abc import Callable, Sequence

from anyio import to_thread

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, SystemPromptPart
from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.plugins.menus import Dialogs, FieldRow, SettingsMenu
from pydantic_clai2.ui.menus.dialogs import run_settings_menu
from pydantic_clai2.ui.menus.field_menu import Runners
from pydantic_clai2.ui.rendering.tool_output import terminal_text

_KEY = 'run.instructions'
INSTRUCTIONS = 'instructions'
APPEND = 'append'
PROMPT = 'prompt'
NONE = 'none'
"""What the instructions row shows while you have none."""
_SUMMARY_CHARS = 40
_ADDED = (
    "Sent after CLAI's built-in instructions, AGENTS.md, and plugin instructions, "
    'which stay as they are. Changes apply from your next prompt.'
)


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


def summary(text: str) -> str:
    """`text`'s first line, cut to fit a menu row, and how many lines it has; `none` when empty.

    Control characters are made inert: the text can come from a repository's project file.
    """
    if not text:
        return NONE
    lines = text.splitlines()
    first = terminal_text(lines[0])
    if len(first) > _SUMMARY_CHARS:
        first = first[: _SUMMARY_CHARS - 1] + '…'
    return f'{first} ({len(lines)} line{"s" if len(lines) != 1 else ""})'


class SystemPromptSource:
    """The rows `/system_prompt` lists, and the edits its actions make, all through `context`."""

    title = 'System prompt'

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

    def rows(self) -> list[FieldRow]:
        """Your instructions, appending to them, and the full prompt; each opens a dialog."""
        project = self._context.from_project(_KEY)
        return [
            FieldRow(
                key=INSTRUCTIONS,
                label='Your instructions',
                description=_ADDED,
                default=NONE,
                note='the project file sets them again at next start' if project else '',
                button='Edit…',
            ),
            FieldRow(
                key=APPEND,
                label='Append to yours',
                description='Add a paragraph after your instructions, with a blank line between.',
                default='',
                button='Append…',
            ),
            FieldRow(
                key=PROMPT,
                label='Full system prompt',
                description="What the model received with this conversation's latest request, yours included. "
                'Read-only.',
                default='',
                button='View…',
            ),
        ]

    def current(self, row: FieldRow) -> str:
        """A summary of your instructions; the other rows have no value."""
        return summary(self.instructions) if row.key == INSTRUCTIONS else ''

    def problem(self, row: FieldRow, text: str) -> str | None:
        """Only your instructions take a value, as one line; the others are opened."""
        return None if row.key == INSTRUCTIONS else f'{row.label} takes no value; open it instead.'

    def apply(self, row: FieldRow, raw: str) -> str:
        """Replace your instructions with `raw`."""
        if (problem := self.problem(row, raw)) is not None:
            raise ValueError(problem)
        return self._save(raw.strip())

    def reset(self, row: FieldRow) -> str:
        """Remove your instructions, leaving CLAI's defaults."""
        if row.key != INSTRUCTIONS:
            return f'{row.label} has nothing to reset.'
        if not self.instructions:
            return 'You have no instructions to remove.'
        return self._save('')

    async def edit(self, dialogs: Dialogs) -> list[str]:
        """Edit your instructions in a multi-line dialog; emptying them removes them."""
        edited = await dialogs.text(
            title='Edit your instructions',
            label='Your instructions',
            initial=self.instructions,
            description=_ADDED,
            multiline=True,
        )
        if edited is None or edited == self.instructions:
            return []
        # Saving waits for any other save of a setting, so it runs off the loop, as the source's methods do.
        return [await to_thread.run_sync(self._save, edited)]

    async def append(self, dialogs: Dialogs) -> list[str]:
        """Add a paragraph after your instructions."""
        added = await dialogs.text(
            title='Append to your instructions',
            label='New paragraph',
            description='Added after your instructions, with a blank line between.',
            multiline=True,
        )
        if not added:
            return []
        return [await to_thread.run_sync(self._save, '\n\n'.join(part for part in (self.instructions, added) if part))]

    async def view(self, dialogs: Dialogs) -> list[str]:
        """Show the full system prompt, read-only."""
        await dialogs.view(title='System prompt (read-only)', text=self.full_prompt())
        return []

    def _save(self, text: str) -> str:
        if text:
            self._context.set_setting([_KEY, text])
        else:
            self._context.reset_setting(_KEY)
        saved = 'Saved your instructions.' if text else 'Removed your instructions.'
        return f'{saved} They apply from your next prompt.'


def system_prompt_menu(context: CommandContext, *, history: Callable[[], Sequence[ModelMessage]]) -> SettingsMenu:
    """`/system_prompt`'s rows as a declared menu, which the terminal runs and the web UI shows."""
    source = SystemPromptSource(context, history=history)
    return SettingsMenu(
        source=source,
        actions={INSTRUCTIONS: source.edit, APPEND: source.append, PROMPT: source.view},
        searchable=False,
    )


async def system_prompt_command(
    context: CommandContext,
    args: list[str],
    *,
    history: Callable[[], Sequence[ModelMessage]],
    runners: Runners | None = None,
) -> str:
    """Show the menu; edits save as they happen."""
    if args:
        raise ValueError('Usage: /system_prompt. To set your instructions directly: /set run.instructions TEXT')
    messages = await run_settings_menu(system_prompt_menu(context, history=history), runners)
    return '\n'.join(messages) or 'No changes.'
