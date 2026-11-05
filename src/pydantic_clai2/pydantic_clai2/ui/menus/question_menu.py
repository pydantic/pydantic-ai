"""Inline question picker state and terminal rendering, shared by plugins and the shell."""

from collections.abc import Callable
from dataclasses import dataclass, field

from rich.console import Console
from rich.text import Text
from termflow.tui.layout import truncate
from termflow.tui.terminal import raw_mode

from pydantic_ai_harness.ask_user import Question
from pydantic_clai2.ui.prompt.prompt_buffer import PromptBuffer
from pydantic_clai2.ui.prompt.prompt_surface import PromptSurface
from pydantic_clai2.ui.prompt.question_input import Paste, QuestionKey, TranscriptKey
from pydantic_clai2.ui.rendering import theme


@dataclass(kw_only=True)
class QuestionMenu:
    """Inline picker state, independent of terminal input and rendering."""

    question: Question
    position: int
    total: int
    asker: str | None = None
    """Who asks, such as `Task [1a2b3c4d]`, when it is not the run the user prompted."""
    allow_custom: bool = True
    """Whether a last `Other (type answer)` row takes a custom answer; without it, only the options answer."""
    details: str | None = None
    """Dimmed details below a title that asks the question itself, in place of the model's question text."""
    cursor: int = 0
    selected: set[int] = field(default_factory=set[int])
    custom: PromptBuffer = field(default_factory=PromptBuffer)
    editing_custom: bool = False

    @property
    def title(self) -> str:
        """Include who asks, and progress when the request contains several questions."""
        title = self.question.header if self.asker is None else f'{self.asker}: {self.question.header}'
        if self.total == 1:
            return title
        return f'{title} (question {self.position} of {self.total})'

    @property
    def hint(self) -> str:
        """Show the available actions without a separate Space-key convention."""
        if self.editing_custom:
            return 'Enter submits - Esc back - Ctrl-C decline'
        move = 'Up/Down or Tab/Shift+Tab move'
        if self.cursor == self.other:
            return f'{move} - type your answer - Esc decline'
        action = 'toggle; Done submits' if self.question.multi_select else 'select'
        return f'{move} - number/Enter {action} - Esc decline'

    @property
    def other(self) -> int:
        """The row index of `Other (type answer)`, after the options and any `Done` row."""
        return len(self.question.options) + int(self.question.multi_select)

    def edit_custom(self, key: str | Paste) -> str | None:
        """Edit the typed answer; return it when Enter submits nonblank text."""
        if isinstance(key, Paste):
            self.custom.insert(key.text.replace('\t', '    '))
        elif key == 'escape':
            self.editing_custom = False
        elif key == 'enter':
            return self.custom.text.strip() or None
        elif key != 'ctrl-r':
            self.custom.edit(key)
        return None

    def choose(self, key: str | Paste) -> tuple[str, ...] | str | None:
        """Apply a key; return selections only when a nonempty answer is submitted."""
        typed = isinstance(key, Paste) or (len(key) == 1 and key.isprintable())
        if typed and self.cursor == self.other:
            # Text typed on the highlighted `Other` row starts the answer, without an Enter first.
            self.editing_custom = True
        if self.editing_custom:
            return self.edit_custom(key)
        if isinstance(key, Paste):
            return None
        rows = self.other + int(self.allow_custom)
        if key in ('up', 'down', 'tab', 'backtab'):
            self.cursor = (self.cursor + (-1 if key in ('up', 'backtab') else 1)) % rows
        elif key == 'enter' or key in tuple(str(i) for i in range(1, rows + 1)):
            if key != 'enter':
                self.cursor = int(key) - 1
            return self._pick()
        return None

    def _pick(self) -> tuple[str, ...] | None:
        """Act on the highlighted row: start typing on `Other`, answer with an option, or toggle and submit."""
        if self.cursor == self.other:
            self.editing_custom = True
            return None
        if not self.question.multi_select:
            return (self.question.options[self.cursor].label,)
        if self.cursor == len(self.question.options):
            if self.selected:
                return tuple(option.label for i, option in enumerate(self.question.options) if i in self.selected)
        elif self.cursor in self.selected:
            self.selected.remove(self.cursor)
        else:
            self.selected.add(self.cursor)
        return None

    def frame(self, *, width: int, height: int) -> tuple[str, ...]:
        """Bound the picker to half the viewport, scrolling choices around the cursor.

        The question is pinned between the title and its choices: output that streams while it is
        open, from a delegated task say, lands in the transcript above and cannot push it away.
        """
        budget = max(3, height // 2)
        question = self._question_rows(width=width, limit=max(1, (budget - 2) // 2))
        if self.editing_custom:
            return (
                theme.sgr(theme.ACCENT) + truncate(f'{self.title}: Other (type answer)', width) + '\x1b[0m',
                *question,
                *self.custom.rows(width=width, limit=max(1, budget - 2 - len(question))),
                theme.sgr(theme.MUTED) + truncate(self.hint, width) + '\x1b[0m',
            )
        lines: list[str] = []
        focus = 0
        for index, choice in enumerate(self._choices()):
            if index == self.cursor:
                focus = len(lines)
            for line_index, line in enumerate(_wrap(choice, width=width - 2)):
                prefix = '> ' if index == self.cursor and line_index == 0 else '  '
                row = truncate(prefix + line, width)
                # Unhighlighted choices keep the terminal's foreground, readable on light and dark backgrounds alike.
                lines.append(theme.sgr(theme.ACCENT) + row + '\x1b[0m' if index == self.cursor else row)
        visible = max(1, budget - 2 - len(question))
        start = min(focus, max(0, len(lines) - visible))
        title = truncate(self.title, width)
        return (
            theme.sgr(theme.ACCENT, bold=True) + title + '\x1b[0m',
            *question,
            *lines[start : start + visible],
            theme.sgr(theme.MUTED) + truncate(self.hint, width) + '\x1b[0m',
        )

    def _choices(self) -> list[str]:
        """One numbered label per row: the options, `Done` when several can be picked, then `Other`."""
        choices: list[str] = []
        for index, option in enumerate(self.question.options):
            marker = ('[x] ' if index in self.selected else '[ ] ') if self.question.multi_select else ''
            description = f' - {option.description}' if option.description else ''
            choices.append(f'{index + 1}. {marker}{option.label}{description}')
        if self.question.multi_select:
            choices.append(f'{len(choices) + 1}. Done' + ('' if self.selected else ' (select at least one)'))
        if self.allow_custom:
            choices.append(f'{len(choices) + 1}. Other (type answer)')
        return choices

    def _question_rows(self, *, width: int, limit: int) -> list[str]:
        """The question's wrapped text, cut to `limit` rows with an ellipsis so the choices keep room."""
        rows = _wrap(self.details if self.details is not None else self.question.question, width=width)
        if len(rows) > limit:
            rows = [*rows[: limit - 1], rows[limit - 1] + '…']
        if self.details is not None:
            return [theme.sgr(theme.MUTED) + truncate(row, width) + '\x1b[0m' for row in rows]
        return [truncate(row, width) for row in rows]

    def run(self, *, console: Console, key_source: Callable[[], QuestionKey]) -> tuple[str, ...] | str | None:
        """Borrow the released editor's live panel, or open one for this question alone."""
        surface = console.file
        owned = not isinstance(surface, PromptSurface)
        if not isinstance(surface, PromptSurface):
            surface = PromptSurface(output=surface, size=lambda: console.size)
            console = Console(file=surface, width=console.width, height=console.height)
        try:
            with raw_mode():
                while True:
                    surface.paint(self.frame(width=console.width, height=console.height), select_pinned=True)
                    key = key_source()
                    if isinstance(key, TranscriptKey):
                        # Scrolling the transcript or copying painted text must not answer or edit the question.
                        surface.transcript_key(key.key, key.data)
                        continue
                    if key == 'ctrl-c' or (key == 'escape' and not self.editing_custom):
                        return None
                    result = self.choose(key)
                    if result is not None:
                        return result
        finally:
            # The panel showed the question while it was open; the transcript keeps it above the answer.
            console.print(
                Text(self.details, style=theme.color(theme.MUTED))
                if self.details is not None
                else Text(self.question.question, style=theme.color(theme.ACCENT))
            )
            if owned:
                surface.restore()
            else:
                surface.release()


def _wrap(text: str, *, width: int) -> list[str]:
    """Fold `text` to plain rows of at most `width` cells, keeping its own line breaks."""
    return [line.plain for line in Text(text).wrap(Console(), width=max(1, width), overflow='fold')]
