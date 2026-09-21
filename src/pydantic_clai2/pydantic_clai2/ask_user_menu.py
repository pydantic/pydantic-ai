"""The built-in `ask_user` plugin: inline questions that keep the transcript visible."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

from pydantic_ai_harness.ask_user import (
    AskUser,
    AskUserAnswer,
    AskUserAnsweredEvent,
    AskUserRequest,
    AskUserResponse,
    Question,
)
from rich.console import Console, RenderableType
from rich.text import Text
from termflow.tui.layout import truncate  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.terminal import raw_mode  # pyright: ignore[reportMissingTypeStubs]

from . import theme
from .menu_worker import menu_key, run_worker
from .plugins import FullScreen, PluginHost
from .prompt_surface import PromptSurface


@dataclass(kw_only=True)
class QuestionMenu:
    """Inline picker state, independent of terminal input and rendering."""

    question: Question
    position: int
    total: int
    cursor: int = 0
    selected: set[int] = field(default_factory=set[int])

    @property
    def title(self) -> str:
        """Include progress when the request contains several questions."""
        if self.total == 1:
            return self.question.header
        return f'{self.question.header} (question {self.position} of {self.total})'

    @property
    def hint(self) -> str:
        """Show the available actions without a separate Space-key convention."""
        action = 'toggle; Done submits' if self.question.multi_select else 'select'
        return f'Up/Down move - number/Enter {action} - Esc decline'

    def choose(self, key: str) -> tuple[str, ...] | None:
        """Apply a key; return selections only when a nonempty answer is submitted."""
        count = len(self.question.options)
        rows = count + int(self.question.multi_select)
        if key in ('up', 'down', 'tab'):
            self.cursor = (self.cursor + (-1 if key == 'up' else 1)) % rows
        elif key == 'enter' or key in tuple(str(i) for i in range(1, rows + 1)):
            if key != 'enter':
                self.cursor = int(key) - 1
            if not self.question.multi_select:
                return (self.question.options[self.cursor].label,)
            if self.cursor == count:
                if self.selected:
                    return tuple(option.label for i, option in enumerate(self.question.options) if i in self.selected)
            elif self.cursor in self.selected:
                self.selected.remove(self.cursor)
            else:
                self.selected.add(self.cursor)
        return None

    def frame(self, *, width: int, height: int) -> tuple[str, ...]:
        """Bound the picker to half the viewport, scrolling choices around the cursor."""
        budget = max(3, height // 2)
        choices: list[str] = []
        for index, option in enumerate(self.question.options):
            marker = ('[x] ' if index in self.selected else '[ ] ') if self.question.multi_select else ''
            description = f' - {option.description}' if option.description else ''
            choices.append(f'{index + 1}. {marker}{option.label}{description}')
        if self.question.multi_select:
            choices.append(f'{len(choices) + 1}. Done' + ('' if self.selected else ' (select at least one)'))
        lines: list[str] = []
        focus = 0
        console = Console()
        for index, choice in enumerate(choices):
            if index == self.cursor:
                focus = len(lines)
            wrapped = Text(choice).wrap(console, width=max(1, width - 2), overflow='fold')
            for line_index, line in enumerate(wrapped):
                prefix = '> ' if index == self.cursor and line_index == 0 else '  '
                role = theme.ACCENT if index == self.cursor else theme.INFO
                lines.append(theme.sgr(role) + truncate(prefix + line.plain, width) + '\x1b[0m')
        visible = max(1, budget - 2)
        start = min(focus, max(0, len(lines) - visible))
        title = truncate(self.title, width)
        return (
            theme.sgr(theme.ACCENT, bold=True) + title + '\x1b[0m',
            *lines[start : start + visible],
            theme.sgr(theme.MUTED) + truncate(self.hint, width) + '\x1b[0m',
        )

    def run(self, *, console: Console, key_source: Callable[[], str] = menu_key) -> tuple[str, ...] | None:
        """Borrow the released editor surface, never entering the alternate screen."""
        surface = console.file
        if not isinstance(surface, PromptSurface):
            surface = PromptSurface(output=surface, size=lambda: console.size)
        try:
            console.print(Text(self.question.question, style=theme.color(theme.ACCENT)))
            with raw_mode():
                while True:
                    surface.paint(self.frame(width=console.width, height=console.height))
                    key = key_source()
                    if key in ('escape', 'ctrl-c'):
                        return None
                    result = self.choose(key)
                    if result is not None:
                        return result
        finally:
            surface.release()


class TerminalAnswerer:
    """Serialize inline question requests while the shell's input reader is suspended."""

    def __init__(
        self,
        *,
        full_screen: FullScreen,
        console: Console | None = None,
        runner: Callable[[QuestionMenu], tuple[str, ...] | None] | None = None,
    ) -> None:
        """Use the shell handoff for exclusive input ownership, not an alternate screen."""
        self._full_screen = full_screen
        self._console = console if console is not None else Console()
        self._runner = runner
        self._terminal = asyncio.Lock()

    async def __call__(self, request: AskUserRequest, /) -> AskUserResponse:
        """Answer every question or decline the entire request."""
        answers: list[AskUserAnswer] = []
        async with self._terminal, self._full_screen():
            for position, question in enumerate(request.questions, start=1):
                menu = QuestionMenu(question=question, position=position, total=len(request.questions))
                operation = partial(self._runner, menu) if self._runner else partial(menu.run, console=self._console)
                selected = await run_worker(operation)
                if selected is None:
                    return AskUserResponse(cancelled=True)
                answers.append(AskUserAnswer(header=question.header, selected=selected))
        return AskUserResponse(answers=tuple(answers))


def render_answer(event: AskUserAnsweredEvent) -> RenderableType:
    """Leave a record of what was picked in the transcript, since the menu itself is gone."""
    text = Text()
    if event.response.cancelled:
        text.append('● You declined to answer', style=theme.color(theme.MUTED))
        return text
    for index, answer in enumerate(event.response.answers):
        if index:
            text.append('\n')
        text.append('● ', style=theme.color(theme.MUTED))
        text.append(answer.header, style=theme.color(theme.ACCENT))
        text.append(f': {", ".join(answer.selected)}', style=theme.color(theme.MUTED))
    return text


def activate(host: PluginHost[None]) -> None:
    """Register `AskUser` with the terminal answerer and a transcript line per answer."""
    host.add(AskUser(answerer=TerminalAnswerer(full_screen=host.full_screen, console=host.console)))
    host.render(AskUserAnsweredEvent)(render_answer)
