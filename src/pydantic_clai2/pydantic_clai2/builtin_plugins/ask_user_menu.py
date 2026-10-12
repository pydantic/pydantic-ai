"""Let the model ask you multiple-choice questions inline, without leaving the transcript.

The built-in `ask_user` plugin: inline questions that keep the transcript visible.
"""

from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial

import anyio
from pydantic import BaseModel, ValidationError
from rich.console import Console, RenderableType
from rich.text import Text
from termflow.tui.layout import truncate
from termflow.tui.terminal import raw_mode

from pydantic_ai import AgentStreamEvent, FunctionToolCallEvent
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai_harness.ask_user import (
    TOOL_NAME,
    AskUser,
    AskUserAnswer,
    AskUserAnsweredEvent,
    AskUserRequest,
    AskUserResponse,
    Question,
)
from pydantic_ai_harness.subagents import DelegationTasks
from pydantic_clai2.plugins import FullScreen, Plugin
from pydantic_clai2.ui.menus.menu_worker import run_worker
from pydantic_clai2.ui.prompt.prompt_buffer import PromptBuffer
from pydantic_clai2.ui.prompt.prompt_surface import PromptSurface
from pydantic_clai2.ui.prompt.question_input import Paste, QuestionKey, TranscriptKey, question_input
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering.tool_output import tool_header


@dataclass(kw_only=True)
class QuestionMenu:
    """Inline picker state, independent of terminal input and rendering."""

    question: Question
    position: int
    total: int
    asker: str | None = None
    """Who asks, such as `Task [1a2b3c4d]`, when it is not the run the user prompted."""
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
        action = 'toggle; Done submits' if self.question.multi_select else 'select'
        return f'Up/Down move - number/Enter {action} - Esc decline'

    def choose(self, key: str | Paste) -> tuple[str, ...] | str | None:
        """Apply a key; return selections only when a nonempty answer is submitted."""
        if isinstance(key, Paste):
            if self.editing_custom:
                self.custom.insert(key.text.replace('\t', '    '))
            return None
        if self.editing_custom:
            if key == 'escape':
                self.editing_custom = False
            elif key == 'enter':
                return self.custom.text.strip() or None
            elif key != 'ctrl-r':
                self.custom.edit(key)
            return None
        count = len(self.question.options)
        rows = count + int(self.question.multi_select) + 1
        if key in ('up', 'down', 'tab'):
            self.cursor = (self.cursor + (-1 if key == 'up' else 1)) % rows
        elif key == 'enter' or key in tuple(str(i) for i in range(1, rows + 1)):
            if key != 'enter':
                self.cursor = int(key) - 1
            if self.cursor == rows - 1:
                self.editing_custom = True
                return None
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
        choices: list[str] = []
        for index, option in enumerate(self.question.options):
            marker = ('[x] ' if index in self.selected else '[ ] ') if self.question.multi_select else ''
            description = f' - {option.description}' if option.description else ''
            choices.append(f'{index + 1}. {marker}{option.label}{description}')
        if self.question.multi_select:
            choices.append(f'{len(choices) + 1}. Done' + ('' if self.selected else ' (select at least one)'))
        choices.append(f'{len(choices) + 1}. Other (type answer)')
        lines: list[str] = []
        focus = 0
        for index, choice in enumerate(choices):
            if index == self.cursor:
                focus = len(lines)
            for line_index, line in enumerate(_wrap(choice, width=width - 2)):
                prefix = '> ' if index == self.cursor and line_index == 0 else '  '
                role = theme.ACCENT if index == self.cursor else theme.INFO
                lines.append(theme.sgr(role) + truncate(prefix + line, width) + '\x1b[0m')
        visible = max(1, budget - 2 - len(question))
        start = min(focus, max(0, len(lines) - visible))
        title = truncate(self.title, width)
        return (
            theme.sgr(theme.ACCENT, bold=True) + title + '\x1b[0m',
            *question,
            *lines[start : start + visible],
            theme.sgr(theme.MUTED) + truncate(self.hint, width) + '\x1b[0m',
        )

    def _question_rows(self, *, width: int, limit: int) -> list[str]:
        """The question's wrapped text, cut to `limit` rows with an ellipsis so the choices keep room."""
        rows = _wrap(self.question.question, width=width)
        if len(rows) > limit:
            rows = [*rows[: limit - 1], rows[limit - 1] + '…']
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
                    surface.paint(self.frame(width=console.width, height=console.height))
                    key = key_source()
                    if isinstance(key, TranscriptKey):
                        # Reading back through or copying the transcript must not answer or edit the question.
                        surface.transcript_key(key.key, key.data)
                        continue
                    if key == 'ctrl-c' or (key == 'escape' and not self.editing_custom):
                        return None
                    result = self.choose(key)
                    if result is not None:
                        return result
        finally:
            # The panel showed the question while it was open; the transcript keeps it above the answer.
            console.print(Text(self.question.question, style=theme.color(theme.ACCENT)))
            if owned:
                surface.restore()
            else:
                surface.release()


class TerminalAnswerer:
    """Serialize inline question requests while the shell's input reader is suspended."""

    def __init__(
        self,
        *,
        full_screen: FullScreen,
        console: Console | None = None,
        runner: Callable[[QuestionMenu], tuple[str, ...] | str | None] | None = None,
    ) -> None:
        """Use the shell handoff for exclusive input ownership, not an alternate screen."""
        self._full_screen = full_screen
        self._console = console if console is not None else Console()
        self._runner = runner
        self._terminal = anyio.Lock(fast_acquire=True)

    async def __call__(self, request: AskUserRequest, /) -> AskUserResponse:
        """Answer every question or decline the entire request."""
        async with self._terminal, self._full_screen():
            child_id = DelegationTasks.child_id()
            asker = None if child_id is None else f'Task [{child_id[:8]}]'
            if asker is not None:
                self._console.print(f'{asker} requests your input', markup=False)
            with question_input() if self._runner is None else nullcontext(None) as key_source:
                return await self.answer_questions(request=request, key_source=key_source, asker=asker)

    async def answer_questions(
        self, *, request: AskUserRequest, key_source: Callable[[], QuestionKey] | None, asker: str | None = None
    ) -> AskUserResponse:
        """Keep one decoder for the batch so pasted text cannot escape to the next question."""
        answers: list[AskUserAnswer] = []
        for position, question in enumerate(request.questions, start=1):
            menu = QuestionMenu(question=question, position=position, total=len(request.questions), asker=asker)
            if self._runner is not None:
                operation = partial(self._runner, menu)
            else:
                assert key_source is not None
                operation = partial(menu.run, console=self._console, key_source=key_source)
            selected = await run_worker(operation, inline=True)
            if selected is None:
                return AskUserResponse(cancelled=True)
            answers.append(
                AskUserAnswer(header=question.header, custom_answer=selected)
                if isinstance(selected, str)
                else AskUserAnswer(header=question.header, selected=selected)
            )
        return AskUserResponse(answers=tuple(answers))


def _wrap(text: str, *, width: int) -> list[str]:
    """Fold `text` to plain rows of at most `width` cells, keeping its own line breaks."""
    return [line.plain for line in Text(text).wrap(Console(), width=max(1, width), overflow='fold')]


class _Header(BaseModel):
    header: str


class _Headers(BaseModel):
    """Only the display fields of the call; the harness validates the rest."""

    questions: list[_Header]


def render_call(event: FunctionToolCallEvent) -> RenderableType:
    """Name the questions, not their raw JSON: the picker shows each one in full right after."""
    try:
        headers = _Headers.model_validate_json(event.part.args_as_json_str()).questions
    except ValidationError:
        headers = []
    return tool_header(name=TOOL_NAME, argument=', '.join(question.header for question in headers))


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
        value = answer.custom_answer if answer.custom_answer is not None else ', '.join(answer.selected)
        text.append(f': {value}', style=theme.color(theme.MUTED))
    return text


class AskUserPlugin(Plugin):
    """`AskUser` with the terminal answerer, a readable call header, and a transcript line per answer."""

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        return (AskUser(answerer=TerminalAnswerer(full_screen=self.host.full_screen, console=self.host.console)),)

    def render(self, event: AgentStreamEvent) -> RenderableType | None:
        if isinstance(event, FunctionToolCallEvent) and event.part.tool_name == TOOL_NAME:
            return render_call(event)
        return render_answer(event) if isinstance(event, AskUserAnsweredEvent) else None
