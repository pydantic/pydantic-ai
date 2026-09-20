"""The built-in `ask_user` plugin: `AskUser` answered from a full-screen terminal menu.

`build_question_menu` is pure; `TerminalAnswerer` runs one menu per question through `Runners`
so tests script it. Swap the answerer for something else (a web form, a scripted test) by
constructing `AskUser` yourself; see PLUGINS.md.
"""

import asyncio
from dataclasses import dataclass
from functools import partial

from pydantic_ai_harness.ask_user import (
    AskUser,
    AskUserAnswer,
    AskUserAnsweredEvent,
    AskUserRequest,
    AskUserResponse,
    Question,
)
from rich.console import RenderableType
from rich.text import Text
from termflow.tui import MenuBuilder, MenuItem  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import Menu  # pyright: ignore[reportMissingTypeStubs]

from . import theme
from ._rendering import markdown_style
from .field_menu import TERMINAL, Runners
from .menu_worker import menu_key, run_worker
from .plugins import FullScreen, PluginHost

_SINGLE_HINT = 'Enter select - Esc decline'
_MULTI_HINT = 'Space toggle - Enter confirm - Esc decline'
_NO_DESCRIPTION = '(no description)'


@dataclass(frozen=True, kw_only=True)
class QuestionMenu:
    """One question as the menu shows it: options as rows, the question and the option's meaning alongside."""

    question: Question
    position: int
    total: int

    @property
    def title(self) -> str:
        """The header, plus where this question sits when there are several."""
        if self.total == 1:
            return self.question.header
        return f'{self.question.header} (question {self.position} of {self.total})'

    @property
    def hint(self) -> str:
        """The footer: how to pick, and that Esc declines."""
        return _MULTI_HINT if self.question.multi_select else _SINGLE_HINT

    def items(self) -> list[MenuItem]:
        """One row per option, valued by its label."""
        return [MenuItem(option.label, value=option.label) for option in self.question.options]

    def preview(self, item: MenuItem) -> str:
        """The right-hand panel: the question text, then what the highlighted option means."""
        descriptions = {option.label: option.description or _NO_DESCRIPTION for option in self.question.options}
        return f'{self.question.question}\n\n{descriptions[str(item.value)]}'

    def build(self) -> Menu:
        """Wire rows, preview, and keys into a termflow menu on the alternate screen."""
        return (
            MenuBuilder(self.title)
            .style(markdown_style())
            .items(self.items())
            .multi_select(self.question.multi_select)
            .preview(self.preview)
            .footer_hint(self.hint)
            .key_source(menu_key)
            .build()
        )


def build_question_menu(question: Question, *, position: int, total: int) -> Menu:
    """The menu for `question`, the `position`th of `total`."""
    return QuestionMenu(question=question, position=position, total=total).build()


class TerminalAnswerer:
    """Ask each question in turn on the alternate screen; Esc or Ctrl-C on any of them declines the lot.

    There is one terminal, so requests are answered one at a time: when the model calls the tool
    twice in parallel, the second request's menus open after the first request is fully answered.
    """

    def __init__(self, *, full_screen: FullScreen, runners: Runners = TERMINAL) -> None:
        """`full_screen` settles the shell's output first; `runners` shows the menus."""
        self._full_screen = full_screen
        self._runners = runners
        self._terminal = asyncio.Lock()

    async def __call__(self, request: AskUserRequest, /) -> AskUserResponse:
        """Answer every question or report the user declined; never raise for a cancel."""
        answers: list[AskUserAnswer] = []
        total = len(request.questions)
        async with self._terminal, self._full_screen():
            for position, question in enumerate(request.questions, start=1):
                menu = build_question_menu(question, position=position, total=total)
                result = await run_worker(partial(self._runners.run_choice, menu))
                if result.cancelled:
                    return AskUserResponse(cancelled=True)
                # Never empty: on a multi-select menu, Enter with nothing toggled picks the highlighted row.
                selected = tuple(str(item.value) for item in result.items)
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
    host.add(AskUser(answerer=TerminalAnswerer(full_screen=host.full_screen)))
    host.render(AskUserAnsweredEvent)(render_answer)
