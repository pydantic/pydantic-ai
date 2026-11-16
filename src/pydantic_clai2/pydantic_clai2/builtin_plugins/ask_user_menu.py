"""Let the model ask you multiple-choice questions inline, without leaving the transcript.

The built-in `ask_user` plugin: inline questions that keep the transcript visible.
"""

from collections.abc import Callable, Sequence
from contextlib import nullcontext
from functools import partial

import anyio
from pydantic import BaseModel, ValidationError
from rich.console import Console, RenderableType
from rich.text import Text

from pydantic_ai import AgentStreamEvent, FunctionToolCallEvent
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai_harness.ask_user import (
    TOOL_NAME,
    AskUser,
    AskUserAnswer,
    AskUserAnsweredEvent,
    AskUserRequest,
    AskUserResponse,
)
from pydantic_ai_harness.permissions import Approver, PermissionDecision, PermissionRequest
from pydantic_ai_harness.sub_agents import DelegationTasks
from pydantic_clai2.plugins import FullScreen, Plugin
from pydantic_clai2.runtime.permissions import approval_handler, approval_question, decision, shown_in_full
from pydantic_clai2.ui.menus.menu_worker import run_worker
from pydantic_clai2.ui.menus.question_menu import QuestionMenu
from pydantic_clai2.ui.prompt.question_input import QuestionKey, question_input
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering.rendering import prompt_line
from pydantic_clai2.ui.rendering.tool_output import terminal_text, tool_header


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
        return await self._ask(request)

    async def ask_approval(self, request: PermissionRequest) -> AskUserResponse:
        """Ask `approval_question` about a held call, first printing what it runs in full.

        The picker cuts its question to the rows the terminal leaves it, which a resize can shrink while
        it is open, so a command is always printed in full, and a path when the question cuts it.
        """
        held = (
            None
            if request.action == 'write' and shown_in_full(request)
            else Text(f'{terminal_text(request.call.tool_name)} waits for approval to run:\n')
        )
        if held is not None:
            held.append(terminal_text(request.subject, keep='\t\n'))
        return await self._ask(approval_question(request), held=held)

    async def _ask(self, request: AskUserRequest, *, held: Text | None = None) -> AskUserResponse:
        async with self._terminal, self._full_screen():
            child_id = DelegationTasks.child_id()
            asker = None if child_id is None else f'Task [{child_id[:8]}]'
            if asker is not None:
                self._console.print(f'{asker} requests your input', markup=False)
            if held is not None:
                self._console.print(held)
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
    if event.failed:
        text.append('● The question failed', style=theme.color(theme.MUTED))
        return text
    if event.response.cancelled:
        text.append('● You declined to answer', style=theme.color(theme.MUTED))
        return text
    for index, answer in enumerate(event.response.answers):
        if index:
            text.append('\n')
        text.append('● ', style=theme.color(theme.MUTED))
        text.append(answer.header, style=theme.color(theme.ACCENT))
        if answer.custom_answer is not None:
            # A typed answer is the user's own words, so it reads like a prompt they typed, not muted output.
            text.append(':\n', style=theme.color(theme.MUTED))
            text.append_text(prompt_line(answer.custom_answer))
        else:
            text.append(f': {", ".join(answer.selected)}', style=theme.color(theme.MUTED))
    return text


def render_question(event: AgentStreamEvent) -> RenderableType | None:
    """The transcript's lines for an `ask_user_question` call and its answer; `None` for other events."""
    if isinstance(event, FunctionToolCallEvent) and event.part.tool_name == TOOL_NAME:
        return render_call(event)
    return render_answer(event) if isinstance(event, AskUserAnsweredEvent) else None


def terminal_approver(answerer: TerminalAnswerer) -> Approver:
    """Ask about each call `Permissions` holds in the terminal's picker."""

    async def approve(request: PermissionRequest) -> PermissionDecision:
        return decision(await answerer.ask_approval(request))

    return approve


class AskUserPlugin(Plugin):
    """`AskUser` and `/permissions` approvals with the terminal answerer, a readable call header, and answer lines."""

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        # The picker needs a terminal; a session without one, such as one the browser runs, gets no question tool.
        if not self.host.console.is_terminal:
            return ()
        answerer = TerminalAnswerer(full_screen=self.host.full_screen, console=self.host.console)
        return (AskUser(answerer=answerer), approval_handler(terminal_approver(answerer)))

    def render(self, event: AgentStreamEvent) -> RenderableType | None:
        return render_question(event)
