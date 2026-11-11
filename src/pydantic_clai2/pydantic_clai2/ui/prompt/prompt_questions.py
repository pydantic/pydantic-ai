"""Non-modal questions and explicitly focused approvals on the composer's sole key reader."""

import asyncio
from collections.abc import Awaitable, Callable
from contextvars import ContextVar

import anyio
from rich.console import Console
from termflow.tui.layout import truncate

from pydantic_ai_harness.ask_user import AskUserAnswer, AskUserRequest, AskUserResponse
from pydantic_ai_harness.subagents import DelegationTasks
from pydantic_clai2.config.input_bindings import InputBindings, QuestionSubmit
from pydantic_clai2.ui.menus.question_menu import QuestionMenu
from pydantic_clai2.ui.prompt.image_input import MARKER
from pydantic_clai2.ui.prompt.prompt_buffer import PromptBuffer
from pydantic_clai2.ui.prompt.question_input import Paste
from pydantic_clai2.ui.rendering.tool_output import terminal_text

ComposerAnswerer = Callable[[AskUserRequest, Console, bool], Awaitable[AskUserResponse | None]]
composer_answerer: ContextVar[ComposerAnswerer | None] = ContextVar('clai2_composer_answerer', default=None)


class PromptQuestions:
    """Keep pending questions separate from the draft and require explicit focus for choices.

    The request coroutine owns and clears its state, including on cancellation. No
    terminal is acquired and no worker or second reader is started.
    """

    def __init__(
        self,
        *,
        paint: Callable[[], None],
        preference: Callable[[], QuestionSubmit],
        remember: Callable[[QuestionSubmit], object],
    ) -> None:
        self.paint = paint
        self.preference = preference
        self.remember = remember
        self.menu: QuestionMenu | None = None
        self.focused = False
        self.routing = False
        self.remember_choice = False
        self.notice = ''
        self.approval = False
        self._reviewed = False
        self._armed = False
        self._ready = anyio.Event()
        self._answer: tuple[str, ...] | str | None = None
        self._lock = anyio.Lock()

    async def ask(self, request: AskUserRequest, *, approval: bool = False) -> AskUserResponse:
        """Serialize batches without taking composer focus; release all state on cancellation."""
        async with self._lock:
            self.approval = approval
            answers: list[AskUserAnswer] = []
            child = DelegationTasks.child_id()
            asker = f'Task [{child[:8]}]' if child is not None else None
            try:
                for position, question in enumerate(request.questions, start=1):
                    self.menu = QuestionMenu(
                        question=question, position=position, total=len(request.questions), asker=asker
                    )
                    self.focused = self.routing = False
                    self._reviewed = position == 1
                    self._ready = anyio.Event()
                    self._answer = None
                    self.paint()
                    await self._ready.wait()
                    answer = self._answer
                    if answer is None:
                        return AskUserResponse(cancelled=True)
                    answers.append(
                        AskUserAnswer(header=question.header, custom_answer=answer)
                        if isinstance(answer, str)
                        else AskUserAnswer(header=question.header, selected=answer)
                    )
                return AskUserResponse(answers=tuple(answers))
            finally:
                self.menu = None
                self.focused = self.routing = False
                self.paint()

    def submit(self, answer: tuple[str, ...] | str | None) -> None:
        """Complete only the currently displayed question, once."""
        if self.menu is None or self._ready.is_set():
            return
        self._answer = answer
        self._ready.set()
        self.focused = self.routing = False
        self._armed = False

    def route(self, draft: PromptBuffer) -> bool:
        """Intercept a generic send while a question waits; false means queue a follow-up normally."""
        if self.approval or self.menu is None or self._ready.is_set() or not draft.text.strip():
            return False
        preference = self.preference()
        if preference == 'follow-up':
            return False
        if preference == 'answer' and self._reviewed:
            self.answer_draft(draft)
        else:
            self.routing = True
            self.focused = False
            self.remember_choice = False
            self.arm_later()
        return True

    def arm_later(self) -> None:
        """Typeahead already decoded before an action opens a control cannot activate it."""
        self._armed = False
        asyncio.get_running_loop().call_soon(self.arm)

    def arm(self) -> None:
        self._armed = True

    def answer_draft(self, draft: PromptBuffer) -> None:
        """Move text only after validation and successful explicit submission; never drop images."""
        if self.approval:
            return
        if MARKER.search(draft.text):
            self.notice = 'Question answers are text only. Queue the draft to keep its images.'
            self.routing = True
            self.arm_later()
            return
        answer = draft.text.strip()
        if not answer or self.menu is None or self._ready.is_set():
            return
        self.submit(answer)
        draft.reset()

    def feed(
        self, key: str, data: str, *, draft: PromptBuffer, bindings: InputBindings, queue: Callable[[], None]
    ) -> bool:
        """Consume explicitly focused controls; ordinary typing stays in the composer."""
        if self.menu is None:
            return False
        if self._ready.is_set():
            return bindings.action(key) in ('send', 'steer')
        if key == 'f2':
            self.focused = not self.focused
            self.routing = False
            self._reviewed = True
            self.arm_later()
            return True
        if not (self.focused or self.routing):
            return False
        if not self._armed:
            if key == 'paste':
                draft.insert(data, paste=True)
            elif len(key) == 1 and key.isprintable():
                draft.insert(key)
            return True
        if self.routing:
            return self.route_key(key, draft=draft, queue=queue)
        return self.choose_key(key, data, bindings=bindings)

    def choose_key(self, key: str, data: str, *, bindings: InputBindings) -> bool:
        """Handle only keys explicitly directed to the focused question."""
        assert self.menu is not None
        if key == 'escape':
            if self.menu.editing_custom:
                self.menu.choose(key)
            else:
                self.focused = False
            return True
        if key == 'ctrl-c':
            self.submit(None)
            return True
        action = bindings.action(key)
        if action == 'newline':
            if self.menu.editing_custom:
                self.menu.custom.insert('\n')
            return True
        if key == 'enter' and action != 'send':
            return True
        result = self.menu.choose(Paste(text=data) if key == 'paste' else 'enter' if action == 'send' else key)
        if result is not None:
            self.submit(result)
        return True

    def route_key(self, key: str, *, draft: PromptBuffer, queue: Callable[[], None]) -> bool:
        """Explicit answer/follow-up choice. Enter has no default action here."""
        if key in ('escape', 'ctrl-c'):
            self.routing = False
        elif key == 'r':
            self.remember_choice = not self.remember_choice
        elif key in ('a', 'q'):
            preference: QuestionSubmit = 'answer' if key == 'a' else 'follow-up'
            if self.remember_choice:
                self.remember(preference)
            if key == 'a':
                self.answer_draft(draft)
            else:
                self.routing = False
                queue()
        return True

    def hint(self, *, send_label: str) -> str:
        """Describe what the effective send key does in the current question context."""
        if self.focused:
            return f'{send_label}: choose | F2 composer | Ctrl+J newline'
        if self.routing:
            return 'a Answer | q Follow-up | r Remember | Esc cancel'
        if self.approval:
            return f'{send_label}: queue | F2 review approval'
        preference = self.preference()
        action = (
            'answer or follow-up'
            if preference == 'ask' or not self._reviewed
            else 'answer question'
            if preference == 'answer'
            else 'queue follow-up'
        )
        return f'{send_label}: {action} | F2 choices | Ctrl+J newline'

    def rows(self, *, width: int, height: int, send_label: str = 'Enter') -> tuple[str, ...]:
        """Pending target, focused choices, or an explicit draft routing preview."""
        if self.menu is None:
            return ()
        if self.focused:
            frame = self.menu.frame(width=width, height=height)
            return (
                *frame[:-1],
                frame[-1].replace('Enter', send_label).replace('Esc decline', 'Esc composer'),
                truncate('F2 composer | Ctrl+J newline | Ctrl-C decline', width),
            )
        if self.routing:
            return (
                truncate(f'Answer target: {self.menu.title}', width),
                truncate(terminal_text(self.menu.question.question), width),
                truncate('a Answer this question | q Queue follow-up | Esc cancel', width),
                truncate(f'r Remember my choice: {"yes" if self.remember_choice else "no"} | /input changes it', width),
                truncate(self.notice, width),
            )
        label = 'Approval pending' if self.approval else 'Question pending'
        return (truncate(f'{label}: {self.menu.title} | F2 review/answer; keep typing below', width),)
