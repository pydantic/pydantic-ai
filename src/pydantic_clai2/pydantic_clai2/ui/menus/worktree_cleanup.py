"""The inline exit question for a changed linked checkout."""

from functools import partial
from typing import Literal

from rich.console import Console

from pydantic_ai_harness.ask_user import Question, QuestionOption
from pydantic_clai2.ui.menus.menu_worker import run_worker
from pydantic_clai2.ui.menus.question_menu import QuestionMenu
from pydantic_clai2.ui.prompt.question_input import question_input
from pydantic_clai2.ui.rendering.tool_output import terminal_text


async def ask_removal(details: str) -> Literal['remove', 'keep', 'never']:
    """Default to keeping the checkout; cancelling also keeps it."""
    menu = QuestionMenu(
        question=Question(
            header='Remove this worktree?',
            question='Remove this worktree?',
            options=(
                QuestionOption(label='Yes'),
                QuestionOption(label='No'),
                QuestionOption(label="No, don't ask again for this worktree"),
            ),
        ),
        position=1,
        total=1,
        allow_custom=False,
        details=terminal_text(details, keep='\n\t'),
        cursor=1,
    )
    with question_input() as key_source:
        answer = await run_worker(partial(menu.run, console=Console(), key_source=key_source), inline=True)
    if answer == ('Yes',):
        return 'remove'
    if answer == ("No, don't ask again for this worktree",):
        return 'never'
    return 'keep'
