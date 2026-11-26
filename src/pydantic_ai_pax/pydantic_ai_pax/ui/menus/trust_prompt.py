"""The inline question the shell asks the first time it starts in a folder: do you trust its files?"""

from functools import partial

from rich.console import Console

from pydantic_ai_harness.ask_user import Question, QuestionOption
from pydantic_ai_pax.ui.menus.menu_worker import run_worker
from pydantic_ai_pax.ui.menus.question_menu import QuestionMenu
from pydantic_ai_pax.ui.prompt.question_input import question_input

TRUST = 'Yes, I trust this folder'
DISTRUST = 'No, keep it untrusted'


async def ask_trust(details: str, *, console: Console) -> bool | None:
    """Whether you trust the folder `details` describes; `None` when dismissed with Esc or Ctrl-C."""
    menu = QuestionMenu(
        question=Question(
            header='Trust this folder?',
            question='Do you trust the files in this folder?',
            options=(QuestionOption(label=TRUST), QuestionOption(label=DISTRUST)),
        ),
        position=1,
        total=1,
        allow_custom=False,
        details=details,
        cursor=1,
    )
    with question_input() as key_source:
        answer = await run_worker(partial(menu.run, console=console, key_source=key_source), inline=True)
    return None if answer is None else answer == (TRUST,)
