"""The Coder skill list: turn each Agent Skill on or off, asked through `Dialogs`."""

from __future__ import annotations

from collections.abc import Sequence
from sqlite3 import Error as SQLiteError
from typing import TYPE_CHECKING

from anyio import to_thread

from pydantic_ai_harness.skills import SkillDefinition
from pydantic_clai2.builtin_plugins.coder_folders import prose
from pydantic_clai2.plugins import DepsT
from pydantic_clai2.plugins.menus import Dialogs, Option
from pydantic_clai2.ui.menus.field_menu import SAVE_AND_CLOSE
from pydantic_clai2.ui.rendering.tool_output import terminal_text

if TYPE_CHECKING:
    from pydantic_clai2.builtin_plugins.coder import CoderSource

_CLOSE = ':close'
"""The list's last option. Skill names never start with `:`."""

_HELP = (
    'Each pick turns a skill on or off and saves it. A skill that is off is hidden from the model '
    'and has no /command, in every project.'
)


def _skill_options(skills: Sequence[SkillDefinition], disabled: Sequence[str]) -> list[Option]:
    """One toggle per skill, `[x]` marking the ones on, then the turned-off names no folder here has."""
    found = {skill.name for skill in skills}
    options = [
        # Skill files come from the repository, so control characters must not reach the terminal.
        Option(
            value=skill.name,
            label=f'{"[ ]" if skill.name in disabled else "[x]"} {terminal_text(skill.name)}',
            description=f'{prose(terminal_text(skill.description))}\n\n{terminal_text(str(skill.path))}',
        )
        for skill in skills
    ]
    options += [
        Option(
            value=name,
            label=f'[ ] {terminal_text(name)} (not found here)',
            description=prose('Turned off, but no skill folder here has it. Turning it on forgets it.'),
        )
        for name in disabled
        if name not in found
    ]
    options.append(Option(value=_CLOSE, label=SAVE_AND_CLOSE, description='Changes are already saved.'))
    return options


async def edit_skills(source: CoderSource[DepsT], skills: Sequence[SkillDefinition], dialogs: Dialogs) -> list[str]:
    """The **Skills** row's action: toggle each skill and save it, until the user closes the list."""
    messages: list[str] = []
    notice = '' if skills else 'No skills were found in the skill folders when Coder loaded.'
    while True:
        disabled = source.disabled_skills()
        choice = await dialogs.choose(
            title='Skills',
            options=_skill_options(skills, disabled),
            description=prose(f'{notice}\n\n{_HELP}' if notice else _HELP),
        )
        if choice is None or choice == _CLOSE:
            return messages
        turned_on = choice in disabled
        names = [name for name in disabled if name != choice] if turned_on else [*disabled, choice]
        try:
            await to_thread.run_sync(source.save_disabled_skills, names)
        except (OSError, ValueError, SQLiteError) as exc:
            notice = f'Could not save: {exc}'
            continue
        notice = f'Turned {terminal_text(choice)} {"on" if turned_on else "off"}.'
        messages.append(notice)
