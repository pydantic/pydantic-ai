"""Tools that offer skills below `'operator'` authority to the model as tool output."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import NotRequired

from typing_extensions import TypedDict

from pydantic_ai import ModelRetry
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.skills._loader import SkillDefinition, catalog_description


class SkillListing(TypedDict):
    """What `list_skills` returns."""

    skills: dict[str, str]
    """Each skill's name and what it's for: its description, then its `when_to_use`, as the catalog shows them."""
    skipped: NotRequired[list[str]]
    """Why each skill that couldn't be read was skipped, when one was."""


@dataclass(frozen=True)
class UntrustedSkills:
    """The skills below `'operator'` authority the model may load, and messages for the ones a read skipped."""

    skills: list[SkillDefinition]
    skipped: list[str]


class UntrustedSkillsToolset(FunctionToolset[AgentDepsT]):
    """`list_skills` and `load_skill`, for skills whose files the operator doesn't control.

    One instance serves every run of the agent, so the tools read the libraries again through
    `ctx.workspace` rather than keeping a run's catalog.
    """

    def __init__(self, read: Callable[[RunContext[AgentDepsT]], Awaitable[UntrustedSkills]], *, id: str | None) -> None:
        """`read` gives the skills below `'operator'` authority the model may load, for the run in `ctx`."""
        super().__init__(id=id)
        self._read = read
        self.add_function(self.list_skills)
        self.add_function(self.load_skill)

    async def list_skills(self, ctx: RunContext[AgentDepsT]) -> SkillListing:
        """List the skills held back from your instructions: `skills` maps each one's name to its description.

        They aren't offered as the operator's instructions. Call `load_skill` to read one's instructions. `skipped`,
        when present, says which skills couldn't be read and why.
        """
        read = await self._read(ctx)
        listing = SkillListing(skills={skill.name: catalog_description(skill) for skill in read.skills})
        if read.skipped:
            listing['skipped'] = read.skipped
        return listing

    async def load_skill(self, ctx: RunContext[AgentDepsT], name: str) -> str:
        """Read the instructions of a skill that `list_skills` lists.

        Args:
            ctx: Framework-provided run context.
            name: The skill's name, as `list_skills` lists it.
        """
        read = await self._read(ctx)
        for skill in read.skills:
            if skill.name == name:
                return skill.render()
        skipped = ''.join(f'\n- {message}' for message in read.skipped)
        if skipped:
            skipped = f'\nSkipped:{skipped}'
        raise ModelRetry(f'No skill is named {name!r}. Call `list_skills` to see the skills.{skipped}')
