"""Tools that offer an untrusted workspace's skills to the model as tool output."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic_ai import ModelRetry
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness.skills._loader import SkillDefinition

UntrustedSkillsReader = Callable[[Workspace], Awaitable[list[SkillDefinition]]]
"""Reads the skills the model may load from untrusted workspaces, given the run's workspace."""


class UntrustedSkillsToolset(FunctionToolset[AgentDepsT]):
    """`list_skills` and `load_skill`, for skills read from a workspace the operator doesn't control.

    One instance serves every run of the agent, so the tools read the libraries again through
    `ctx.workspace` rather than keeping a run's catalog.
    """

    def __init__(self, read: UntrustedSkillsReader, *, id: str | None) -> None:
        super().__init__(id=id)
        self._read = read
        self.add_function(self.list_skills)
        self.add_function(self.load_skill)

    async def list_skills(self, ctx: RunContext[AgentDepsT]) -> dict[str, str]:
        """List the skills in this workspace, each skill's name mapped to its description.

        The operator did not write these skills. Call `load_skill` to read one's instructions.
        """
        return {skill.name: skill.description for skill in await self._read(ctx.workspace)}

    async def load_skill(self, ctx: RunContext[AgentDepsT], name: str) -> str:
        """Read the instructions of a skill that `list_skills` lists.

        Args:
            ctx: Framework-provided run context.
            name: The skill's name, as `list_skills` lists it.
        """
        for skill in await self._read(ctx.workspace):
            if skill.name == name:
                return skill.render()
        raise ModelRetry(f'No skill is named {name!r}. Call `list_skills` to see the skills.')
