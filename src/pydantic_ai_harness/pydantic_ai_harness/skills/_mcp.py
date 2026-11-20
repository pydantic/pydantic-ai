"""Offer the Agent Skills an MCP server serves as deferred capabilities."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability, CombinedCapability
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai_harness.skills._capability import _Skill  # pyright: ignore[reportPrivateUsage]
from pydantic_ai_harness.skills._loader import parse_skill

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPToolset


@dataclass
class MCPSkills(AbstractCapability[AgentDepsT]):
    """Offer the skills an MCP server serves, the way `Skills` offers a library's.

    The server's skills are read at the start of each run: through the
    [Skills extension](https://github.com/modelcontextprotocol/ext-skills) when the toolset's
    client opted into it with FastMCP's `SkillsClientExtension`, and otherwise from the server's `skill://`
    resources. FastMCP checks every file of a skill against the server's manifest before it is offered. The
    model sees each skill's name and description, and loads it with `load_capability`.

    Requires FastMCP 4.2 or later.
    """

    toolset: MCPToolset[Any]
    """The MCP server to read skills from. The agent can use the same toolset for the server's tools."""

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        """Read the server's skills and offer each one the model may load as a deferred capability."""
        async with self.toolset:
            client = self.toolset.client
            try:
                uris = [entry.uri for entry in await client.list_skills()]
            except RuntimeError:
                # Raised when the client or the server did not negotiate the Skills extension.
                uris = [str(resource.uri) for resource in await client.list_skill_resources()]
            skills = [(uri, await client.read_skill(uri)) for uri in uris]
        definitions = [
            replace(parse_skill(skill.files['SKILL.md'].decode(), uri), in_run_workspace=False) for uri, skill in skills
        ]
        return CombinedCapability(
            [_Skill[AgentDepsT](definition) for definition in definitions if definition.model_invocable]
        )
