from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.skills import MCPSkills

pytest.importorskip('fastmcp.skills.provider', reason='MCPSkills needs FastMCP 4.2 and its server to test against')

from fastmcp import Client, FastMCP
from fastmcp.skills import SkillsClientExtension
from fastmcp.skills.provider import SkillsDirectoryProvider

from pydantic_ai.mcp import MCPToolset


def _server(tmp_path: Path) -> FastMCP:
    for name, frontmatter in [
        ('review', 'name: review\ndescription: Review a change.'),
        ('people-only', 'name: people-only\ndescription: Only for people.\ndisable-model-invocation: true'),
    ]:
        (tmp_path / name).mkdir()
        (tmp_path / name / 'SKILL.md').write_text(f'---\n{frontmatter}\n---\n\nReport findings by severity.\n')
    return FastMCP('skills', providers=[SkillsDirectoryProvider(tmp_path)])


async def _run(toolset: MCPToolset) -> tuple[str | None, str]:
    """The catalog the model saw, and what loading `review` returned."""
    seen: list[str | None] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        if len(messages) == 1:
            seen.append(request.instructions)
            return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'review'}, tool_call_id='load')])
        seen.extend(
            part.content for part in request.parts if isinstance(part, UserPromptPart) and isinstance(part.content, str)
        )
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), capabilities=[MCPSkills(toolset)])
    await agent.run('go')
    return seen[0], '\n'.join(str(content) for content in seen[1:])


class TestMCPSkills:
    @pytest.mark.parametrize('extension', [True, False], ids=['skills-extension', 'resources'])
    async def test_server_skills_load_on_demand(self, tmp_path: Path, extension: bool) -> None:
        server = _server(tmp_path)
        client = Client(server, extensions=[SkillsClientExtension()] if extension else [])

        catalog, loaded = await _run(MCPToolset(client))

        assert catalog is not None
        assert '- review: Review a change.' in catalog
        assert 'people-only' not in catalog
        assert '# Skill: review\n\nReport findings by severity.' in loaded
