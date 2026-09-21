"""The installed authoring guide is discovered cheaply and read only on demand."""

from importlib.resources import files
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.ask_user import AskUser, AskUserRequest, AskUserResponse
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.repo_context import RepoContext

from pydantic_clai2 import Session
from pydantic_clai2._app import create_agent
from pydantic_clai2.customization import customization_guide, read_clai_customization_guide


async def test_default_agent_does_not_read_guide_for_normal_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_read(*args: object, **kwargs: object) -> str:
        raise AssertionError('Guide must not be read until requested')

    monkeypatch.setattr('pydantic_clai2.customization.files', unexpected_read)
    agent = create_agent()
    model = TestModel(call_tools=[], custom_output_text='hello')
    with agent.override(model=model):
        result = await Session(agent, deps=None).prompt('hello')
    assert result.output == 'hello'
    assert model.last_model_request_parameters is not None
    assert 'read_clai_customization_guide' in {tool.name for tool in model.last_model_request_parameters.function_tools}
    parts = model.last_model_request_parameters.instruction_parts
    assert parts is not None
    instructions = '\n'.join(part.content for part in parts)
    assert 'first call read_clai_customization_guide' in instructions
    assert '# Customizing CLAI 2' not in instructions


async def test_default_agent_can_read_guide_through_tool() -> None:
    agent = create_agent()
    model = TestModel(call_tools=['read_clai_customization_guide'], custom_output_text='guide loaded')
    with agent.override(model=model):
        result = await Session(agent, deps=None).prompt('How do I customize CLAI menus and providers?')
    returns = [
        part.content
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'read_clai_customization_guide'
    ]
    assert returns == [read_clai_customization_guide()]


async def test_instruction_order_puts_the_hint_between_guidance_and_repository(tmp_path: Path) -> None:
    """The hint follows what the agent is here to do, and precedes the repository's own instructions."""
    (tmp_path / 'AGENTS.md').write_text('# House rules\n')

    async def decline(request: AskUserRequest, /) -> AskUserResponse:
        return AskUserResponse(cancelled=True)

    agent = create_agent()
    model = TestModel(call_tools=[], custom_output_text='hello')
    capabilities: list[AgentCapability[None]] = [
        Coder(workspace=tmp_path, unrestricted_filesystem=True, repo_context=False),
        AskUser(answerer=decline),
        RepoContext(workspace_dir=tmp_path, expose_inventory_tool=False),
    ]
    with agent.override(model=model):
        await agent.run('hello', capabilities=capabilities)
    params = model.last_model_request_parameters
    assert params is not None
    parts = [part.content for part in params.instruction_parts or []]
    assert parts[0].startswith('You are a software engineering agent')
    assert 'ask_user_question' in parts[1]
    assert parts[2].startswith('When asked to customize CLAI itself')
    assert parts[3].startswith('<context-file path="AGENTS.md">')
    assert '# House rules' in parts[3]
    assert 'read_clai_customization_guide' in {tool.name for tool in params.function_tools}


async def test_hint_still_follows_the_coding_guidance_without_ask_user(tmp_path: Path) -> None:
    """Disabling the questions plugin leaves the hint behind the coding guidance, not ahead of it."""
    (tmp_path / 'AGENTS.md').write_text('# House rules\n')
    agent = create_agent()
    model = TestModel(call_tools=[], custom_output_text='hello')
    capabilities: list[AgentCapability[None]] = [
        Coder(workspace=tmp_path, unrestricted_filesystem=True, repo_context=False),
        RepoContext(workspace_dir=tmp_path, expose_inventory_tool=False),
    ]
    with agent.override(model=model):
        await agent.run('hello', capabilities=capabilities)
    params = model.last_model_request_parameters
    assert params is not None
    parts = [part.content for part in params.instruction_parts or []]
    assert parts[0].startswith('You are a software engineering agent')
    assert parts[1].startswith('When asked to customize CLAI itself')
    assert parts[2].startswith('<context-file path="AGENTS.md">')


async def test_custom_agent_can_opt_in() -> None:
    agent = Agent(
        TestModel(call_tools=['read_clai_customization_guide']),
        deps_type=type(None),
        capabilities=[customization_guide()],
    )
    result = await agent.run('Help me write a plugin')
    assert 'Create and install a plugin' in result.output


def test_guide_is_packaged_and_independent_of_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    guide = read_clai_customization_guide()
    assert guide == files('pydantic_clai2').joinpath('customization.md').read_text(encoding='utf-8')
    for section in (
        'Create and install a plugin',
        'Hooks, tools and settings',
        'CLI UX and rendering',
        'Custom TUI menus',
        'Custom models and providers',
        'Test and verify',
    ):
        assert f'## {section}' in guide


@pytest.mark.parametrize('arguments', ['{}', '{"reasoning_effort": "low"}', '{"extra": {"nested": [1, null, true]}}'])
async def test_guide_ignores_extra_tool_arguments(arguments: str) -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool = info.function_tools[0]
        assert tool.strict is False
        assert tool.parameters_json_schema == {'additionalProperties': True, 'properties': {}, 'type': 'object'}
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool.name, arguments)])
        return ModelResponse(parts=[TextPart('guide loaded')])

    agent = Agent(FunctionModel(respond), deps_type=type(None), capabilities=[customization_guide()])
    result = await agent.run('Read the guide')
    parts = [part for message in result.all_messages() for part in message.parts]
    returns = [part.content for part in parts if isinstance(part, ToolReturnPart)]
    assert returns == [read_clai_customization_guide()]
    assert result.output == 'guide loaded'
