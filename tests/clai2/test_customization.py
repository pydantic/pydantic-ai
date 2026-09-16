"""The installed authoring guide is discovered cheaply and read only on demand."""

from importlib.resources import files
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai.models.test import TestModel

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
