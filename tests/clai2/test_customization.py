"""The installed authoring guide is a deferred capability: one catalog line until the model loads it."""

from collections.abc import AsyncIterator
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import (
    CapabilityInstructionSource,
    InstructionDeltaPart,
    InstructionId,
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_clai2 import Session
from pydantic_clai2.app import create_agent
from pydantic_clai2.builtin_plugins.repo_context import RepoContextPlugin
from pydantic_clai2.customization import CAPABILITY_ID, customization_guide
from pydantic_clai2.plugins import PluginHost, load_plugin


@pytest.mark.parametrize('supported', [True, False])
async def test_workspace_defaults_follow_platform_support(supported: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('pydantic_clai2.runtime.session.sys.platform', 'linux' if supported else 'win32')
    monkeypatch.setenv('OPENAI_API_KEY', 'held-back')
    monkeypatch.setenv('CLAI_USER_VARIABLE', 'forwarded')
    agent = Agent(TestModel(custom_output_text='hello'), deps_type=type(None))
    session = Session(agent, deps=None)
    with patch.object(agent, 'run', wraps=agent.run) as run:
        await session.prompt('hello')

    call = run.call_args
    assert call is not None
    capabilities = call.kwargs['capabilities']
    # Last, so a sandbox plugin listed earlier supplies the workspace instead.
    workspace = capabilities[-1] if capabilities else None
    assert isinstance(workspace, LocalWorkspace) is supported
    if isinstance(workspace, LocalWorkspace):
        assert workspace.working_dir == session.workspace
        assert workspace.env is not None
        assert workspace.env['CLAI_USER_VARIABLE'] == 'forwarded'
        assert 'OPENAI_API_KEY' not in workspace.env

    loaded = load_plugin(RepoContextPlugin, PluginHost[None](name='repo_context', console=Console(), settings={}))
    assert any(isinstance(capability, RepoContext) for capability in loaded.capabilities) is supported


GUIDE = files('pydantic_clai2').joinpath('customization.md').read_text(encoding='utf-8')
GUIDE_ID = InstructionId(source=CapabilityInstructionSource(id=CAPABILITY_ID))
CATALOG_LINE = f'- {CAPABILITY_ID}: Use when asked to customize CLAI itself'


def load_guide(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Load the guide on the first request, then answer."""
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('load_capability', {'id': CAPABILITY_ID})])
    return ModelResponse(parts=[TextPart('guide loaded')])


async def stream_load_guide(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    """`load_guide`, streamed as a session requests it, checking that loading the guide adds no tools."""
    assert [tool.name for tool in info.function_tools] == ['load_capability']
    if len(messages) == 1:
        yield {0: DeltaToolCall(name='load_capability', json_args=f'{{"id": "{CAPABILITY_ID}"}}')}
    else:
        yield 'guide loaded'


def loaded_instructions(messages: list[ModelMessage]) -> list[str | None]:
    """The guide's instructions, stated once the model has loaded it."""
    return [
        part.content
        for message in messages
        for part in message.parts
        if isinstance(part, InstructionDeltaPart) and part.id == GUIDE_ID
    ]


async def test_default_agent_lists_the_guide_without_reading_it(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_read(*args: object, **kwargs: object) -> str:
        raise AssertionError('Guide must not be read until requested')  # pragma: no cover

    monkeypatch.setattr('pydantic_clai2.customization.files', unexpected_read)
    agent = create_agent()
    model = TestModel(call_tools=[], custom_output_text='hello')
    with agent.override(model=model):
        result = await Session(agent, deps=None).prompt('hello')
    assert result.output == 'hello'
    params = model.last_model_request_parameters
    assert params is not None
    assert [tool.name for tool in params.function_tools] == ['load_capability']
    instructions = '\n'.join(part.content for part in params.instruction_parts or ())
    assert CATALOG_LINE in instructions
    assert '# Customizing CLAI 2' not in instructions


async def test_default_agent_loads_the_guide_on_demand(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The packaged guide comes back whatever the working directory, and its tools stay unchanged."""
    monkeypatch.chdir(tmp_path)
    agent = create_agent()
    with agent.override(model=FunctionModel(stream_function=stream_load_guide)):
        result = await Session(agent, deps=None).prompt('How do I customize CLAI menus and providers?')
    assert result.output == 'guide loaded'
    assert loaded_instructions(result.all_messages()) == [GUIDE.strip()]
    for section in (
        'Create and install a plugin',
        'Hooks, tools and settings',
        'CLI UX and rendering',
        'Custom TUI menus',
        'Custom models and providers',
        'Test and verify',
    ):
        assert f'## {section}' in GUIDE


async def test_custom_agent_can_opt_in() -> None:
    agent = Agent(FunctionModel(load_guide), deps_type=type(None), capabilities=[customization_guide()])
    result = await agent.run('Help me write a plugin')
    assert loaded_instructions(result.all_messages()) == [GUIDE.strip()]
    assert result.output == 'guide loaded'
