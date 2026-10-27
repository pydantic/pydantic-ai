"""CLAI lists its docs as one deferred capability, then reads them only on demand."""

from collections.abc import AsyncIterator
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_clai2 import Session
from pydantic_clai2.app import create_agent
from pydantic_clai2.builtin_plugins.repo_context import RepoContextPlugin
from pydantic_clai2.clai_docs import read_clai_docs
from pydantic_clai2.customization import CAPABILITY_ID, customization_guide
from pydantic_clai2.plugins import PluginHost, load_plugin

DOC_TOOLS = {'search_clai_docs', 'read_clai_docs'}
CATALOG_LINE = f'- {CAPABILITY_ID}: Use when asked to customize CLAI itself'


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


def tool_returns(messages: list[ModelMessage]) -> list[object]:
    return [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'read_clai_docs'
    ]


async def test_default_agent_lists_docs_without_reading_them(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_read(*args: object, **kwargs: object) -> str:
        raise AssertionError('Docs must not be read until requested')  # pragma: no cover

    monkeypatch.setattr('pydantic_clai2.clai_docs._read', unexpected_read)
    agent = create_agent()
    model = TestModel(call_tools=[], custom_output_text='hello')
    with agent.override(model=model):
        result = await Session(agent, deps=None).prompt('hello')
    assert result.output == 'hello'
    params = model.last_model_request_parameters
    assert params is not None
    assert [tool.name for tool in params.declared_function_tools] == ['load_capability']
    assert {tool.name for tool in params.function_tools if tool.defer_loading} == DOC_TOOLS
    instructions = '\n'.join(part.content for part in params.instruction_parts or ())
    assert CATALOG_LINE in instructions
    assert '# Customizing CLAI 2' not in instructions
    assert 'search_clai_docs' not in instructions


async def test_default_agent_loads_docs_then_reads_guide(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Load the one catalog entry, then read the guide from the installed package."""
    monkeypatch.chdir(tmp_path)
    seen: list[list[str]] = []

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        tools = [tool.name for tool in info.function_tools]
        seen.append(tools)
        if len(seen) == 1:
            assert tools == ['load_capability']
            yield {0: DeltaToolCall(name='load_capability', json_args=f'{{"id": "{CAPABILITY_ID}"}}')}
        elif len(seen) == 2:
            assert DOC_TOOLS <= set(tools)
            yield {0: DeltaToolCall(name='read_clai_docs', json_args='{"path": "customization"}')}
        else:
            yield 'guide loaded'

    agent = create_agent()
    with agent.override(model=FunctionModel(stream_function=stream)):
        result = await Session(agent, deps=None).prompt('How do I customize CLAI menus and providers?')
    assert result.output == 'guide loaded'
    guide = files('pydantic_clai2').joinpath('customization.md').read_text(encoding='utf-8')
    assert tool_returns(result.all_messages()) == [guide]
    assert len(seen) == 3
    for section in (
        'Create and install a plugin',
        'Hooks, tools and settings',
        'CLI UX and rendering',
        'Custom TUI menus',
        'Custom models and providers',
        'Test and verify',
    ):
        assert f'## {section}' in guide


async def test_custom_agent_can_opt_in_with_the_former_name() -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tools = {tool.name for tool in info.function_tools}
        if len(messages) == 1:
            assert tools == {'load_capability'}
            return ModelResponse(parts=[ToolCallPart('load_capability', {'id': CAPABILITY_ID})])
        if len(messages) == 3:
            assert DOC_TOOLS <= tools
            return ModelResponse(parts=[ToolCallPart('read_clai_docs', {'path': 'customization'})])
        return ModelResponse(parts=[TextPart('guide loaded')])

    agent = Agent(FunctionModel(respond), deps_type=type(None), capabilities=[customization_guide()])
    result = await agent.run('Help me write a plugin')
    guide = tool_returns(result.all_messages())
    assert len(guide) == 1 and isinstance(guide[0], str)
    assert '## Create and install a plugin' in guide[0]


@pytest.mark.parametrize(
    'arguments',
    [
        '{"path": "customization"}',
        '{"path": "customization", "reasoning_effort": "low"}',
        '{"path": "customization", "extra": {"nested": [1, null, true]}}',
    ],
)
async def test_docs_tools_ignore_extra_tool_arguments(arguments: str) -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('load_capability', {'id': CAPABILITY_ID})])
        tools = {tool.name: tool for tool in info.function_tools}
        assert tools['read_clai_docs'].strict is False
        assert tools['read_clai_docs'].parameters_json_schema['additionalProperties'] is True
        if len(messages) == 3:
            return ModelResponse(parts=[ToolCallPart('read_clai_docs', arguments)])
        return ModelResponse(parts=[TextPart('guide loaded')])

    agent = Agent(FunctionModel(respond), deps_type=type(None), capabilities=[customization_guide()])
    result = await agent.run('Read the guide')
    assert tool_returns(result.all_messages()) == [read_clai_docs('customization')]
    assert result.output == 'guide loaded'
