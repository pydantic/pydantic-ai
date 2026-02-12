import sys
import types
from collections.abc import Callable
from io import StringIO
from typing import Any

import pytest
from dirty_equals import IsInstance, IsStr
from inline_snapshot import snapshot
from pytest import CaptureFixture
from pytest_mock import MockerFixture
from rich.console import Console

from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from .conftest import TestEnv, try_import

with try_import() as imports_successful:
    from openai import OpenAIError
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.shortcuts import PromptSession

    from pydantic_ai._cli import cli, cli_agent, handle_slash_command
    from pydantic_ai._cli.web import run_web_command
    from pydantic_ai.models.openai import OpenAIChatModel

pytestmark = pytest.mark.skipif(not imports_successful(), reason='install cli extras to run cli tests')


def test_cli_version(capfd: CaptureFixture[str]):
    assert cli(['--version']) == 0
    assert capfd.readouterr().out.startswith('clai - Pydantic AI CLI')


def test_invalid_model(capfd: CaptureFixture[str]):
    assert cli(['--model', 'potato']) == 1
    assert capfd.readouterr().out.splitlines() == snapshot(['Error initializing potato:', 'Unknown model: potato'])


@pytest.fixture
def create_test_module():
    def _create_test_module(**namespace: Any) -> None:
        assert 'test_module' not in sys.modules

        test_module = types.ModuleType('test_module')
        for key, value in namespace.items():
            setattr(test_module, key, value)

        sys.modules['test_module'] = test_module

    try:
        yield _create_test_module
    finally:
        if 'test_module' in sys.modules:  # pragma: no branch
            del sys.modules['test_module']


def test_agent_flag(
    capfd: CaptureFixture[str],
    mocker: MockerFixture,
    env: TestEnv,
    create_test_module: Callable[..., None],
):
    env.remove('OPENAI_API_KEY')
    env.set('COLUMNS', '150')

    test_agent = Agent(TestModel(custom_output_text='Hello from custom agent'))
    create_test_module(custom_agent=test_agent)

    # Mock ask_agent to avoid actual execution but capture the agent
    mock_ask = mocker.patch('pydantic_ai._cli.ask_agent')

    # Test CLI with custom agent
    assert cli(['--agent', 'test_module:custom_agent', 'hello']) == 0

    # Verify the output contains the custom agent message
    assert 'using custom agent test_module:custom_agent' in capfd.readouterr().out.replace('\n', '')

    # Verify ask_agent was called with our custom agent
    mock_ask.assert_called_once()
    assert mock_ask.call_args[0][0] is test_agent


def test_agent_flag_no_model(env: TestEnv, create_test_module: Callable[..., None]):
    env.remove('OPENAI_API_KEY')
    test_agent = Agent()
    create_test_module(custom_agent=test_agent)

    msg = 'The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable'
    with pytest.raises(OpenAIError, match=msg):
        cli(['--agent', 'test_module:custom_agent', 'hello'])


def test_agent_flag_set_model(
    capfd: CaptureFixture[str],
    mocker: MockerFixture,
    env: TestEnv,
    create_test_module: Callable[..., None],
):
    env.set('OPENAI_API_KEY', 'xxx')
    env.set('COLUMNS', '150')

    custom_agent = Agent(TestModel(custom_output_text='Hello from custom agent'))
    create_test_module(custom_agent=custom_agent)

    mocker.patch('pydantic_ai._cli.ask_agent')

    assert cli(['--agent', 'test_module:custom_agent', '--model', 'openai:gpt-4o', 'hello']) == 0

    assert 'using custom agent test_module:custom_agent with openai:gpt-4o' in capfd.readouterr().out.replace('\n', '')

    assert isinstance(custom_agent.model, OpenAIChatModel)


def test_agent_flag_non_agent(
    capfd: CaptureFixture[str], mocker: MockerFixture, create_test_module: Callable[..., None]
):
    test_agent = 'Not an Agent object'
    create_test_module(custom_agent=test_agent)

    assert cli(['--agent', 'test_module:custom_agent', 'hello']) == 1
    assert 'Could not load agent from test_module:custom_agent' in capfd.readouterr().out


def test_agent_flag_bad_module_variable_path(capfd: CaptureFixture[str], mocker: MockerFixture, env: TestEnv):
    assert cli(['--agent', 'bad_path', 'hello']) == 1
    assert 'Could not load agent from bad_path' in capfd.readouterr().out


def test_no_command_defaults_to_chat(mocker: MockerFixture):
    """Test that running clai with no command defaults to chat mode."""
    # Mock _run_chat_command to avoid actual execution
    mock_run_chat = mocker.patch('pydantic_ai._cli._run_chat_command', return_value=0)
    result = cli([])
    assert result == 0
    mock_run_chat.assert_called_once()


def test_list_models(capfd: CaptureFixture[str]):
    assert cli(['--list-models']) == 0
    output = capfd.readouterr().out.splitlines()
    assert output[:3] == snapshot([IsStr(regex='clai - Pydantic AI CLI .*'), '', 'Available models:'])

    providers = (
        'openai',
        'anthropic',
        'bedrock',
        'cerebras',
        'google-vertex',
        'google-gla',
        'groq',
        'mistral',
        'cohere',
        'deepseek',
        'gateway/',
        'heroku',
        'moonshotai',
        'grok',
        'xai',
        'huggingface',
    )
    models = {line.strip().split(' ')[0] for line in output[3:]}
    for provider in providers:
        models = models - {model for model in models if model.startswith(provider)}
    assert models == set(), models


def test_cli_prompt(capfd: CaptureFixture[str], env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    with cli_agent.override(model=TestModel(custom_output_text='# result\n\n```py\nx = 1\n```')):
        assert cli(['hello']) == 0
        assert capfd.readouterr().out.splitlines() == snapshot([IsStr(), '# result', '', 'py', 'x = 1', '/py'])
        assert cli(['--no-stream', 'hello']) == 0
        assert capfd.readouterr().out.splitlines() == snapshot([IsStr(), '# result', '', 'py', 'x = 1', '/py'])


def test_chat(capfd: CaptureFixture[str], mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')

    # mocking is needed because of ci does not have xclip or xselect installed
    def mock_copy(text: str) -> None:
        pass

    mocker.patch('pyperclip.copy', mock_copy)
    with create_pipe_input() as inp:
        inp.send_text('\n')
        inp.send_text('hello\n')
        inp.send_text('/markdown\n')
        inp.send_text('/cp\n')
        inp.send_text('/exit\n')
        session = PromptSession[Any](input=inp, output=DummyOutput())
        m = mocker.patch('pydantic_ai._cli.PromptSession', return_value=session)
        m.return_value = session
        m = TestModel(custom_output_text='goodbye')
        with cli_agent.override(model=m):
            assert cli([]) == 0
        assert capfd.readouterr().out.splitlines() == snapshot(
            [
                IsStr(),
                IsStr(regex='goodbye *Markdown output of last question:'),
                '',
                'goodbye',
                'Copied last output to clipboard.',
                'Exiting…',
            ]
        )


def test_handle_slash_command_markdown():
    io = StringIO()
    assert handle_slash_command('/markdown', [], False, Console(file=io), 'default') == (None, False)
    assert io.getvalue() == snapshot('No markdown output available.\n')

    messages: list[ModelMessage] = [ModelResponse(parts=[TextPart('[hello](# hello)'), ToolCallPart('foo', '{}')])]
    io = StringIO()
    assert handle_slash_command('/markdown', messages, True, Console(file=io), 'default') == (None, True)
    assert io.getvalue() == snapshot("""\
Markdown output of last question:

[hello](# hello)
""")


def test_handle_slash_command_multiline():
    io = StringIO()
    assert handle_slash_command('/multiline', [], False, Console(file=io), 'default') == (None, True)
    assert io.getvalue()[:70] == IsStr(regex=r'Enabling multiline mode.*')

    io = StringIO()
    assert handle_slash_command('/multiline', [], True, Console(file=io), 'default') == (None, False)
    assert io.getvalue() == snapshot('Disabling multiline mode.\n')


def test_handle_slash_command_copy(mocker: MockerFixture):
    io = StringIO()
    # mocking is needed because of ci does not have xclip or xselect installed
    mock_clipboard: list[str] = []

    def append_to_clipboard(text: str) -> None:
        mock_clipboard.append(text)

    mocker.patch('pyperclip.copy', append_to_clipboard)
    assert handle_slash_command('/cp', [], False, Console(file=io), 'default') == (None, False)
    assert io.getvalue() == snapshot('No output available to copy.\n')
    assert mock_clipboard == snapshot([])

    messages: list[ModelMessage] = [ModelResponse(parts=[TextPart(''), ToolCallPart('foo', '{}')])]
    io = StringIO()
    assert handle_slash_command('/cp', messages, True, Console(file=io), 'default') == (None, True)
    assert io.getvalue() == snapshot('No text content to copy.\n')
    assert mock_clipboard == snapshot([])

    messages: list[ModelMessage] = [ModelResponse(parts=[TextPart('hello'), ToolCallPart('foo', '{}')])]
    io = StringIO()
    assert handle_slash_command('/cp', messages, True, Console(file=io), 'default') == (None, True)
    assert io.getvalue() == snapshot('Copied last output to clipboard.\n')
    assert mock_clipboard == snapshot(['hello'])


def test_handle_slash_command_exit():
    io = StringIO()
    assert handle_slash_command('/exit', [], False, Console(file=io), 'default') == (0, False)
    assert io.getvalue() == snapshot('Exiting…\n')


def test_handle_slash_command_other():
    io = StringIO()
    assert handle_slash_command('/foobar', [], False, Console(file=io), 'default') == (None, False)
    assert io.getvalue() == snapshot('Unknown command `/foobar`\n')


def test_code_theme_unset(mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')
    cli([])
    mock_run_chat.assert_awaited_once_with(True, IsInstance(Agent), IsInstance(Console), 'monokai', 'clai')


def test_code_theme_light(mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')
    cli(['--code-theme=light'])
    mock_run_chat.assert_awaited_once_with(True, IsInstance(Agent), IsInstance(Console), 'default', 'clai')


def test_code_theme_dark(mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')
    cli(['--code-theme=dark'])
    mock_run_chat.assert_awaited_once_with(True, IsInstance(Agent), IsInstance(Console), 'monokai', 'clai')


def test_agent_to_cli_sync(mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')
    cli_agent.to_cli_sync()
    mock_run_chat.assert_awaited_once_with(
        stream=True,
        agent=IsInstance(Agent),
        console=IsInstance(Console),
        code_theme='monokai',
        prog_name='pydantic-ai',
        deps=None,
        message_history=None,
        model_settings=None,
        usage_limits=None,
    )


@pytest.mark.anyio
async def test_agent_to_cli_async(mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')
    await cli_agent.to_cli()
    mock_run_chat.assert_awaited_once_with(
        stream=True,
        agent=IsInstance(Agent),
        console=IsInstance(Console),
        code_theme='monokai',
        prog_name='pydantic-ai',
        deps=None,
        message_history=None,
        model_settings=None,
        usage_limits=None,
    )


@pytest.mark.anyio
async def test_agent_to_cli_with_message_history(mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')

    # Create some test message history - cast to the proper base type
    test_messages: list[ModelMessage] = [ModelResponse(parts=[TextPart('Hello!')])]

    await cli_agent.to_cli(message_history=test_messages)
    mock_run_chat.assert_awaited_once_with(
        stream=True,
        agent=IsInstance(Agent),
        console=IsInstance(Console),
        code_theme='monokai',
        prog_name='pydantic-ai',
        deps=None,
        message_history=test_messages,
        model_settings=None,
        usage_limits=None,
    )


def test_agent_to_cli_sync_with_message_history(mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')

    # Create some test message history - cast to the proper base type
    test_messages: list[ModelMessage] = [ModelResponse(parts=[TextPart('Hello!')])]

    cli_agent.to_cli_sync(message_history=test_messages)
    mock_run_chat.assert_awaited_once_with(
        stream=True,
        agent=IsInstance(Agent),
        console=IsInstance(Console),
        code_theme='monokai',
        prog_name='pydantic-ai',
        deps=None,
        message_history=test_messages,
        model_settings=None,
        usage_limits=None,
    )


@pytest.mark.parametrize(
    ('model_name', 'expected'),
    [
        ('gpt-5', 'GPT 5'),
        ('gpt-4.1', 'GPT 4.1'),
        ('o1', 'O1'),
        ('o3', 'O3'),
        ('claude-sonnet-4-5', 'Claude Sonnet 4.5'),
        ('claude-haiku-4-5', 'Claude Haiku 4.5'),
        ('gemini-2.5-pro', 'Gemini 2.5 Pro'),
        ('gemini-2.5-flash', 'Gemini 2.5 Flash'),
        ('sonnet-4-5', 'Sonnet 4.5'),
        ('custom-model', 'Custom Model'),
    ],
)
def test_model_label(model_name: str, expected: str):
    """Test Model.label formatting for UI."""
    from pydantic_ai.models.test import TestModel

    model = TestModel(model_name=model_name)
    assert model.label == expected


def test_clai_web_generic_agent(mocker: MockerFixture, env: TestEnv):
    """Test web command without agent creates generic agent."""
    env.set('OPENAI_API_KEY', 'test')
    mock_run_web = mocker.patch('pydantic_ai._cli.web.run_web_command', return_value=0)

    assert cli(['web', '-m', 'openai:gpt-5', '-t', 'web_search'], prog_name='clai') == 0

    mock_run_web.assert_called_once_with(
        agent_path=None,
        host='127.0.0.1',
        port=7932,
        models=['openai:gpt-5'],
        tools=['web_search'],
        instructions=None,
        default_model='openai:gpt-5',
        html_source=None,
    )


def test_clai_web_success(mocker: MockerFixture, create_test_module: Callable[..., None], env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')

    mock_run_web = mocker.patch('pydantic_ai._cli.web.run_web_command', return_value=0)

    test_agent = Agent(TestModel(custom_output_text='test'))
    create_test_module(custom_agent=test_agent)

    assert cli(['web', '--agent', 'test_module:custom_agent'], prog_name='clai') == 0

    mock_run_web.assert_called_once_with(
        agent_path='test_module:custom_agent',
        host='127.0.0.1',
        port=7932,
        models=[],
        tools=[],
        instructions=None,
        default_model='openai:gpt-5',
        html_source=None,
    )


def test_clai_web_with_models(mocker: MockerFixture, create_test_module: Callable[..., None], env: TestEnv):
    """Test web command with multiple -m flags."""
    env.set('OPENAI_API_KEY', 'test')

    mock_run_web = mocker.patch('pydantic_ai._cli.web.run_web_command', return_value=0)

    test_agent = Agent(TestModel(custom_output_text='test'))
    create_test_module(custom_agent=test_agent)

    assert (
        cli(
            [
                'web',
                '--agent',
                'test_module:custom_agent',
                '-m',
                'openai:gpt-5',
                '-m',
                'anthropic:claude-sonnet-4-5',
            ],
            prog_name='clai',
        )
        == 0
    )

    mock_run_web.assert_called_once_with(
        agent_path='test_module:custom_agent',
        host='127.0.0.1',
        port=7932,
        models=['openai:gpt-5', 'anthropic:claude-sonnet-4-5'],
        tools=[],
        instructions=None,
        default_model='openai:gpt-5',
        html_source=None,
    )


def test_clai_web_with_tools(mocker: MockerFixture, create_test_module: Callable[..., None], env: TestEnv):
    """Test web command with multiple -t flags."""
    env.set('OPENAI_API_KEY', 'test')

    mock_run_web = mocker.patch('pydantic_ai._cli.web.run_web_command', return_value=0)

    test_agent = Agent(TestModel(custom_output_text='test'))
    create_test_module(custom_agent=test_agent)

    assert (
        cli(
            ['web', '--agent', 'test_module:custom_agent', '-t', 'web_search', '-t', 'code_execution'], prog_name='clai'
        )
        == 0
    )

    mock_run_web.assert_called_once_with(
        agent_path='test_module:custom_agent',
        host='127.0.0.1',
        port=7932,
        models=[],
        tools=['web_search', 'code_execution'],
        instructions=None,
        default_model='openai:gpt-5',
        html_source=None,
    )


def test_clai_web_generic_with_instructions(mocker: MockerFixture, env: TestEnv):
    """Test generic agent with custom instructions."""
    env.set('OPENAI_API_KEY', 'test')

    mock_run_web = mocker.patch('pydantic_ai._cli.web.run_web_command', return_value=0)

    assert cli(['web', '-m', 'openai:gpt-5', '-i', 'You are a helpful coding assistant'], prog_name='clai') == 0

    mock_run_web.assert_called_once_with(
        agent_path=None,
        host='127.0.0.1',
        port=7932,
        models=['openai:gpt-5'],
        tools=[],
        instructions='You are a helpful coding assistant',
        default_model='openai:gpt-5',
        html_source=None,
    )


def test_clai_web_with_custom_port(mocker: MockerFixture, create_test_module: Callable[..., None], env: TestEnv):
    """Test web command with custom host/port."""
    env.set('OPENAI_API_KEY', 'test')

    mock_run_web = mocker.patch('pydantic_ai._cli.web.run_web_command', return_value=0)

    test_agent = Agent(TestModel(custom_output_text='test'))
    create_test_module(custom_agent=test_agent)

    assert (
        cli(['web', '--agent', 'test_module:custom_agent', '--host', '0.0.0.0', '--port', '7932'], prog_name='clai')
        == 0
    )

    mock_run_web.assert_called_once_with(
        agent_path='test_module:custom_agent',
        host='0.0.0.0',
        port=7932,
        models=[],
        tools=[],
        instructions=None,
        default_model='openai:gpt-5',
        html_source=None,
    )


def test_run_web_command_agent_with_model(
    mocker: MockerFixture, create_test_module: Callable[..., None], capfd: CaptureFixture[str]
):
    """Test run_web_command uses agent's model when no -m flag provided."""

    mock_uvicorn_run = mocker.patch('uvicorn.run')
    mocker.patch('pydantic_ai._cli.web.create_web_app')

    test_agent = Agent(TestModel(custom_output_text='test'))
    create_test_module(custom_agent=test_agent)

    result = run_web_command(agent_path='test_module:custom_agent')

    assert result == 0
    mock_uvicorn_run.assert_called_once()


def test_run_web_command_generic_agent_no_model(mocker: MockerFixture, capfd: CaptureFixture[str]):
    """Test run_web_command uses default model when no agent and no model provided."""
    mock_uvicorn_run = mocker.patch('uvicorn.run')
    mock_create_app = mocker.patch('pydantic_ai._cli.web.create_web_app')

    result = run_web_command()

    assert result == 0
    mock_uvicorn_run.assert_called_once()
    # Verify default model was passed
    call_kwargs = mock_create_app.call_args.kwargs
    assert call_kwargs['models'] == ['openai:gpt-5']


def test_run_web_command_generic_agent_with_instructions(mocker: MockerFixture, capfd: CaptureFixture[str]):
    """Test run_web_command passes instructions to create_web_app for generic agent."""

    mock_uvicorn_run = mocker.patch('uvicorn.run')
    mock_create_app = mocker.patch('pydantic_ai._cli.web.create_web_app')

    result = run_web_command(models=['test'], instructions='You are a helpful assistant')

    assert result == 0
    mock_uvicorn_run.assert_called_once()

    # Verify instructions were passed to create_web_app (not to Agent constructor)
    call_kwargs = mock_create_app.call_args.kwargs
    assert call_kwargs['instructions'] == 'You are a helpful assistant'


def test_run_web_command_agent_with_instructions(
    mocker: MockerFixture, create_test_module: Callable[..., None], capfd: CaptureFixture[str]
):
    """Test run_web_command passes instructions to create_web_app when agent is provided."""

    mock_uvicorn_run = mocker.patch('uvicorn.run')
    mock_create_app = mocker.patch('pydantic_ai._cli.web.create_web_app')

    test_agent = Agent(TestModel(custom_output_text='test'))
    create_test_module(custom_agent=test_agent)

    result = run_web_command(agent_path='test_module:custom_agent', instructions='Always respond in Spanish')

    assert result == 0
    mock_uvicorn_run.assert_called_once()

    # Verify instructions were passed to create_web_app
    call_kwargs = mock_create_app.call_args.kwargs
    assert call_kwargs['instructions'] == 'Always respond in Spanish'


def test_run_web_command_agent_load_failure(capfd: CaptureFixture[str]):
    """Test run_web_command returns error when agent path is invalid."""

    result = run_web_command(agent_path='nonexistent_module:agent')

    assert result == 1
    output = capfd.readouterr().out
    assert 'Could not load agent' in output


def test_run_web_command_unknown_tool(mocker: MockerFixture, capfd: CaptureFixture[str]):
    """Test run_web_command warns about unknown tool IDs."""

    mock_uvicorn_run = mocker.patch('uvicorn.run')
    mocker.patch('pydantic_ai._cli.web.create_web_app')

    result = run_web_command(models=['test'], tools=['unknown_tool_xyz'])

    assert result == 0
    mock_uvicorn_run.assert_called_once()
    output = capfd.readouterr().out
    assert 'Unknown tool "unknown_tool_xyz"' in output


def test_run_web_command_memory_tool(mocker: MockerFixture, capfd: CaptureFixture[str]):
    """Test run_web_command warns about memory tool requiring agent configuration."""

    mock_uvicorn_run = mocker.patch('uvicorn.run')
    mocker.patch('pydantic_ai._cli.web.create_web_app')

    result = run_web_command(models=['test'], tools=['memory'])

    assert result == 0
    mock_uvicorn_run.assert_called_once()
    output = capfd.readouterr().out
    assert '"memory" requires configuration and cannot be enabled via CLI' in output


def test_run_web_command_agent_builtin_tools_not_duplicated(
    mocker: MockerFixture, create_test_module: Callable[..., None], capfd: CaptureFixture[str]
):
    """Test run_web_command only passes CLI-provided tools, not agent's builtin tools."""
    from pydantic_ai.builtin_tools import WebSearchTool

    mock_uvicorn_run = mocker.patch('uvicorn.run')
    mock_create_app = mocker.patch('pydantic_ai._cli.web.create_web_app')

    # Create agent with web_search tool already configured
    test_agent = Agent(TestModel(custom_output_text='test'), builtin_tools=[WebSearchTool()])
    create_test_module(custom_agent=test_agent)

    # Add code_execution via CLI
    result = run_web_command(agent_path='test_module:custom_agent', tools=['code_execution'])

    assert result == 0
    mock_uvicorn_run.assert_called_once()

    # Verify only CLI-provided tools are passed (agent's tools are handled by create_web_app)
    call_kwargs = mock_create_app.call_args.kwargs
    builtin_tools = call_kwargs.get('builtin_tools', [])
    tool_kinds = {t.kind for t in builtin_tools}
    # web_search is on the agent, so it's NOT passed here (it's handled internally)
    assert 'web_search' not in tool_kinds
    # code_execution was provided via CLI, so it IS passed
    assert 'code_execution' in tool_kinds


def test_run_web_command_cli_models_passed_to_create_web_app(
    mocker: MockerFixture, create_test_module: Callable[..., None]
):
    """Test that CLI models are passed directly to create_web_app (agent model merging happens there)."""
    mock_uvicorn_run = mocker.patch('uvicorn.run')
    mock_create_app = mocker.patch('pydantic_ai._cli.web.create_web_app')

    test_agent = Agent(TestModel(custom_output_text='test'))
    create_test_module(custom_agent=test_agent)

    result = run_web_command(
        agent_path='test_module:custom_agent', models=['openai:gpt-5', 'anthropic:claude-sonnet-4-5']
    )

    assert result == 0
    mock_uvicorn_run.assert_called_once()

    call_kwargs = mock_create_app.call_args.kwargs
    # CLI models passed as list; agent model merging/deduplication happens in create_web_app
    assert call_kwargs.get('models') == ['openai:gpt-5', 'anthropic:claude-sonnet-4-5']


def test_agent_to_cli_sync_with_args(mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')

    model_settings = ModelSettings(temperature=0.5)
    usage_limits = UsageLimits(request_limit=10)

    cli_agent.to_cli_sync(model_settings=model_settings, usage_limits=usage_limits)

    mock_run_chat.assert_awaited_once_with(
        stream=True,
        agent=IsInstance(Agent),
        console=IsInstance(Console),
        code_theme='monokai',
        prog_name='pydantic-ai',
        deps=None,
        message_history=None,
        model_settings=model_settings,
        usage_limits=usage_limits,
    )


@pytest.mark.anyio
async def test_agent_to_cli_async_with_args(mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')

    model_settings = ModelSettings(temperature=0.5)
    usage_limits = UsageLimits(request_limit=10)

    await cli_agent.to_cli(model_settings=model_settings, usage_limits=usage_limits)

    mock_run_chat.assert_awaited_once_with(
        stream=True,
        agent=IsInstance(Agent),
        console=IsInstance(Console),
        code_theme='monokai',
        prog_name='pydantic-ai',
        deps=None,
        message_history=None,
        model_settings=model_settings,
        usage_limits=usage_limits,
    )


def test_clai_web_with_html_source(mocker: MockerFixture, env: TestEnv):
    """Test web command with --html-source flag."""
    env.set('OPENAI_API_KEY', 'test')
    mock_run_web = mocker.patch('pydantic_ai._cli.web.run_web_command', return_value=0)

    custom_url = 'https://internal.company.com/pydantic-ai-ui/index.html'
    assert cli(['web', '-m', 'openai:gpt-5', '--html-source', custom_url], prog_name='clai') == 0

    mock_run_web.assert_called_once_with(
        agent_path=None,
        host='127.0.0.1',
        port=7932,
        models=['openai:gpt-5'],
        tools=[],
        instructions=None,
        default_model='openai:gpt-5',
        html_source=custom_url,
    )
