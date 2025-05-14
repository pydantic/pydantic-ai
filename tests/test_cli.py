import sys
import tempfile
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from dirty_equals import IsInstance, IsStr
from inline_snapshot import snapshot
from pytest import CaptureFixture
from pytest_mock import MockerFixture
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.test import TestModel

from .conftest import TestEnv, try_import

with try_import() as imports_successful:
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.shortcuts import PromptSession

    from pydantic_ai._cli import cli, cli_agent, handle_slash_command

pytestmark = pytest.mark.skipif(not imports_successful(), reason='install cli extras to run cli tests')


def test_cli_version(capfd: CaptureFixture[str]):
    assert cli(['--version']) == 0
    assert capfd.readouterr().out.startswith('pai - PydanticAI CLI')


def test_invalid_model(capfd: CaptureFixture[str]):
    assert cli(['--model', 'potato']) == 1
    assert capfd.readouterr().out.splitlines() == snapshot(
        [IsStr(), 'Error initializing potato:', 'Unknown model: potato']
    )


def test_agent_flag(capfd: CaptureFixture[str], mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')

    # Create a dynamic module using types.ModuleType
    import types

    test_module = types.ModuleType('test_module')

    # Create and add agent to the module
    test_agent = Agent()
    test_agent.model = TestModel(custom_output_text='Hello from custom agent')
    setattr(test_module, 'custom_agent', test_agent)

    # Register the module in sys.modules
    sys.modules['test_module'] = test_module

    try:
        # Mock ask_agent to avoid actual execution but capture the agent
        mock_ask = mocker.patch('pydantic_ai._cli.ask_agent')

        # Test CLI with custom agent
        assert cli(['--agent', 'test_module:custom_agent', 'hello']) == 0

        # Verify the output contains the custom agent message
        assert 'Using custom agent: test_module:custom_agent' in capfd.readouterr().out

        # Verify ask_agent was called with our custom agent
        mock_ask.assert_called_once()
        assert mock_ask.call_args[0][0] is test_agent

    finally:
        # Clean up by removing the module from sys.modules
        if 'test_module' in sys.modules:
            del sys.modules['test_module']


def test_agent_flag_non_agent(capfd: CaptureFixture[str], mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')

    # Create a dynamic module using types.ModuleType
    import types

    test_module = types.ModuleType('test_module')

    # Create and add agent to the module
    test_agent = 'Not an Agent object'
    setattr(test_module, 'custom_agent', test_agent)

    # Register the module in sys.modules
    sys.modules['test_module'] = test_module

    try:
        assert cli(['--agent', 'test_module:custom_agent', 'hello']) == 1
        assert 'is not an Agent' in capfd.readouterr().out

    finally:
        # Clean up by removing the module from sys.modules
        if 'test_module' in sys.modules:
            del sys.modules['test_module']


def test_agent_flag_bad_module_variable_path(capfd: CaptureFixture[str], mocker: MockerFixture, env: TestEnv):
    assert cli(['--agent', 'bad_path', 'hello']) == 1
    assert 'Agent must be specified in "module:variable" format' in capfd.readouterr().out


def test_list_models(capfd: CaptureFixture[str]):
    assert cli(['--list-models']) == 0
    output = capfd.readouterr().out.splitlines()
    assert output[:2] == snapshot([IsStr(regex='pai - PydanticAI CLI .* using openai:gpt-4o'), 'Available models:'])

    providers = (
        'openai',
        'anthropic',
        'bedrock',
        'google-vertex',
        'google-gla',
        'groq',
        'mistral',
        'cohere',
        'deepseek',
    )
    models = {line.strip().split(' ')[0] for line in output[2:]}
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
    with create_pipe_input() as inp:
        inp.send_text('\n')
        inp.send_text('hello\n')
        inp.send_text('/markdown\n')
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
    mock_run_chat.assert_awaited_once_with(
        IsInstance(PromptSession), True, IsInstance(Agent), IsInstance(Console), 'monokai', 'pai'
    )


def test_code_theme_light(mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')
    cli(['--code-theme=light'])
    mock_run_chat.assert_awaited_once_with(
        IsInstance(PromptSession), True, IsInstance(Agent), IsInstance(Console), 'default', 'pai'
    )


def test_code_theme_dark(mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')
    cli(['--code-theme=dark'])
    mock_run_chat.assert_awaited_once_with(
        IsInstance(PromptSession), True, IsInstance(Agent), IsInstance(Console), 'monokai', 'pai'
    )


def test_agent_to_cli_sync(mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')
    cli_agent.to_cli_sync()
    mock_run_chat.assert_awaited_once_with(
        session=IsInstance(PromptSession),
        stream=True,
        agent=IsInstance(Agent),
        console=IsInstance(Console),
        code_theme='monokai',
        prog_name='pydantic-ai',
        deps=None,
    )


@pytest.mark.anyio
async def test_agent_to_cli_async(mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')
    await cli_agent.to_cli()
    mock_run_chat.assert_awaited_once_with(
        session=IsInstance(PromptSession),
        stream=True,
        agent=IsInstance(Agent),
        console=IsInstance(Console),
        code_theme='monokai',
        prog_name='pydantic-ai',
        deps=None,
    )


def test_prompt_history_env_var(mocker: MockerFixture, env: TestEnv, monkeypatch: pytest.MonkeyPatch):
    """Ensure environment variable overrides default prompt history path."""
    env.set('OPENAI_API_KEY', 'test')

    with tempfile.TemporaryDirectory() as env_dir:
        env_history_path = Path(env_dir) / 'custom-history.txt'

        # Simulate setting the environment variable
        monkeypatch.setenv('PYDANTIC_AI_HISTORY_PATH', str(env_history_path))

        # We need to reload the module to pick up the environment variable
        import importlib

        import pydantic_ai._cli

        importlib.reload(pydantic_ai._cli)

        from pydantic_ai._cli import PROMPT_HISTORY_PATH as env_reloaded_path

        assert str(env_reloaded_path) == str(env_history_path)

        assert cli([]) == 0

        assert env_history_path.exists()
        assert env_history_path.parent.exists()


def test_prompt_history_short_flag_o(mocker: MockerFixture, env: TestEnv):
    """Test that the -o short flag correctly sets the prompt history path."""
    env.set('OPENAI_API_KEY', 'test')

    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat', return_value=0)
    # Mock FileHistory to check its instantiation
    mock_file_history_cls = mocker.patch('pydantic_ai._cli.FileHistory')

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_history_file = Path(temp_dir) / 'short-o-history.txt'
        temp_history_path_str = str(temp_history_file)

        assert cli(['-o', temp_history_path_str]) == 0

        mock_file_history_cls.assert_called_once_with(temp_history_path_str)
        mock_run_chat.assert_called_once()

        # Check that the session passed to run_chat has the correct history object
        # (the instance returned by our mocked FileHistory)
        session_arg = mock_run_chat.call_args[0][0]
        assert isinstance(session_arg, PromptSession)
        assert session_arg.history is mock_file_history_cls.return_value

        assert temp_history_file.parent.exists()
        assert temp_history_file.exists()


def test_prompt_history_cli_flag(mocker: MockerFixture, env: TestEnv):
    """Test custom prompt history path via CLI long flag."""
    env.set('OPENAI_API_KEY', 'test')

    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat', return_value=0)
    mock_file_history_cls = mocker.patch('pydantic_ai._cli.FileHistory')

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_history_file = Path(temp_dir) / 'cli-flag-history.txt'
        temp_history_path_str = str(temp_history_file)

        assert cli(['--prompt-history', temp_history_path_str]) == 0

        mock_file_history_cls.assert_called_once_with(temp_history_path_str)
        mock_run_chat.assert_called_once()

        session_arg = mock_run_chat.call_args[0][0]
        assert isinstance(session_arg, PromptSession)
        assert session_arg.history is mock_file_history_cls.return_value

        assert temp_history_file.parent.exists()
        assert temp_history_file.exists()


def test_prompt_history_precedence(mocker: MockerFixture, env: TestEnv, monkeypatch: pytest.MonkeyPatch):
    env.set('OPENAI_API_KEY', 'test')

    with (
        tempfile.TemporaryDirectory() as env_dir,
        tempfile.TemporaryDirectory() as cli_dir,
    ):
        env_history_path = Path(env_dir) / 'env-history.txt'
        cli_history_path = Path(cli_dir) / 'cli-flag-history.txt'

        monkeypatch.setenv('PYDANTIC_AI_HISTORY_PATH', str(env_history_path))

        # We need to reload the module to pick up the environment variable
        import importlib

        import pydantic_ai._cli

        importlib.reload(pydantic_ai._cli)

        from pydantic_ai._cli import PROMPT_HISTORY_PATH as env_reloaded_path

        assert str(env_reloaded_path) == str(env_history_path)

        # Call the CLI with the custom history path
        assert cli(['--prompt-history', str(cli_history_path)]) == 0

        # Test precedence: CLI flag for history_path overrides environment variable.
        # CLI history path should exist, while env history path should not.
        assert cli_history_path.parent.exists()
        assert cli_history_path.exists()

        assert not env_history_path.exists()
