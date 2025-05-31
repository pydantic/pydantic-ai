import sys
import types
from io import StringIO
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock, patch

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
    from openai import OpenAIError
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.shortcuts import PromptSession

    from pydantic_ai._cli import (
        LeftHeading,
        SimpleCodeBlock,
        cli,
        cli_agent,
        discover_local_models,
        handle_slash_command,
        load_discovery_config,
        save_discovery_config,
    )
    from pydantic_ai.models.openai import OpenAIModel

pytestmark = pytest.mark.skipif(not imports_successful(), reason='install cli extras to run cli tests')


def test_cli_version(capfd: CaptureFixture[str]):
    assert cli(['--version']) == 0
    assert capfd.readouterr().out.startswith('pai - PydanticAI CLI')


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

    assert cli(['--agent', 'test_module:custom_agent', '--model', 'gpt-4o', 'hello']) == 0

    assert 'using custom agent test_module:custom_agent with openai:gpt-4o' in capfd.readouterr().out.replace('\n', '')

    assert isinstance(custom_agent.model, OpenAIModel)


def test_agent_flag_non_agent(
    capfd: CaptureFixture[str],
    mocker: MockerFixture,
    create_test_module: Callable[..., None],
):
    test_agent = 'Not an Agent object'
    create_test_module(custom_agent=test_agent)

    assert cli(['--agent', 'test_module:custom_agent', 'hello']) == 1
    assert 'is not an Agent' in capfd.readouterr().out


def test_agent_flag_bad_module_variable_path(capfd: CaptureFixture[str], mocker: MockerFixture, env: TestEnv):
    assert cli(['--agent', 'bad_path', 'hello']) == 1
    assert 'Agent must be specified in "module:variable" format' in capfd.readouterr().out


def test_list_models(capfd: CaptureFixture[str]):
    assert cli(['--list-models']) == 0
    output = capfd.readouterr().out.splitlines()
    assert output[:3] == snapshot([IsStr(regex='pai - PydanticAI CLI .*'), '', 'Available models:'])

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
    # Filter out lines that are not model names (discovery-related lines, empty lines, etc.)
    model_lines = [
        line
        for line in output[3:]
        if line.strip() and not line.strip().startswith('Last') and not line.strip().startswith('Use')
    ]
    models = {line.strip().split(' ')[0] for line in model_lines}
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
    assert io.getvalue() == snapshot(
        """\
Markdown output of last question:

[hello](# hello)
"""
    )


def test_handle_slash_command_multiline():
    io = StringIO()
    assert handle_slash_command('/multiline', [], False, Console(file=io), 'default') == (None, True)
    assert io.getvalue()[:70] == IsStr(regex=r'Enabling multiline mode.*')

    io = StringIO()
    assert handle_slash_command('/multiline', [], True, Console(file=io), 'default') == (None, False)
    assert io.getvalue() == snapshot('Disabling multiline mode.\n')


def test_handle_slash_command_exit():
    io = StringIO()
    assert handle_slash_command('/exit', [], False, Console(file=io), 'default') == (
        0,
        False,
    )
    assert io.getvalue() == snapshot('Exiting…\n')


def test_handle_slash_command_other():
    io = StringIO()
    assert handle_slash_command('/foobar', [], False, Console(file=io), 'default') == (
        None,
        False,
    )
    assert io.getvalue() == snapshot('Unknown command `/foobar`\n')


def test_code_theme_unset(mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')
    cli([])
    mock_run_chat.assert_awaited_once_with(True, IsInstance(Agent), IsInstance(Console), 'monokai', 'pai')


def test_code_theme_light(mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')
    cli(['--code-theme=light'])
    mock_run_chat.assert_awaited_once_with(True, IsInstance(Agent), IsInstance(Console), 'default', 'pai')


def test_code_theme_dark(mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')
    cli(['--code-theme=dark'])
    mock_run_chat.assert_awaited_once_with(True, IsInstance(Agent), IsInstance(Console), 'monokai', 'pai')


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
    )


# Tests for discovery functionality
def test_save_discovery_config(tmp_path: Path):
    """Test saving discovery configuration."""
    with patch('pydantic_ai._cli.PYDANTIC_AI_HOME', tmp_path):
        save_discovery_config('http://localhost:1234/v1', 'test-model')

        config_file = tmp_path / 'discovery.json'
        assert config_file.exists()

        import json

        with open(config_file) as f:
            config = json.load(f)

        assert config['endpoint'] == 'http://localhost:1234/v1'
        assert config['model_name'] == 'test-model'
        assert 'timestamp' in config


def test_save_discovery_config_error_handling(tmp_path: Path):
    """Test that save_discovery_config handles errors gracefully."""
    # Create a read-only directory to trigger an error
    readonly_dir = tmp_path / 'readonly'
    readonly_dir.mkdir(mode=0o444)

    with patch('pydantic_ai._cli.PYDANTIC_AI_HOME', readonly_dir):
        # Should not raise an exception
        save_discovery_config('http://localhost:1234/v1', 'test-model')


def test_load_discovery_config(tmp_path: Path):
    """Test loading discovery configuration."""
    config_file = tmp_path / 'discovery.json'
    config_data = {
        'endpoint': 'http://localhost:1234/v1',
        'model_name': 'test-model',
        'timestamp': '2024-01-01T12:00:00',
    }

    import json

    with open(config_file, 'w') as f:
        json.dump(config_data, f)

    with patch('pydantic_ai._cli.PYDANTIC_AI_HOME', tmp_path):
        result = load_discovery_config()
        assert result == (
            'http://localhost:1234/v1',
            'test-model',
            '2024-01-01T12:00:00',
        )


def test_load_discovery_config_no_file(tmp_path: Path):
    """Test loading discovery configuration when file doesn't exist."""
    with patch('pydantic_ai._cli.PYDANTIC_AI_HOME', tmp_path):
        result = load_discovery_config()
        assert result is None


def test_load_discovery_config_error_handling(tmp_path: Path):
    """Test that load_discovery_config handles errors gracefully."""
    config_file = tmp_path / 'discovery.json'
    # Create invalid JSON
    with open(config_file, 'w') as f:
        f.write('invalid json')

    with patch('pydantic_ai._cli.PYDANTIC_AI_HOME', tmp_path):
        result = load_discovery_config()
        assert result is None


@pytest.mark.anyio
async def test_discover_local_models_success():
    """Test successful model discovery."""
    mock_response = MagicMock()
    mock_response.json.return_value = {'data': [{'id': 'model-b'}, {'id': 'model-a'}, {'id': 'model-c'}]}

    with patch('httpx.AsyncClient') as mock_client:
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

        models = await discover_local_models('http://localhost:1234/v1')
        assert models == ['model-a', 'model-b', 'model-c']  # Should be sorted


@pytest.mark.anyio
async def test_discover_local_models_no_data():
    """Test model discovery with no data field."""
    mock_response = MagicMock()
    mock_response.json.return_value = {'models': []}

    with patch('httpx.AsyncClient') as mock_client:
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

        models = await discover_local_models('http://localhost:1234/v1')
        assert models == []


@pytest.mark.anyio
async def test_discover_local_models_error():
    """Test model discovery error handling."""
    with patch('httpx.AsyncClient') as mock_client:
        mock_client.return_value.__aenter__.return_value.get.side_effect = Exception('Connection failed')

        with pytest.raises(Exception, match='Failed to discover models from'):
            await discover_local_models('http://localhost:1234/v1')


def test_discover_models_no_endpoint_no_config(capfd: CaptureFixture[str]):
    """Test discover models with no endpoint and no saved config."""
    with patch('pydantic_ai._cli.load_discovery_config', return_value=None):
        assert cli(['-d']) == 1
        output = capfd.readouterr().out
        assert 'No endpoint provided and no previous discovery found' in output


def test_discover_models_use_saved_endpoint(capfd: CaptureFixture[str], mocker: MockerFixture):
    """Test discover models using saved endpoint."""
    mock_config = ('http://localhost:1234/v1', 'test-model', '2024-01-01T12:00:00')
    mock_discover = mocker.patch('pydantic_ai._cli.discover_local_models', return_value=[])

    with patch('pydantic_ai._cli.load_discovery_config', return_value=mock_config):
        assert cli(['-d']) == 1  # Returns 1 because no models found
        output = capfd.readouterr().out
        assert 'Using last discovered endpoint: http://localhost:1234/v1' in output
        mock_discover.assert_called_once()


def test_discover_direct_no_config(capfd: CaptureFixture[str]):
    """Test discover direct with no saved config."""
    with patch('pydantic_ai._cli.load_discovery_config', return_value=None):
        assert cli(['-dd']) == 1
        output = capfd.readouterr().out
        assert 'No previous discovery found' in output


def test_discover_direct_success(capfd: CaptureFixture[str], mocker: MockerFixture, env: TestEnv):
    """Test successful direct discovery connection."""
    env.set('OPENAI_API_KEY', 'test')
    mock_config = ('http://localhost:1234/v1', 'test-model', '2024-01-01T12:00:00')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat', return_value=0)

    with patch('pydantic_ai._cli.load_discovery_config', return_value=mock_config):
        assert cli(['-dd']) == 0
        output = capfd.readouterr().out
        assert 'Connecting directly to: test-model at http://localhost:1234/v1' in output
        mock_run_chat.assert_called_once()


def test_discover_direct_connection_error(capfd: CaptureFixture[str]):
    """Test discover direct with connection error."""
    mock_config = ('http://localhost:1234/v1', 'test-model', '2024-01-01T12:00:00')

    with patch('pydantic_ai._cli.load_discovery_config', return_value=mock_config):
        with patch(
            'pydantic_ai.providers.openai.OpenAIProvider',
            side_effect=Exception('Connection failed'),
        ):
            assert cli(['-dd']) == 1
            output = capfd.readouterr().out
            assert 'Error connecting to http://localhost:1234/v1' in output


def test_list_models_with_discovery_config(capfd: CaptureFixture[str]):
    """Test list models showing last discovery config."""
    mock_config = ('http://localhost:1234/v1', 'test-model', '2024-01-01T12:00:00')

    with patch('pydantic_ai._cli.load_discovery_config', return_value=mock_config):
        assert cli(['-l']) == 0
        output = capfd.readouterr().out
        assert 'Last discovery: test-model at http://localhost:1234/v1' in output
        assert 'Use "clai -d" to reconnect to last endpoint' in output


def test_list_models_with_invalid_timestamp(capfd: CaptureFixture[str]):
    """Test list models with invalid timestamp in config."""
    mock_config = ('http://localhost:1234/v1', 'test-model', 'invalid-timestamp')

    with patch('pydantic_ai._cli.load_discovery_config', return_value=mock_config):
        assert cli(['-l']) == 0
        output = capfd.readouterr().out
        assert 'Last discovery: test-model at http://localhost:1234/v1' in output


# Tests for coverage completion


def test_list_models_with_invalid_timestamp_format(capfd: CaptureFixture[str]):
    """Test list models with invalid timestamp format in discovery config."""

    # Mock load_discovery_config to return a config with invalid timestamp
    # This should trigger the ValueError in datetime.fromisoformat()
    def mock_load_config():
        return ('http://localhost:1234/v1', 'test-model', 'definitely-not-iso-format')

    with patch('pydantic_ai._cli.load_discovery_config', side_effect=mock_load_config):
        assert cli(['-l']) == 0
        output = capfd.readouterr().out
        # Should show the fallback format without timestamp
        assert 'Last discovery: test-model at http://localhost:1234/v1' in output
        assert 'Use "clai -d" to reconnect to last endpoint' in output


def test_list_models_with_none_timestamp(capfd: CaptureFixture[str]):
    """Test list models with None timestamp to trigger TypeError."""

    # Mock load_discovery_config to return a config with None timestamp
    # This should trigger the TypeError in datetime.fromisoformat()
    def mock_load_config():
        return ('http://localhost:1234/v1', 'test-model', None)

    with patch('pydantic_ai._cli.load_discovery_config', side_effect=mock_load_config):
        assert cli(['-l']) == 0
        output = capfd.readouterr().out
        # Should show the fallback format without timestamp
        assert 'Last discovery: test-model at http://localhost:1234/v1' in output
        assert 'Use "clai -d" to reconnect to last endpoint' in output


def test_discover_direct_with_invalid_timestamp(capfd: CaptureFixture[str], mocker: MockerFixture, env: TestEnv):
    """Test discover direct with invalid timestamp in config."""
    env.set('OPENAI_API_KEY', 'test')
    mock_config = ('http://localhost:1234/v1', 'test-model', 'invalid-timestamp')
    mocker.patch('pydantic_ai._cli.run_chat', return_value=0)

    with patch('pydantic_ai._cli.load_discovery_config', return_value=mock_config):
        assert cli(['-dd']) == 0
        output = capfd.readouterr().out
        assert 'Connecting directly to: test-model at http://localhost:1234/v1' in output
    # Should not have timestamp in output due to invalid format


def test_discover_models_with_invalid_last_used_timestamp(mocker: MockerFixture, capfd: CaptureFixture[str]):
    """Test discover models with invalid timestamp in last used config."""
    mock_config = ('http://localhost:1234/v1', 'model-a', 'invalid-timestamp')
    mocker.patch('pydantic_ai._cli.discover_local_models', return_value=['model-a', 'model-b'])
    mocker.patch('builtins.input', return_value='1')
    mocker.patch('pydantic_ai._cli.save_discovery_config')
    mocker.patch('pydantic_ai._cli.run_chat', return_value=0)

    with patch('pydantic_ai._cli.load_discovery_config', return_value=mock_config):
        assert cli(['-d']) == 0

    output = capfd.readouterr().out
    assert 'Last used: model-a' in output  # Should show without timestamp


def test_agent_without_model_flag_branch(
    capfd: CaptureFixture[str],
    mocker: MockerFixture,
    env: TestEnv,
    create_test_module: Callable[..., None],
):
    """Test the specific elif args.agent branch (lines 385-387)."""
    env.set('OPENAI_API_KEY', 'test')

    # Create an agent that already has a model - this is key!
    test_agent = Agent(TestModel(custom_output_text='Hello'))
    create_test_module(custom_agent=test_agent)

    # Mock ask_agent to avoid actual execution
    mock_ask = mocker.patch('pydantic_ai._cli.ask_agent')

    # Call with agent but NO model flag - this should hit the elif branch
    # The condition is: args.agent is True AND model_arg_set is False
    # Since the agent already has a model, we skip the model setting if block
    # and go to the elif args.agent branch
    result = cli(['--agent', 'test_module:custom_agent', 'hello'])
    assert result == 0

    output = capfd.readouterr().out
    # Should show just the agent name without "with model"
    # Handle potential newlines and multiple spaces in the output
    output_clean = ' '.join(output.split())  # This normalizes whitespace
    assert 'using custom agent test_module:custom_agent' in output_clean
    # Should NOT have "with" in the output since no model override
    assert 'test_module:custom_agent with' not in output_clean

    mock_ask.assert_called_once()


def test_simple_code_block():
    """Test SimpleCodeBlock rendering."""
    from rich.console import Console
    from rich.syntax import Syntax
    from rich.text import Text

    console = Console(file=StringIO(), width=80)
    code_block = SimpleCodeBlock('print("hello")', 'python')
    # Set the text attribute that the __rich_console__ method expects
    code_block.text = Text('print("hello")')

    # Test that it renders without background color
    result = list(code_block.__rich_console__(console, console.options))
    assert len(result) == 3
    assert isinstance(result[0], Text)  # Language name
    assert isinstance(result[1], Syntax)  # Code syntax
    assert isinstance(result[2], Text)  # Closing language name


def test_left_heading():
    """Test LeftHeading rendering."""
    from rich.console import Console
    from rich.text import Text

    console = Console(file=StringIO(), width=80)
    heading = LeftHeading('h2')
    heading.text = Text('Test Heading')

    result = list(heading.__rich_console__(console, console.options))
    assert len(result) == 1
    assert isinstance(result[0], Text)
    assert '## Test Heading' in str(result[0])


def test_discover_models_interactive_selection(mocker: MockerFixture, capfd: CaptureFixture[str]):
    """Test interactive model selection in discover models."""
    mocker.patch('pydantic_ai._cli.discover_local_models', return_value=['model-a', 'model-b'])
    mocker.patch('builtins.input', return_value='1')
    mocker.patch('pydantic_ai._cli.save_discovery_config')
    mocker.patch('pydantic_ai._cli.run_chat', return_value=0)

    with patch('pydantic_ai._cli.load_discovery_config', return_value=None):
        assert cli(['-d', 'http://localhost:1234/v1']) == 0

    output = capfd.readouterr().out
    assert 'Found 2 models:' in output
    assert '1. model-a' in output
    assert '2. model-b' in output
    assert 'Selected model: model-a' in output


def test_discover_models_invalid_selection(mocker: MockerFixture, capfd: CaptureFixture[str]):
    """Test invalid model selection in discover models."""
    mocker.patch('pydantic_ai._cli.discover_local_models', return_value=['model-a', 'model-b'])
    mocker.patch('builtins.input', return_value='5')  # Invalid selection

    with patch('pydantic_ai._cli.load_discovery_config', return_value=None):
        assert cli(['-d', 'http://localhost:1234/v1']) == 1

    output = capfd.readouterr().out
    assert 'Invalid selection' in output


def test_discover_models_keyboard_interrupt(mocker: MockerFixture, capfd: CaptureFixture[str]):
    """Test keyboard interrupt during model selection."""
    mocker.patch('pydantic_ai._cli.discover_local_models', return_value=['model-a', 'model-b'])
    mocker.patch('builtins.input', side_effect=KeyboardInterrupt())

    with patch('pydantic_ai._cli.load_discovery_config', return_value=None):
        assert cli(['-d', 'http://localhost:1234/v1']) == 0

    output = capfd.readouterr().out
    assert 'Exiting...' in output


def test_discover_models_default_selection(mocker: MockerFixture, capfd: CaptureFixture[str]):
    """Test default model selection when pressing Enter."""
    mock_config = ('http://localhost:1234/v1', 'model-a', '2024-01-01T12:00:00')
    mocker.patch('pydantic_ai._cli.discover_local_models', return_value=['model-a', 'model-b'])
    mocker.patch('builtins.input', return_value='')  # Empty input (Enter)
    mocker.patch('pydantic_ai._cli.save_discovery_config')
    mocker.patch('pydantic_ai._cli.run_chat', return_value=0)

    with patch('pydantic_ai._cli.load_discovery_config', return_value=mock_config):
        assert cli(['-d']) == 0

    output = capfd.readouterr().out
    assert 'Using default: model-a' in output


def test_discover_models_exit_no_default(mocker: MockerFixture, capfd: CaptureFixture[str]):
    """Test exiting when no default and pressing Enter."""
    mocker.patch('pydantic_ai._cli.discover_local_models', return_value=['model-a', 'model-b'])
    mocker.patch('builtins.input', return_value='')  # Empty input (Enter)

    with patch('pydantic_ai._cli.load_discovery_config', return_value=None):
        assert cli(['-d', 'http://localhost:1234/v1']) == 0

    output = capfd.readouterr().out
    assert 'Exiting...' in output


def test_discover_models_value_error(mocker: MockerFixture, capfd: CaptureFixture[str]):
    """Test ValueError during model selection."""
    mocker.patch('pydantic_ai._cli.discover_local_models', return_value=['model-a', 'model-b'])
    mocker.patch('builtins.input', return_value='abc')  # Non-numeric input

    with patch('pydantic_ai._cli.load_discovery_config', return_value=None):
        assert cli(['-d', 'http://localhost:1234/v1']) == 1

    output = capfd.readouterr().out
    assert 'Invalid input. Please enter a number.' in output


def test_list_models_exception_path(capfd: CaptureFixture[str]):
    """Test that specifically triggers the exception path in list-models."""
    # Return data that will naturally cause datetime.fromisoformat to raise ValueError
    # Using a completely invalid ISO format that will definitely fail
    mock_config = ('http://localhost:1234/v1', 'test-model', 'not-a-date-at-all')

    with patch('pydantic_ai._cli.load_discovery_config', return_value=mock_config):
        result = cli(['--list-models'])
        assert result == 0
        output = capfd.readouterr().out
        # The exception path should show the discovery info without timestamp
        assert 'Last discovery: test-model at http://localhost:1234/v1' in output
        assert 'Use "clai -d" to reconnect to last endpoint' in output
        # Should NOT contain a timestamp since the exception was caught
        discovery_line = output.split('Last discovery:')[1].split('\n')[0]
        assert '(' not in discovery_line  # No timestamp in parentheses


def test_list_models_no_discovery_config(capfd: CaptureFixture[str]):
    """Test --list-models when there is no discovery config (branch coverage)."""
    # Patch load_discovery_config to return None
    with patch('pydantic_ai._cli.load_discovery_config', return_value=None):
        assert cli(['--list-models']) == 0
        output = capfd.readouterr().out
        # Should NOT contain 'Last discovery'
        assert 'Last discovery' not in output


def test_list_models_discovery_config_empty_endpoint(capfd: CaptureFixture[str]):
    """Test --list-models when discovery config has empty endpoint (branch coverage)."""
    with patch(
        'pydantic_ai._cli.load_discovery_config',
        return_value=('', 'model', '2024-01-01T12:00:00'),
    ):
        assert cli(['--list-models']) == 0
        output = capfd.readouterr().out
        assert 'Last discovery' not in output


def test_discover_models_usererror(mocker: MockerFixture, capfd: CaptureFixture[str]):
    """Test error handling when discover_local_models raises UserError."""
    from pydantic_ai._cli import UserError

    # Patch discover_local_models to raise UserError
    mocker.patch('pydantic_ai._cli.discover_local_models', side_effect=UserError('fail!'))
    # Patch load_discovery_config to return None (so endpoint is used directly)
    with patch('pydantic_ai._cli.load_discovery_config', return_value=None):
        assert cli(['-d', 'http://localhost:1234/v1']) == 1
        output = capfd.readouterr().out
        assert 'Error: fail!' in output
