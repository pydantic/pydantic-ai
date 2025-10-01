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

from .conftest import TestEnv, try_import

with try_import() as imports_successful:
    from openai import OpenAIError
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.shortcuts import PromptSession

    from pydantic_ai._cli import cli, cli_agent, handle_slash_command
    from pydantic_ai.models.openai import OpenAIChatModel

pytestmark = pytest.mark.skipif(not imports_successful(), reason='install cli extras to run cli tests')


def test_cli_version(capfd: CaptureFixture[str]):
    assert cli(['--version']) == 0
    assert capfd.readouterr().out.startswith('pai - Pydantic AI CLI')


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
    assert 'is not an Agent' in capfd.readouterr().out


def test_agent_flag_bad_module_variable_path(capfd: CaptureFixture[str], mocker: MockerFixture, env: TestEnv):
    assert cli(['--agent', 'bad_path', 'hello']) == 1
    assert 'Agent must be specified in "module:variable" format' in capfd.readouterr().out


def test_list_models(capfd: CaptureFixture[str]):
    assert cli(['--list-models']) == 0
    output = capfd.readouterr().out.splitlines()
    assert output[:3] == snapshot([IsStr(regex='pai - Pydantic AI CLI .*'), '', 'Available models:'])

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
        'heroku',
        'moonshotai',
        'grok',
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
    assert len(mock_clipboard) == 0

    messages: list[ModelMessage] = [ModelResponse(parts=[TextPart(''), ToolCallPart('foo', '{}')])]
    io = StringIO()
    assert handle_slash_command('/cp', messages, True, Console(file=io), 'default') == (None, True)
    assert io.getvalue() == snapshot('No text content to copy.\n')
    assert len(mock_clipboard) == 0

    messages: list[ModelMessage] = [ModelResponse(parts=[TextPart('hello'), ToolCallPart('foo', '{}')])]
    io = StringIO()
    assert handle_slash_command('/cp', messages, True, Console(file=io), 'default') == (None, True)
    assert io.getvalue() == snapshot('Copied last output to clipboard.\n')
    assert len(mock_clipboard) == 1
    assert mock_clipboard[0] == snapshot('hello')


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
        message_history=None,
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
    )
