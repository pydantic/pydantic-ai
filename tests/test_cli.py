import sys
import types
from collections.abc import Sequence
from io import StringIO
from typing import Any, Callable

import pytest
from click.testing import CliRunner
from dirty_equals import IsInstance, IsStr
from inline_snapshot import snapshot
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

    from pydantic_ai._cli import cli_agent, handle_slash_command
    from pydantic_ai.models.openai import OpenAIModel

pytestmark = pytest.mark.skipif(not imports_successful(), reason='install cli extras to run cli tests')


class CliTester:
    """Tester for our click-based CLI to help us test it."""

    def __init__(self):
        self._runner = CliRunner()

        from pydantic_ai._cli import cli

        self._cli = cli

    def __call__(self, args: Sequence[str]):
        return self._runner.invoke(self._cli, args)


@pytest.fixture
def cli(env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')

    cli = CliTester()
    yield cli


def test_cli_version(cli: CliTester):
    result = cli(['--version'])  # version now becomes a command, rather than an option that triggers the command

    assert 'pai - PydanticAI CLI' in result.output
    assert result.exit_code == 0


def test_invalid_model(cli: CliTester):
    result = cli(['--model', 'potato'])

    assert "Error: Invalid value for '-m' / '--model': 'potato' is not one of" in result.output
    assert result.exit_code == 2  # Raised by click itself


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
    cli: CliTester,
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
    result = cli(['--agent', 'test_module:custom_agent', 'hello'])

    # Verify the output contains the custom agent message
    assert 'using custom agent test_module:custom_agent' in result.output
    assert result.exit_code == 0

    # Verify ask_agent was called with our custom agent
    mock_ask.assert_called_once()
    assert mock_ask.call_args[0][0] is test_agent


def test_agent_flag_no_model(cli: CliTester, env: TestEnv, create_test_module: Callable[..., None]):
    env.remove('OPENAI_API_KEY')
    test_agent = Agent()
    create_test_module(custom_agent=test_agent)

    msg = 'The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable'
    result = cli(['--agent', 'test_module:custom_agent', 'hello'])

    assert str(result.exception) == msg


def test_agent_flag_set_model(
    cli: CliTester,
    mocker: MockerFixture,
    env: TestEnv,
    create_test_module: Callable[..., None],
):
    env.set('OPENAI_API_KEY', 'xxx')
    env.set('COLUMNS', '150')

    custom_agent = Agent(TestModel(custom_output_text='Hello from custom agent'))
    create_test_module(custom_agent=custom_agent)

    mocker.patch('pydantic_ai._cli.ask_agent')

    result = cli(['--agent', 'test_module:custom_agent', '--model', 'openai:gpt-4o', 'hello'])

    assert 'using custom agent test_module:custom_agent with openai:gpt-4o' in result.output
    assert isinstance(custom_agent.model, OpenAIModel)


def test_agent_flag_non_agent(cli: CliTester, create_test_module: Callable[..., None]):
    test_agent = 'Not an Agent object'
    create_test_module(custom_agent=test_agent)

    result = cli(['--agent', 'test_module:custom_agent', 'hello'])
    assert result.exit_code == 1
    assert 'is not an Agent' in str(result.exception)


def test_agent_flag_bad_module_variable_path(cli: CliTester):
    result = cli(['--agent', 'bad_path', 'hello'])
    assert result.exit_code == 1
    assert 'Agent must be specified in "module:variable" format' in str(result.exception)


def test_list_models(cli: CliTester):
    result = cli(['--list-models'])
    assert result.exit_code == 0

    output = result.output.splitlines()
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
        'heroku',
    )
    models = {line.strip().split(' ')[0] for line in output[3:]}
    for provider in providers:
        models = models - {model for model in models if model.startswith(provider)}
    assert models == set(), models


def test_cli_prompt(cli: CliTester, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    with cli_agent.override(model=TestModel(custom_output_text='# result\n\n```py\nx = 1\n```')):
        result = cli(['hello'])

        assert result.exit_code == 0
        assert result.output.splitlines() == snapshot([IsStr(), '# result', '', 'py', 'x = 1', '/py'])

        result = cli(['--no-stream', 'hello'])
        assert result.exit_code == 0
        assert result.output.splitlines() == snapshot([IsStr(), '# result', '', 'py', 'x = 1', '/py'])


def test_chat(cli: CliTester, mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    with create_pipe_input() as inp:
        inp.send_text('\n')
        inp.send_text('hello\n')
        inp.send_text('/markdown\n')
        inp.send_text('/exit\n')
        session = PromptSession[Any](input=inp, output=DummyOutput())

        mocker.patch('pydantic_ai._cli.PromptSession', return_value=session)
        model = TestModel(custom_output_text='goodbye')

        with cli_agent.override(model=model):
            result = cli([])
            assert result.exit_code == 0

        assert result.output.splitlines() == snapshot(
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


def test_code_theme_unset(cli: CliTester, mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')
    cli([])
    mock_run_chat.assert_awaited_once_with(True, IsInstance(Agent), IsInstance(Console), 'monokai', 'pai')


def test_code_theme_light(cli: CliTester, mocker: MockerFixture, env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    mock_run_chat = mocker.patch('pydantic_ai._cli.run_chat')
    cli(['--code-theme=light'])
    mock_run_chat.assert_awaited_once_with(True, IsInstance(Agent), IsInstance(Console), 'default', 'pai')


def test_code_theme_dark(cli: CliTester, mocker: MockerFixture, env: TestEnv):
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
