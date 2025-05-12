from io import StringIO
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
