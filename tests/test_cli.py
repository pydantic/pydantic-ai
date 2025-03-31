import pytest
from dirty_equals import IsStr
from inline_snapshot import snapshot
from pytest import CaptureFixture

from .conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai._cli import cli

pytestmark = pytest.mark.skipif(not imports_successful(), reason='install cli extras to run cli tests')


def test_cli_version(capfd: CaptureFixture[str]):
    assert cli(['--version']) == 0
    assert capfd.readouterr().out.startswith('pai - PydanticAI CLI')


def test_cli_help(capfd: CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc:
        cli(['--help'])
    assert exc.value.code == 0

    assert capfd.readouterr().out.splitlines() == snapshot(
        [
            'usage: pai [-h] [--model [MODEL]] [--list-models] [--no-stream] [--version] [prompt]',
            '',
            IsStr(),
            '',
            'Special prompt:',
            '* `/exit` - exit the interactive mode',
            '* `/markdown` - show the last markdown output of the last question',
            '* `/multiline` - toggle multiline mode',
            '',
            'positional arguments:',
            '  prompt           AI Prompt, if omitted fall into interactive mode',
            '',
            IsStr(),
            '  -h, --help       show this help message and exit',
            '  --model [MODEL]  Model to use, it should be "<provider>:<model>" e.g. "openai:gpt-4o". If omitted it will default to "openai:gpt-4o"',
            '  --list-models    List all available models and exit',
            '  --no-stream      Whether to stream responses from OpenAI',
            '  --version        Show version and exit',
        ]
    )


def test_invalid_model(capfd: CaptureFixture[str]):
    assert cli(['--model', 'invalid_model']) == 1
    assert capfd.readouterr().out.splitlines() == snapshot([IsStr(), 'Invalid model "invalid_model"'])


def test_list_models(capfd: CaptureFixture[str]):
    assert cli(['--list-models']) == 0
    output = capfd.readouterr().out.splitlines()
    assert output[:2] == snapshot(['pai - PydanticAI CLI v0.0.46', 'Available models:'])

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
