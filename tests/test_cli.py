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
            'usage: pai [-h] [--model [MODEL]] [--base-url '
            'BASE_URL] [--api-key API_KEY]',
            '           [--no-stream] [--version]',
            '           [prompt]',
            '',
            IsStr(),
            '',
            'Special prompt:',
            '* `/exit` - exit the interactive mode',
            '* `/markdown` - show the last markdown output of the last question',
            '* `/multiline` - toggle multiline mode',
            '',
            'positional arguments:',
            '  prompt               AI Prompt, if omitted fall into interactive mode',
            '',
            IsStr(),
            '  -h, --help           show this help message and exit',
            '  --model [MODEL]      Model to use, it should be "<provider>:<model>" e.g. "openai:gpt-4o". If omitted it will default to "openai:gpt-4o"',
            '  --base-url BASE_URL  Base url to use with model',
            '  --api-key API_KEY    api key (or set in env, see documentation)',
            '  --no-stream          Whether to stream responses from OpenAI',
            '  --version            Show version and exit',
        ]
    )
