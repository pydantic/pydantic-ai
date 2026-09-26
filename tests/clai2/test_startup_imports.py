"""Check startup import boundaries in fresh interpreters, without timing thresholds."""

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pydantic_clai2 import warm_imports


@pytest.mark.parametrize(
    ('args', 'exit_code'),
    [(['--help'], 0), (['--unknown-option'], 2), (['-p', 'hello', '--resume'], 2)],
)
def test_argument_parsing_does_not_load_agent(args: list[str], exit_code: int, tmp_path: Path) -> None:
    script = """
import sys
from pydantic_clai2.__main__ import main
try:
    main()
finally:
    assert 'pydantic_ai' not in sys.modules
    assert 'pydantic_clai2._app' not in sys.modules
    assert 'pydantic_clai2.headless' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, '-c', script, *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == exit_code, result.stderr
    assert 'AssertionError' not in result.stderr


def test_prompt_ready_without_provider_or_model_menu_imports(tmp_path: Path) -> None:
    script = """
import sys
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from pydantic_ai import models
from pydantic_clai2 import warm_imports
from pydantic_clai2.__main__ import main
models.ALLOW_MODEL_REQUESTS = False
requested = []
start = warm_imports.start
def hold_warming(modules=warm_imports.FIRST_USE_MODULES):
    requested.append(modules)
    return start(())
warm_imports.start = hold_warming
async def read(self, *args, **kwargs):
    completions = self.completer.get_completions(Document('/login ', 7), CompleteEvent())
    assert {item.text for item in completions} == {'openai-codex', 'github-copilot'}
    warmed = ('openai', 'anthropic', *warm_imports.FIRST_USE_MODULES)
    for name in (*warmed, 'pydantic_clai2.headless'):
        assert name not in sys.modules, name
    assert requested == [warm_imports.FIRST_USE_MODULES]
    print('PROMPT_READY')
    start().join()
    for name in warmed:
        assert name in sys.modules, name
    assert 'pydantic_clai2.headless' not in sys.modules
    print('WARMED')
    return '/exit'
PromptSession.prompt_async = read
main()
"""
    result = subprocess.run(
        [sys.executable, '-c', script, '-m', 'test'],
        cwd=tmp_path,
        env=dict(os.environ, CLAI_NO_SPLASH='1'),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert 'PROMPT_READY' in result.stdout
    assert 'WARMED' in result.stdout


def test_warming_failure_is_logged_not_raised(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger=warm_imports.__name__):
        warm_imports.start(('pydantic_clai2._no_such_module', 'pydantic_clai2.vllm')).join()
    assert 'pydantic_clai2.vllm' in sys.modules
    assert [record.getMessage() for record in caplog.records] == ['Could not warm pydantic_clai2._no_such_module']
