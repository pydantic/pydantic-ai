"""Check startup import boundaries in fresh interpreters, without timing thresholds."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


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
from pydantic_clai2.__main__ import main
models.ALLOW_MODEL_REQUESTS = False
async def read(self, *args, **kwargs):
    completions = self.completer.get_completions(Document('/login ', 7), CompleteEvent())
    assert {item.text for item in completions} == {'openai-codex', 'github-copilot'}
    for name in (
        'openai', 'anthropic', 'pydantic_clai2.auth', 'pydantic_clai2.openrouter',
        'pydantic_clai2.vllm', 'pydantic_clai2.github_copilot', 'pydantic_clai2.model_menu',
        'pydantic_clai2.model_settings', 'pydantic_clai2.headless',
    ):
        assert name not in sys.modules, name
    print('PROMPT_READY')
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
