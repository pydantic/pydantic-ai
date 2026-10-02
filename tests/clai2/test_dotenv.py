"""Load project environment variables before CLAI startup consumes them."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import pydantic_clai2.cli._cli
from pydantic_clai2.__main__ import main
from pydantic_clai2.ui.rendering.splash import Splash


@pytest.fixture(autouse=True)
def isolated_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore variables added by dotenv, not just those set through monkeypatch."""
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    for name in ('CLAI_DOTENV_TEST', 'CLAI_DOTENV_PARENT_ONLY', 'CLAI_NO_SPLASH', 'PYTHON_DOTENV_DISABLED'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('PYDANTIC_AI_NO_BANNER', '1')


@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('existing', [None, '', 'from shell'])
def test_startup_loads_nearest_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, nested: bool, existing: str | None
) -> None:
    project = tmp_path / 'project'
    directory = project / 'nested' if nested else project
    directory.mkdir(parents=True)
    (tmp_path / '.env').write_text('CLAI_DOTENV_PARENT_ONLY=ignored\n')
    (project / '.env').write_text("CLAI_DOTENV_TEST='from dotenv'\nCLAI_NO_SPLASH=1\n")
    monkeypatch.chdir(directory)
    if existing is not None:
        monkeypatch.setenv('CLAI_DOTENV_TEST', existing)
    monkeypatch.setattr(sys, 'argv', ['clai2'])
    observed: list[str | None] = []

    def start(splash: Splash) -> None:
        observed.append(os.getenv('CLAI_DOTENV_TEST'))
        assert os.environ['CLAI_NO_SPLASH'] == '1'

    def run(*, splash: Splash | None = None) -> None:
        observed.append(os.getenv('CLAI_DOTENV_TEST'))
        assert 'CLAI_DOTENV_PARENT_ONLY' not in os.environ

    monkeypatch.setattr(Splash, 'start', start)
    monkeypatch.setattr(pydantic_clai2.cli._cli, 'run', run)
    main()
    expected = 'from dotenv' if existing is None else existing
    assert observed == [expected, expected]


@pytest.mark.parametrize('state', ['missing', 'empty', 'disabled'])
def test_startup_without_dotenv_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    if state == 'empty':
        (tmp_path / '.env').write_text('')
    elif state == 'disabled':
        (tmp_path / '.env').write_text('CLAI_DOTENV_TEST=ignored\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('PYTHON_DOTENV_DISABLED', '1' if state == 'disabled' else '0')
    monkeypatch.setattr(sys, 'argv', ['clai2', 'config'])
    observed: list[str | None] = []

    def run(*, splash: Splash | None = None) -> None:
        observed.append(os.getenv('CLAI_DOTENV_TEST'))

    monkeypatch.setattr(pydantic_clai2.cli._cli, 'run', run)
    main()
    assert observed == [None]


def test_dotenv_loaded_before_agent_import(tmp_path: Path) -> None:
    (tmp_path / '.env').write_text('CLAI_DOTENV_TEST=loaded before import\n')
    (tmp_path / 'dotenv_agent.py').write_text(
        'import os\n'
        'from pydantic_ai import Agent\n'
        'from pydantic_ai.models.test import TestModel\n'
        "agent = Agent(TestModel(custom_output_text=os.environ['CLAI_DOTENV_TEST']))\n"
    )
    result = subprocess.run(
        [sys.executable, '-m', 'pydantic_clai2', '--agent', 'dotenv_agent:agent', '-p', 'hello'],
        cwd=tmp_path,
        env=os.environ,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'loaded before import'
