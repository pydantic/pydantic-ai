"""Prompt recall survives terminal sessions without saving model messages."""

import io
import os
from pathlib import Path

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from rich.console import Console

from pydantic_clai2 import chat
from pydantic_clai2.input_history import input_history
from pydantic_clai2.settings_store import SettingsStore


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


async def test_history_survives_reopening(tmp_path: Path) -> None:
    path = tmp_path / 'nested' / 'input-history'
    original = input_history(path)
    original.append_string('first line\nsecond line')
    original.append_string('/help')
    reopened = input_history(path)
    assert [text async for text in reopened.load()] == ['/help', 'first line\nsecond line']
    if os.name != 'nt':
        assert path.stat().st_mode & 0o777 == 0o600


async def test_chat_reuses_input_history(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    path = tmp_path / 'input-history'
    input_history(path).append_string('/help')
    output = io.StringIO()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text('\x1b[A\n/exit\n')
        await chat(Agent(TestModel()), deps=None, console=Console(file=output), store=store)
    assert 'Show commands' in output.getvalue()
    assert '/exit' in [text async for text in input_history(path).load()]
