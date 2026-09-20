"""Untrusted completion text remains data in every editor/transcript projection."""

import io
from pathlib import Path

import anyio
import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from rich.console import Console

from pydantic_clai2 import chat
from pydantic_clai2.settings_store import SettingsStore


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


async def test_completion_control_bytes_are_not_executed_by_preview_or_echo(tmp_path: Path) -> None:
    ready, handled = anyio.Event(), anyio.Event()
    payload = 'file\x1b]52;c;YQ==\x07'

    class Output(io.StringIO):
        def write(self, text: str) -> int:
            if r'\x1b]52;' in text:
                ready.set()
            if 'SAFE_COMMAND_HANDLED' in text:
                handled.set()
            return super().write(text)

    store = SettingsStore(tmp_path / 'config.db')
    store.plugins_dir.mkdir()
    (store.plugins_dir / 'unsafe_completion.py').write_text(
        'from pydantic_clai2.commands import Command\n'
        'def activate(host):\n'
        f'    payload = {payload!r}\n'
        '    def handle(args):\n'
        '        assert args == [payload]\n'
        "        return 'SAFE_COMMAND_HANDLED'\n"
        "    host.commands.register(Command(name='unsafe', description='Completion test', "
        'handler=handle, complete=lambda args: [payload]))\n'
    )
    output = Output()
    done = anyio.Event()

    async def run() -> None:
        await chat(
            Agent(TestModel()), deps=None, console=Console(file=output, force_terminal=True, width=120), store=store
        )
        done.set()

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(run)
            pipe.send_text('/unsafe ')
            await ready.wait()
            pipe.send_text('\t\r')
            await handled.wait()
            pipe.send_text('/exit\r')
            await done.wait()
    assert '\x1b]52;' not in output.getvalue()
    assert '\x07' not in output.getvalue()
    assert r'> /unsafe file\x1b]52;c;YQ==\x07' in output.getvalue()
