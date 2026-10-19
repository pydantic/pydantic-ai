"""Closing a menu without changes leaves no blank lines or no-op chatter in the transcript."""

import io
from collections.abc import Sequence
from pathlib import Path

import pytest
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
from pydantic_clai2 import chat
from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.commands import Command, is_silent
from pydantic_clai2.config import Settings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.ui.menus.menu_worker import run_worker
from tests.clai2.test_app_edges import inputs


@pytest.mark.parametrize(
    ('result', 'silent'),
    [
        ('', True),
        ('\n\n', True),
        ('API key unchanged.', True),
        ('GitHub settings unchanged.\n', True),
        ('Token saved.', False),
        ('Saved.\nGitHub settings unchanged.', False),
        ('Saved.\rGitHub settings unchanged.', False),
        ('unchanged.', False),
    ],
)
def test_is_silent(result: str, silent: bool) -> None:
    assert is_silent(result) is silent


class _Commands(AbstractCapability[None]):
    def get_commands(self, context: CommandContext) -> Sequence[Command]:
        async def closed(args: list[str]) -> str:
            return await run_worker(lambda: '')

        def broken(args: list[str]) -> str:
            raise ValueError('Token rejected.')

        return (
            Command(name='mark', description='', handler=lambda args: f'MARK {args[0]}'),
            Command(name='closed', description='', handler=closed),
            Command(name='same', description='', handler=lambda args: 'GitHub settings unchanged.'),
            Command(name='saved', description='', handler=lambda args: 'Token saved.'),
            Command(name='broken', description='', handler=broken),
        )


async def test_no_op_commands_print_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs(monkeypatch, ['/mark one', '/closed', '/same', '/mark two', '/saved', '/broken', '/exit'])
    output = io.StringIO()
    await chat(
        Agent(TestModel()),
        deps=None,
        plugins=[_Commands()],
        settings=Settings(model='test'),
        console=Console(file=output, width=80),
        store=SettingsStore(tmp_path / 'config.db'),
    )
    text = output.getvalue()
    # Each input still gets its one separating blank line, but silent results add nothing more.
    assert text[text.index('MARK one') :] == (
        'MARK one\n\n\n\n\nMARK two\n\n\nToken saved.\n\n\nToken rejected.\n\n\nGoodbye.\n\n'
    )
